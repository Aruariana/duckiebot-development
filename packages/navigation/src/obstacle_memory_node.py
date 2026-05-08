#!/usr/bin/env python3
"""
Obstacle Memory Node - Maintains a persistent memory of detected obstacles in global coordinates.

This node handles:
1. Receipt of YOLOv5 detections from object_detection_node
2. Ground projection of bounding box centers to local robot frame
3. TF-based transformation to global map frame
4. Temporal filtering with moving average smoothing
5. Obstacle persistence and timeout management

Architecture:
- ObstacleMemory: Core class managing the obstacle list and filtering logic
- ObstacleMemoryNode: ROS node wrapper with publishers/subscribers

The obstacle memory decouples obstacle detection from planning, allowing the planner
to work with a global, persistent map of obstacles even as the robot moves.
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from geometry_msgs.msg import Point as PointMsg, PointStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from duckiebot_msgs.msg import DetectedObstacle, DetectedObject
from duckietown.dtros import DTROS, NodeType, TopicType
import yaml


@dataclass
class SemanticPrior:
    """Stores depth estimation priors for each detected object class."""
    radius: float  # Physical radius of the obstacle in meters
    safety_margin: float  # Additional safety margin around the obstacle
    confidence_threshold: float = 0.5  # Min confidence to consider detection valid


@dataclass
class ObstacleState:
    """Tracks a single obstacle in the global coordinate frame."""
    obstacle_id: int
    class_name: str  # e.g., 'duckie'
    position_history: deque  # Stores last N positions for moving average (max 5)
    last_global_position: np.ndarray  # [x, y] in map frame
    last_update_time: float  # ROS time of last detection
    creation_time: float  # When this obstacle was first created
    radius: float
    safety_margin: float
    visibility_count: int = 0  # How many frames this obstacle has been visible


class ObstacleMemory:
    """
    Core obstacle memory management system.
    
    Responsibilities:
    - Convert pixel detections to local ground coordinates
    - Transform local coordinates to global map frame using TF
    - Maintain obstacle persistence with temporal filtering
    - Handle obstacle lifecycle (creation, update, timeout)
    
    Note: Frame names are robot-namespaced (e.g., robot_name/footprint, robot_name/map)
    """
    
    def __init__(self, semantic_priors: Dict[str, SemanticPrior], 
                 frame_map: str,
                 frame_footprint: str,
                 max_obstacle_age: float = 5.0,
                 moving_avg_window: int = 5,
                 position_update_threshold: float = 0.05):
        """
        Args:
            semantic_priors: Dictionary mapping class names to SemanticPrior objects
            frame_map: Name of the global map frame (e.g., 'robot_name/map')
            frame_footprint: Name of the robot footprint frame (e.g., 'robot_name/footprint')
            max_obstacle_age: Remove obstacles not updated after this duration (seconds)
            moving_avg_window: Number of past positions to use for moving average
            position_update_threshold: Minimum distance to consider as actual new position
        """
        self.semantic_priors = semantic_priors
        self.frame_map = frame_map
        self.frame_footprint = frame_footprint
        self.max_obstacle_age = max_obstacle_age
        self.moving_avg_window = moving_avg_window
        self.position_update_threshold = position_update_threshold
        
        # Dictionary mapping obstacle_id -> ObstacleState
        self.obstacles: Dict[int, ObstacleState] = {}
        self.next_obstacle_id = 0
        
        # TF2 infrastructure
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # For tracking which obstacles were visible this frame
        self.visible_obstacles_this_frame = set()
        
    def add_or_update_obstacle(self, local_position: np.ndarray, 
                              class_name: str,
                              current_time: float) -> int:
        """
        Process a newly detected obstacle.
        
        Logic:
        1. Transform local position to global map coordinates
        2. Check if this is a new obstacle or update to existing one
        3. Apply moving average smoothing if update
        4. Store/update in global memory
        
        Args:
            local_position: [x, y] in robot's local frame (meters)
            class_name: Name of detected class (e.g., 'duckie')
            current_time: ROS timestamp of detection
            
        Returns:
            The obstacle ID (either existing or newly assigned)
        """
        
        # Step 1: Get semantic priors for this class
        if class_name not in self.semantic_priors:
            rospy.logwarn(f"Unknown obstacle class: {class_name}")
            return -1
        
        priors = self.semantic_priors[class_name]
        
        # Step 2: Transform local position to global map frame
        global_position = self._transform_to_global(local_position)
        if global_position is None:
            rospy.logwarn(f"Failed to transform obstacle {class_name} to global frame")
            return -1
        
        # Step 3: Check if this matches an existing nearby obstacle
        matched_obstacle_id = self._find_nearby_obstacle(global_position, class_name)
        
        if matched_obstacle_id is not None:
            # Update existing obstacle
            obstacle = self.obstacles[matched_obstacle_id]
            
            # Apply moving average smoothing
            obstacle.position_history.append(global_position)
            smoothed_position = np.mean(list(obstacle.position_history), axis=0)
            obstacle.last_global_position = smoothed_position
            obstacle.last_update_time = current_time
            obstacle.visibility_count += 1
            
            rospy.logdebug(f"Updated obstacle {matched_obstacle_id} (class: {class_name}) "
                          f"at global pos [{smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}]")
            
            return matched_obstacle_id
        else:
            # Create new obstacle
            obstacle_id = self.next_obstacle_id
            self.next_obstacle_id += 1
            
            position_history = deque([global_position], maxlen=self.moving_avg_window)
            
            self.obstacles[obstacle_id] = ObstacleState(
                obstacle_id=obstacle_id,
                class_name=class_name,
                position_history=position_history,
                last_global_position=global_position,
                last_update_time=current_time,
                creation_time=current_time,
                radius=priors.radius,
                safety_margin=priors.safety_margin,
                visibility_count=1
            )
            
            rospy.loginfo(f"Created new obstacle {obstacle_id} (class: {class_name}) "
                         f"at global pos [{global_position[0]:.3f}, {global_position[1]:.3f}]")
            
            return obstacle_id
    
    def _transform_to_global(self, local_position: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a position from robot's local frame to global map frame using TF.
        
        TF chain expected: {veh}/map -> {veh}/odom -> {veh}/footprint
        We transform from {veh}/footprint (local) to {veh}/map (global).
        
        Args:
            local_position: [x, y] in footprint frame
            
        Returns:
            [x, y] in map frame, or None if transform fails
        """
        try:
            # Create PointStamped in footprint frame
            local_point = PointStamped()
            local_point.header.frame_id = self.frame_footprint
            local_point.header.stamp = rospy.Time(0)  # Use latest available transform
            local_point.point.x = local_position[0]
            local_point.point.y = local_position[1]
            local_point.point.z = 0.0
            
            # Transform to map frame
            global_point = self.tf_buffer.transform(local_point, self.frame_map)
            
            return np.array([global_point.point.x, global_point.point.y])
        
        except tf2_ros.TransformException as e:
            rospy.logwarn(f"TF transform error: {e}")
            return None
    
    def _find_nearby_obstacle(self, position: np.ndarray, 
                             class_name: str,
                             distance_threshold: float = 0.15) -> Optional[int]:
        """
        Check if a detection matches an existing obstacle within distance threshold.
        
        Matching heuristic: If an obstacle of the same class is within distance_threshold,
        assume it's the same obstacle that moved slightly (helps handle YOLO jitter).
        
        Args:
            position: [x, y] position to match
            class_name: Class name of detection
            distance_threshold: Max distance to consider as same obstacle
            
        Returns:
            Obstacle ID if match found, None otherwise
        """
        for obs_id, obstacle in self.obstacles.items():
            if obstacle.class_name != class_name:
                continue
            
            distance = np.linalg.norm(position - obstacle.last_global_position)
            
            if distance < distance_threshold:
                return obs_id
        
        return None
    
    def mark_frame_start(self):
        """Call at the start of each detection frame to reset visibility tracking."""
        self.visible_obstacles_this_frame = set()
    
    def mark_obstacle_visible(self, obstacle_id: int):
        """Record that an obstacle was visible in this frame."""
        self.visible_obstacles_this_frame.add(obstacle_id)
    
    def update_static_positions(self, current_time: float):
        """
        Called after processing all detections in a frame.
        
        Handles:
        1. For obstacles NOT visible this frame: keep their last global position
           (obstacle persistence - they don't disappear from FOV)
        2. Timeout and remove very old obstacles
        """
        obstacles_to_remove = []
        
        for obstacle_id, obstacle in self.obstacles.items():
            age = current_time - obstacle.last_update_time
            
            # Remove if too old and not recently visible
            if age > self.max_obstacle_age:
                rospy.loginfo(f"Removing obstacle {obstacle_id} (age: {age:.2f}s, class: {obstacle.class_name})")
                obstacles_to_remove.append(obstacle_id)
        
        for obstacle_id in obstacles_to_remove:
            del self.obstacles[obstacle_id]
    
    def get_obstacles(self) -> List[ObstacleState]:
        """Return list of all currently tracked obstacles."""
        return list(self.obstacles.values())


class ObstacleMemoryNode(DTROS):
    """ROS node wrapper for obstacle memory system."""
    
    def __init__(self, node_name: str = "obstacle_memory_node"):
        super(ObstacleMemoryNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PLANNING
        )
        
        # Get robot vehicle name and construct frame names
        self.veh = rospy.get_param("~veh")
        self.frame_map = f"{self.veh}/map"
        self.frame_odom = f"{self.veh}/odom"
        self.frame_footprint = f"{self.veh}/footprint"
        
        # Load semantic priors configuration
        self.semantic_priors = self._load_semantic_priors()
        
        # Initialize obstacle memory
        max_age = rospy.get_param("~max_obstacle_age", 5.0)
        window_size = rospy.get_param("~moving_avg_window", 5)
        
        self.obstacle_memory = ObstacleMemory(
            semantic_priors=self.semantic_priors,
            frame_map=self.frame_map,
            frame_footprint=self.frame_footprint,
            max_obstacle_age=max_age,
            moving_avg_window=window_size
        )
        
        # Subscriber for object detection
        self.sub_obstacle = rospy.Subscriber(
            "~detected_obstacle",
            DetectedObstacle,
            self.cb_obstacle_detection,
            queue_size=1
        )
        
        # Publisher for obstacle list (MarkerArray for visualization)
        self.pub_obstacle_markers = rospy.Publisher(
            "~obstacles_markers",
            MarkerArray,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )
        
        # Publisher for raw obstacle list (for planning)
        self.pub_obstacle_list = rospy.Publisher(
            "~obstacles",
            MarkerArray,  # Using MarkerArray as a simple container
            queue_size=1,
            dt_topic_type=TopicType.PLANNING
        )
        
        rospy.loginfo(f"[{node_name}] Obstacle Memory Node initialized")
        
    def _load_semantic_priors(self) -> Dict[str, SemanticPrior]:
        """
        Load semantic priors from ROS parameters.
        
        Expected parameter structure (YAML):
        ~semantic_priors:
          duckie:
            radius: 0.05
            safety_margin: 0.05
            confidence_threshold: 0.5
          cone:
            radius: 0.08
            safety_margin: 0.05
        """
        priors_dict = rospy.get_param("~semantic_priors", {})
        
        semantic_priors = {}
        
        # Default priors if not specified
        if not priors_dict:
            rospy.logwarn("No semantic priors found, using defaults")
            priors_dict = {
                'duckie': {'radius': 0.05, 'safety_margin': 0.05, 'confidence_threshold': 0.5},
            }
        
        for class_name, params in priors_dict.items():
            semantic_priors[class_name] = SemanticPrior(
                radius=params.get('radius', 0.05),
                safety_margin=params.get('safety_margin', 0.05),
                confidence_threshold=params.get('confidence_threshold', 0.5)
            )
        
        rospy.loginfo(f"Loaded semantic priors for classes: {list(semantic_priors.keys())}")
        return semantic_priors
    
    # In ObstacleMemoryNode -> cb_obstacle_detection
    def cb_obstacle_detection(self, msg: DetectedObstacle):
        if not msg.detected or len(msg.objects) == 0:
            return
            
        self.obstacle_memory.mark_frame_start()
        current_time = rospy.get_time()
        
        for obj in msg.objects:
            if obj.object_type == "duckie":
                local_position = np.array([obj.position.x, obj.position.y])
                class_name = 'duckie'
                
                obstacle_id = self.obstacle_memory.add_or_update_obstacle(
                    local_position=local_position,
                    class_name=class_name,
                    current_time=current_time
                )
                
                if obstacle_id >= 0:
                    self.obstacle_memory.mark_obstacle_visible(obstacle_id)
        
        self.obstacle_memory.update_static_positions(current_time)

        self._publish_obstacles()
    
    def _publish_obstacles(self):
        """Publish current obstacle positions as MarkerArray for visualization and planning."""
        marker_array = MarkerArray()
        
        for obstacle in self.obstacle_memory.get_obstacles():
            # Create sphere marker for each obstacle
            marker = Marker()
            marker.header.frame_id = self.frame_map
            marker.header.stamp = rospy.Time.now()
            marker.id = obstacle.obstacle_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = obstacle.last_global_position[0]
            marker.pose.position.y = obstacle.last_global_position[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Size (diameter = 2 * (radius + safety_margin))
            total_radius = obstacle.radius + obstacle.safety_margin
            marker.scale = Vector3(
                x=2.0 * total_radius,
                y=2.0 * total_radius,
                z=0.01  # Thin cylinder representation
            )
            
            # Color based on class (red for duckie)
            if obstacle.class_name == 'duckie':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
        
        self.pub_obstacle_markers.publish(marker_array)
        self.pub_obstacle_list.publish(marker_array)


if __name__ == '__main__':
    node = ObstacleMemoryNode(node_name='obstacle_memory_node')
    rospy.spin()
