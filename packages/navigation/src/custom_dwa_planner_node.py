#!/usr/bin/env python3
"""
Custom DWA (Dynamic Window Approach) Planner Node - Lightweight local planning without costmap_2d.

This node implements a simplified but effective DWA algorithm tailored for Duckiebot constraints:
- Mono camera means limited obstacle detection range
- Small size means we can use discrete trajectory sampling
- Pure pursuit tracking of global path

Algorithm Overview:
1. Sample (v, ω) velocity pairs from kinematic limits
2. For each velocity pair, simulate trajectory over prediction horizon
3. Evaluate cost function (goal/path following + obstacle avoidance + velocity reward)
4. Select trajectory with lowest cost
5. Publish the corresponding velocity command

Cost Function Components:
- Path cost: Distance from trajectory to global path (enforces lane following)
- Velocity cost: Encourage higher speeds when safe
- Obstacle cost: Repulsive field from obstacles (infinity if collision)
- Heading cost: Smooth heading changes
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import namedtuple
from geometry_msgs.msg import (
    PoseStamped, PointStamped, Twist, Vector3Stamped
)
from nav_msgs.msg import Path, Odometry
from duckietown_msgs.msg import Twist2DStamped
from visualization_msgs.msg import Marker, MarkerArray
from duckietown.dtros import DTROS, NodeType, TopicType
import threading


@dataclass
class TrajectoryPoint:
    """Single point in a simulated trajectory."""
    x: float
    y: float
    theta: float
    v: float
    omega: float


@dataclass
class TrajectoryEvaluation:
    """Result of evaluating a single trajectory."""
    velocity: Tuple[float, float]  # (v, omega)
    trajectory: List[TrajectoryPoint]
    path_cost: float
    obstacle_cost: float
    velocity_reward: float
    heading_cost: float
    total_cost: float


class CustomDWAPlanner:
    """
    Core DWA (Dynamic Window Approach) planner implementation without costmap_2d.
    
    The Dynamic Window Approach constrains velocity sampling to velocities reachable
    from the current velocity given acceleration limits. This ensures smooth, realistic
    trajectories while enabling dynamic obstacle avoidance.
    
    Key Features:
    - Dynamic velocity sampling based on current velocity and acceleration limits
    - Trajectory simulation under differential drive kinematics
    - Multi-objective cost function (path following + obstacle avoidance + smoothness)
    - Continuous replanning at fixed frequency for reactive behavior
    
    Responsibilities:
    - Sample velocity candidates from robot's kinematic and dynamic limits
    - Simulate trajectories for each velocity
    - Evaluate trajectories against cost function
    - Select best trajectory for execution
    """
    
    def __init__(self,
                 max_velocity: float = 0.4,  # m/s
                 max_omega: float = 1.5,  # rad/s
                 max_accel_v: float = 0.5,  # m/s^2 (acceleration limit)
                 max_accel_omega: float = 2.0,  # rad/s^2 (angular acceleration limit)
                 velocity_resolution: int = 5,  # Number of velocity samples
                 omega_resolution: int = 9,  # Number of angular velocity samples
                 prediction_horizon: float = 1.0,  # Simulation time
                 simulation_dt: float = 0.1):  # Simulation timestep
        """
        Args:
            max_velocity: Maximum linear velocity (m/s). Robot kinematic limit.
            max_omega: Maximum angular velocity (rad/s). Robot kinematic limit.
            max_accel_v: Maximum linear acceleration (m/s²). Defines dynamic window.
                         Reachable velocities: [v_current - max_accel_v*dt, v_current + max_accel_v*dt]
                         Smaller values = smoother but less reactive. Larger values = more responsive.
            max_accel_omega: Maximum angular acceleration (rad/s²). Defines angular dynamic window.
                            Same principle as max_accel_v but for rotation.
            velocity_resolution: Number of linear velocity samples to evaluate (typical: 3-7).
                                Higher = better solution quality but slower computation.
            omega_resolution: Number of angular velocity samples to evaluate (typical: 5-15).
            prediction_horizon: How far into future to simulate (seconds).
                               Shorter = more reactive to nearby obstacles.
                               Longer = more predictive but slower computation.
            simulation_dt: Timestep for trajectory simulation (seconds).
                          Smaller = more accurate simulation but slower.
                          Larger = faster but may skip thin obstacles.
        """
        self.max_velocity = max_velocity
        self.max_omega = max_omega
        self.max_accel_v = max_accel_v
        self.max_accel_omega = max_accel_omega
        self.velocity_resolution = velocity_resolution
        self.omega_resolution = omega_resolution
        self.prediction_horizon = prediction_horizon
        self.simulation_dt = simulation_dt
        
        # Cost function weights
        self.weight_path = 1.0  # Following global path
        self.weight_velocity = 0.1  # Encourage movement
        self.weight_obstacle = 10.0  # Avoid obstacles (high penalty)
        self.weight_heading = 0.05  # Smooth heading changes
        
        # Obstacles list (will be updated by external callback)
        self.obstacles = []  # List of (x, y, radius) tuples
        
        # Global path to follow
        self.global_path = None  # Nav_msgs/Path or list of (x, y) points
        self.current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.current_velocity = np.array([0.0, 0.0])  # [v, omega]
        
        # TF2 infrastructure
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    
    def sample_velocity_space(self, current_velocity: np.ndarray) -> List[Tuple[float, float]]:
        """
        Sample (v, omega) pairs from the dynamic window.
        
        The dynamic window constrains reachable velocities based on:
        - Current velocity
        - Acceleration limits
        - Time horizon
        
        This ensures sampled velocities are physically reachable, which is the key
        feature of DWA ("Dynamic" Window Approach).
        
        Args:
            current_velocity: [v, omega] current velocity
        
        Returns:
            List of (v, omega) tuples within dynamic window
        """
        velocities = []
        current_v = current_velocity[0]
        current_omega = current_velocity[1]
        
        # Compute dynamic window bounds for linear velocity
        v_min = max(0, current_v - self.max_accel_v * self.simulation_dt)
        v_max = min(self.max_velocity, current_v + self.max_accel_v * self.simulation_dt)
        
        # Compute dynamic window bounds for angular velocity
        omega_min = max(-self.max_omega, current_omega - self.max_accel_omega * self.simulation_dt)
        omega_max = min(self.max_omega, current_omega + self.max_accel_omega * self.simulation_dt)
        
        # Sample within dynamic window
        v_samples = np.linspace(v_min, v_max, self.velocity_resolution)
        omega_samples = np.linspace(omega_min, omega_max, self.omega_resolution)
        
        for v in v_samples:
            for omega in omega_samples:
                velocities.append((v, omega))
        
        return velocities
    
    def simulate_trajectory(self, v: float, omega: float,
                           start_pose: np.ndarray) -> List[TrajectoryPoint]:
        """
        Simulate a trajectory under constant velocity for prediction_horizon.
        
        Uses simple differential drive kinematics:
        - x_dot = v * cos(theta)
        - y_dot = v * sin(theta)
        - theta_dot = omega
        
        Args:
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)
            start_pose: Initial [x, y, theta]
            
        Returns:
            List of trajectory points
        """
        trajectory = []
        pose = start_pose.copy()
        
        num_steps = int(self.prediction_horizon / self.simulation_dt)
        
        for step in range(num_steps):
            # Update pose using differential drive kinematics
            if abs(omega) > 1e-6:
                # Curved path
                radius = v / omega
                pose[0] += radius * (np.sin(pose[2] + omega * self.simulation_dt) - 
                                    np.sin(pose[2]))
                pose[1] += radius * (-np.cos(pose[2] + omega * self.simulation_dt) + 
                                    np.cos(pose[2]))
            else:
                # Straight path
                pose[0] += v * np.cos(pose[2]) * self.simulation_dt
                pose[1] += v * np.sin(pose[2]) * self.simulation_dt
            
            pose[2] += omega * self.simulation_dt
            pose[2] = np.arctan2(np.sin(pose[2]), np.cos(pose[2]))  # Normalize angle
            
            trajectory.append(TrajectoryPoint(
                x=pose[0],
                y=pose[1],
                theta=pose[2],
                v=v,
                omega=omega
            ))
        
        return trajectory
    
    def evaluate_trajectory(self, velocity: Tuple[float, float],
                           trajectory: List[TrajectoryPoint]) -> TrajectoryEvaluation:
        """
        Evaluate a trajectory using multi-component cost function.
        
        Cost Components:
        1. Path Cost: Distance from trajectory to global path (lane following)
        2. Obstacle Cost: Proximity to obstacles (infinity if collision)
        3. Velocity Reward: Encourage faster speeds
        4. Heading Cost: Prefer smooth heading changes
        
        Args:
            velocity: (v, omega) tuple
            trajectory: Simulated trajectory
            
        Returns:
            TrajectoryEvaluation with cost breakdown
        """
        v, omega = velocity
        
        # ===== Cost 1: Global Path Following =====
        path_cost = self._compute_path_cost(trajectory)
        
        # ===== Cost 2: Obstacle Avoidance =====
        obstacle_cost = self._compute_obstacle_cost(trajectory)
        
        # If collision detected, reject this trajectory
        if obstacle_cost == float('inf'):
            evaluation = TrajectoryEvaluation(
                velocity=velocity,
                trajectory=trajectory,
                path_cost=float('inf'),
                obstacle_cost=obstacle_cost,
                velocity_reward=0.0,
                heading_cost=0.0,
                total_cost=float('inf')
            )
            return evaluation
        
        # ===== Cost 3: Velocity Reward (encourage forward motion) =====
        velocity_reward = -v * 0.1  # Negative because lower cost is better
        
        # ===== Cost 4: Heading Stability =====
        heading_cost = self._compute_heading_cost(trajectory)
        
        # ===== Total Cost =====
        total_cost = (self.weight_path * path_cost +
                     self.weight_obstacle * obstacle_cost +
                     self.weight_velocity * velocity_reward +
                     self.weight_heading * heading_cost)
        
        evaluation = TrajectoryEvaluation(
            velocity=velocity,
            trajectory=trajectory,
            path_cost=path_cost,
            obstacle_cost=obstacle_cost,
            velocity_reward=velocity_reward,
            heading_cost=heading_cost,
            total_cost=total_cost
        )
        
        return evaluation
    
    def _compute_path_cost(self, trajectory: List[TrajectoryPoint]) -> float:
        """
        Compute cost based on deviation from global path.
        
        Strategy: Find closest point on global path for each trajectory point,
        sum distances. Encourages staying on the planned route.
        """
        if self.global_path is None or len(self.global_path) == 0:
            return 0.0  # No path to follow
        
        total_deviation = 0.0
        
        for point in trajectory:
            # Find minimum distance to any point on the global path
            min_distance = float('inf')
            
            for path_point in self.global_path:
                distance = np.sqrt((point.x - path_point[0])**2 + 
                                 (point.y - path_point[1])**2)
                min_distance = min(min_distance, distance)
            
            total_deviation += min_distance
        
        # Average deviation
        return total_deviation / max(len(trajectory), 1)
    
    def _compute_obstacle_cost(self, trajectory: List[TrajectoryPoint]) -> float:
        """
        Compute cost based on obstacle proximity.
        
        Strategy:
        - For each trajectory point, check distance to all obstacles
        - If any point penetrates obstacle (including safety margin), return infinity
        - Otherwise, return minimum distance to any obstacle (lower = higher cost)
        """
        if len(self.obstacles) == 0:
            return 0.0
        
        min_distance_to_obstacle = float('inf')
        
        for point in trajectory:
            trajectory_point = np.array([point.x, point.y])
            
            for obstacle in self.obstacles:
                # obstacle is (x, y, radius)
                obs_pos = np.array([obstacle[0], obstacle[1]])
                obs_radius = obstacle[2]  # Already includes radius + safety_margin
                
                distance = np.linalg.norm(trajectory_point - obs_pos)
                
                # COLLISION CHECK: If within obstacle radius, return infinity
                if distance < obs_radius:
                    return float('inf')
                
                # Track minimum safe distance
                min_distance_to_obstacle = min(min_distance_to_obstacle, distance)
        
        # Cost increases as we get closer to obstacles (inverse square law for smoothness)
        if min_distance_to_obstacle == float('inf'):
            return 0.0
        
        # Use repulsive potential field: cost = 1 / (distance - radius)
        # Add small epsilon to avoid division by zero
        safety_threshold = 0.3  # meters
        if min_distance_to_obstacle < safety_threshold:
            return 1.0 / (min_distance_to_obstacle + 0.01)
        else:
            return 0.0
    
    def _compute_heading_cost(self, trajectory: List[TrajectoryPoint]) -> float:
            """
            Calculates how well the trajectory's final heading aligns with the global path.
            """
            if self.global_path is None or len(self.global_path) < 2:
                return 0.0
                
            # 1. Get the final point of the simulated trajectory
            final_point = trajectory[-1]
            
            # 2. Find the closest waypoint on the global path to this final point
            min_dist = float('inf')
            closest_idx = 0
            for i, path_point in enumerate(self.global_path[:-1]):
                dist = np.hypot(final_point.x - path_point[0], final_point.y - path_point[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            # 3. Calculate the angle of the path at that waypoint
            p1 = self.global_path[closest_idx]
            p2 = self.global_path[closest_idx + 1]
            path_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            # 4. Return the absolute angular difference
            angle_diff = abs(final_point.theta - path_angle)
            # Normalize to [-pi, pi]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            return abs(angle_diff)
    
    def plan(self, current_pose: np.ndarray,
            current_velocity: np.ndarray,
            global_path: List[Tuple[float, float]],
            obstacles: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
        """
        Execute DWA planning: find best velocity for next time step.
        
        Args:
            current_pose: [x, y, theta] in map frame
            current_velocity: [v, omega] current velocity
            global_path: List of [x, y] waypoints to follow
            obstacles: List of (x, y, radius) in map frame
            
        Returns:
            (v, omega) velocity command, or None if no valid trajectory found
        """
        self.current_pose = current_pose
        self.current_velocity = current_velocity
        self.global_path = global_path
        self.obstacles = obstacles
        
        # Step 1: Sample velocity space (constrained by dynamic window)
        velocity_candidates = self.sample_velocity_space(current_velocity)
        
        # Step 2: Simulate and evaluate trajectories
        evaluations = []
        for velocity in velocity_candidates:
            trajectory = self.simulate_trajectory(velocity[0], velocity[1], current_pose)
            evaluation = self.evaluate_trajectory(velocity, trajectory)
            evaluations.append(evaluation)
        
        # Step 3: Find best trajectory (lowest cost)
        valid_trajectories = [e for e in evaluations if e.total_cost != float('inf')]
        
        if not valid_trajectories:
            rospy.logwarn("No valid trajectory found! Using stop command.")
            return (0.0, 0.0)
        
        best_evaluation = min(valid_trajectories, key=lambda e: e.total_cost)
        
        rospy.logdebug(f"Selected velocity: v={best_evaluation.velocity[0]:.3f}, "
                      f"omega={best_evaluation.velocity[1]:.3f}, "
                      f"cost={best_evaluation.total_cost:.3f}")
        
        return best_evaluation.velocity


class CustomDWAPlannerNode(DTROS):
    """ROS node wrapper for custom DWA planner."""
    
    def __init__(self, node_name: str = "custom_dwa_planner_node"):
        super(CustomDWAPlannerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PLANNING
        )
        
        # Get robot vehicle name and construct frame names
        self.veh = rospy.get_param("~veh")
        self.frame_map = f"{self.veh}/map"
        self.frame_odom = f"{self.veh}/odom"
        self.frame_footprint = f"{self.veh}/footprint"
        
        # Load parameters
        max_v = rospy.get_param("~max_velocity", 0.4)
        max_omega = rospy.get_param("~max_omega", 1.5)
        max_accel_v = rospy.get_param("~max_accel_v", 0.5)
        max_accel_omega = rospy.get_param("~max_accel_omega", 2.0)
        v_res = rospy.get_param("~velocity_resolution", 5)
        omega_res = rospy.get_param("~omega_resolution", 9)
        horizon = rospy.get_param("~prediction_horizon", 1.0)
        dt = rospy.get_param("~simulation_dt", 0.1)
        
        # Initialize planner
        self.planner = CustomDWAPlanner(
            max_velocity=max_v,
            max_omega=max_omega,
            max_accel_v=max_accel_v,
            max_accel_omega=max_accel_omega,
            velocity_resolution=v_res,
            omega_resolution=omega_res,
            prediction_horizon=horizon,
            simulation_dt=dt
        )
        
        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # State storage
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.global_path = []
        self.obstacles = []
        self.state_lock = threading.Lock()
        
        # Subscribers
        self.sub_path = rospy.Subscriber(
            "~path_in",
            Path,
            self.cb_path,
            queue_size=1
        )
        
        self.sub_odom = rospy.Subscriber(
            "~odom",
            Odometry,
            self.cb_odom,
            queue_size=1
        )
        
        self.sub_obstacles = rospy.Subscriber(
            "~obstacles",
            MarkerArray,  # From obstacle memory node
            self.cb_obstacles,
            queue_size=1
        )
        
        # Publishers
        # Note: Publishing to ~car_cmd to match lane_controller conventions
        # The motor controller expects this topic name (not ~cmd_vel)
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_best_trajectory = rospy.Publisher(
            "~debug/best_trajectory",
            MarkerArray,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )
        
        # Planning timer (run at 10 Hz)
        self.plan_timer = rospy.Timer(rospy.Duration(0.1), self.plan_step)
        
        rospy.loginfo(f"[{node_name}] Custom DWA Planner initialized")
    
    def cb_path(self, msg: Path):
        """Receive global path from navigation package."""
        with self.state_lock:
            self.global_path = []
            for pose_stamped in msg.poses:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                self.global_path.append((x, y))
        
        rospy.logdebug(f"Received path with {len(self.global_path)} waypoints")
    
    def cb_odom(self, msg: Odometry):
        """
        Receive current odometry pose from deadreckoning node.
        Extracts position and orientation from Odometry message.
        """
        with self.state_lock:
            self.current_pose[0] = msg.pose.pose.position.x
            self.current_pose[1] = msg.pose.pose.position.y
            
            # Extract theta from quaternion
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            
            # Convert quaternion to yaw
            self.current_pose[2] = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )
            
            # Also store current velocities from odometry
            self.current_velocity[0] = msg.twist.twist.linear.x
            self.current_velocity[1] = msg.twist.twist.angular.z
    
    def cb_obstacles(self, msg: MarkerArray):
        """Receive obstacle list from obstacle memory node."""
        obstacles = []
        
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            # Diameter is stored in scale, so radius = scale / 2
            radius = marker.scale.x / 2.0
            obstacles.append((x, y, radius))
        
        with self.state_lock:
            self.obstacles = obstacles
        
        rospy.logdebug(f"Received {len(obstacles)} obstacles")
    
    def plan_step(self, event):
        """Execute one planning step (10 Hz)."""
        if not self.global_path:
            return
            
        with self.state_lock:
            odom_pose = self.current_pose.copy()
            velocity = self.current_velocity.copy()
            path = self.global_path.copy()
            obstacles = self.obstacles.copy()

        # --- NEW: Transform odom_pose to map_pose ---
        try:
            # Get latest map -> odom transform
            trans = self.tf_buffer.lookup_transform(
                self.frame_map, 
                self.frame_odom, 
                rospy.Time(0), # Get the most recent available
                rospy.Duration(0.1)
            )
            
            # Apply 2D transform (translation + rotation)
            import tf.transformations as tr
            q = [trans.transform.rotation.x, trans.transform.rotation.y, 
                 trans.transform.rotation.z, trans.transform.rotation.w]
            _, _, yaw_offset = tr.euler_from_quaternion(q)
            
            map_pose = np.zeros(3)
            # Rotate odom vector and add map translation
            dx = odom_pose[0] * np.cos(yaw_offset) - odom_pose[1] * np.sin(yaw_offset)
            dy = odom_pose[0] * np.sin(yaw_offset) + odom_pose[1] * np.cos(yaw_offset)
            
            map_pose[0] = dx + trans.transform.translation.x
            map_pose[1] = dy + trans.transform.translation.y
            map_pose[2] = odom_pose[2] + yaw_offset
            
        except tf2_ros.TransformException as e:
            rospy.logwarn(f"Waiting for map->odom TF: {e}")
            return
            
        # Run planner using the MAP pose, not the odom pose
        best_velocity = self.planner.plan(map_pose, velocity, path, obstacles)
        
        if best_velocity is None:
            best_velocity = (0.0, 0.0)
        
        # Publish command
        self._publish_cmd_vel(best_velocity)
    
    def _publish_cmd_vel(self, velocity: Tuple[float, float]):
        """Publish velocity command in Duckiebot format."""
        v, omega = velocity
        
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        
        msg.header.frame_id = self.frame_footprint 
        
        msg.v = v
        msg.omega = omega
        
        self.pub_car_cmd.publish(msg)


if __name__ == '__main__':
    node = CustomDWAPlannerNode(node_name='custom_dwa_planner_node')
    rospy.spin()
