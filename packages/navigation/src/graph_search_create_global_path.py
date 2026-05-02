#!/usr/bin/env python3
"""
Graph Search with Move Base Integration
Combines custom graph search (global planner) with move_base + local planner
for obstacle avoidance and smoother trajectories.
"""

import cv2
import os
import rospy
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import tf
import math

from navigation.srv import *
from navigation.generate_duckietown_map import graph_creator
from navigation.graph_search import GraphSearchProblem


class GraphSearchGlobalPath:
    def __init__(self):
        print("Graph Search + Move Base Service Started")

        # ROS parameters
        self.veh = rospy.get_param("~veh")
        self.origin_frame = rospy.get_param("~origin_frame").replace("~", self.veh)
        self.target_frame = rospy.get_param("~target_frame").replace("~", self.veh)
        self.map_name = "map"  # Default map name


        # Loading map
        self.script_dir = os.path.dirname(__file__)
        self.map_path = self.script_dir + "/maps/" + self.map_name
        self.map_img = self.script_dir + "/maps/map.png"

        # Build graph for global planning
        gc = graph_creator()
        self.duckietown_graph = gc.build_graph_from_csv(csv_filename=self.map_name)
        self.duckietown_problem = GraphSearchProblem(self.duckietown_graph, None, None)

        print("Map loaded successfully.\n")

        # Publishers for visualization
        self.image_pub = rospy.Publisher("~map_graph", Image, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("~global_path", Path, queue_size=1)
        self.bridge = CvBridge()

        # TF listener for pose transformations
        self.tf_listener = tf.TransformListener()

        # Service
        self.service = rospy.Service("graph_search", GraphSearch, self.handle_graph_search)

        # Publish initial map
        self._publish_map_image(None, [])

    def handle_graph_search(self, req):
        """Handle graph search request and execute with move_base"""
        
        # Determine source node (use current robot position if not provided)
        if req.source_node:
            source_node = req.source_node
            print(f"Using provided source node: {source_node}")
        else:
            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    self.origin_frame, self.target_frame, rospy.Time(0)
                )
                source_node = self._find_closest_node_to_point(trans[0], trans[1])
                print(f"Using current robot position, closest node: {source_node}")
            except Exception as e:
                print(f"Could not get robot position: {e}")
                return GraphSearchResponse([])
        
        # Determine target node (use coordinates if provided, otherwise use node name)
        if hasattr(req, 'target_x') and hasattr(req, 'target_y') and (req.target_x != 0 or req.target_y != 0):
            target_node = self._find_closest_node_to_point(req.target_x, req.target_y)
            print(f"Finding closest node to click ({req.target_x}, {req.target_y}): {target_node}")
        else:
            target_node = req.target_node
        
        print(f"Request: {source_node} -> {target_node}")

        # Validate nodes
        if (source_node not in self.duckietown_graph) or (target_node not in self.duckietown_graph):
            print("Source or target node do not exist.")
            self._publish_map_image(None, [])
            return GraphSearchResponse([])

        # Run A* search for global plan
        self.duckietown_problem.start = source_node
        self.duckietown_problem.goal = target_node
        path = self.duckietown_problem.astar_search()
        if not path or not path.actions:
            print("No path found")
            self._publish_map_image(None, [])
            return GraphSearchResponse([])

        print(f"Path found with {len(path.actions)} actions")

        waypoints = self._get_waypoints_from_path(path)
        if not waypoints:
                print("Could not extract waypoints from path")
                return

        self._publish_global_path(waypoints)

        # Visualize the solution
        self._publish_map_image(req, path)

        # TODO: Need to implement path execution logic

        return GraphSearchResponse(path.actions)


    def _find_closest_node_to_point(self, x, y):
        """Find the graph node closest to coordinate (x, y)"""
        min_distance = float('inf')
        closest_node = None
        
        try:
            for node_name in self.duckietown_graph._nodes:
                
                node_pos = self.duckietown_graph.node_positions.get(node_name, None)
                if node_pos is None:
                    print(f"Warning: Node {node_name} does not have position information")
                    continue

                distance = math.sqrt((node_pos[0] - x)**2 + (node_pos[1] - y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node_name
            
            if closest_node:
                print(f"Closest node to ({x}, {y}): {closest_node} (distance: {min_distance:.2f})")
            else:
                print(f"Warning: Could not find closest node to ({x}, {y})")
            
            return closest_node
        
        except Exception as e:
            print(f"Error finding closest node: {e}")
            return None

    def _get_waypoints_from_path(self, path):
        """
        Convert graph path to ROS PoseStamped waypoints.
        Extracts node positions from the graph and creates waypoints.
        """
        waypoints = []
        print("Path:",path)
        print(f"Path nodes: {path.path}")
        print(f"Path actions: {path.actions}")

        try:
            # Get node positions from graph
            node_positions = self.duckietown_graph.node_positions
            
            if not node_positions:
                print("Warning: Could not get node positions from graph")
                return []
            
            # path.path contains the sequence of node names from start to goal
            for target_node in path.path:
                # Get position from graph
                if target_node not in node_positions:
                    print(f"Warning: Node {target_node} not found in graph positions")
                    continue
                
                node_pos = node_positions[target_node]
                
                # Extract x, y (z is typically not used in 2D maps)
                if isinstance(node_pos, (tuple, list)):
                    x = float(node_pos[0])
                    y = float(node_pos[1])
                    theta = 0.0  # Default orientation, can be improved
                else:
                    print(f"Warning: Invalid node position format for {target_node}")
                    continue
                
                # Create PoseStamped
                pose = PoseStamped()
                pose.header.frame_id = self.origin_frame
                pose.header.stamp = rospy.Time.now()
                
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                
                # Convert theta to quaternion
                quat = tf.transformations.quaternion_from_euler(0, 0, theta)
                pose.pose.orientation = Quaternion(*quat)
                
                waypoints.append(pose)
                print(f"Added waypoint: {target_node} at ({x:.2f}, {y:.2f})")

        except Exception as e:
            print(f"Error extracting waypoints: {e}")
            import traceback
            traceback.print_exc()

        return waypoints

    def _publish_global_path(self, waypoints):
        """
        Publish waypoints as a ROS Path message for visualization in rviz.
        """
        try:
            if not waypoints:
                print("Warning: No waypoints to publish")
                return
            
            # Create ROS Path message
            ros_path = Path()
            ros_path.header.frame_id = self.origin_frame
            ros_path.header.stamp = rospy.Time.now()
            
            # Add all waypoints (PoseStamped) to the path
            ros_path.poses = waypoints
            
            # Publish the path
            self.path_pub.publish(ros_path)
            print(f"Published global path with {len(waypoints)} poses")
            
        except Exception as e:
            print(f"Error publishing global path: {e}")
            import traceback
            traceback.print_exc()

    def _publish_map_image(self, req, path):
        """Visualize the solution on the map"""
        try:
            if path:
                self.duckietown_graph.draw(
                    self.script_dir,
                    highlight_edges=path.edges() if hasattr(path, 'edges') else None,
                    map_name=self.map_name,
                    highlight_nodes=[req.source_node, req.target_node] if req else None,
                )
            else:
                self.duckietown_graph.draw(self.script_dir, highlight_edges=None, map_name=self.map_name)

            cv_image = cv2.imread(self.map_path + ".png", cv2.IMREAD_COLOR)
            overlay = self._prep_image(cv_image)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(overlay, "bgr8"))

        except Exception as e:
            print(f"Error publishing map image: {e}")

    def _prep_image(self, cv_image):
        """Prepare image for publishing"""
        try:
            # If cv_image is None or empty, return a blank image
            if cv_image is None or cv_image.size == 0:
                print("Warning: cv_image is empty, returning blank image")
                return cv2.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_BGR2BGR)
            
            # Try to load and blend with map image if available
            try:
                if os.path.exists(self.map_img):
                    map_img = cv2.imread(self.map_img, cv2.IMREAD_COLOR)
                    if map_img is not None:
                        # Crop and resize map image to match cv_image
                        map_crop = map_img[16:556, 29:408, :]
                        target_height = min(cv_image.shape[0], 955)
                        target_width = cv_image.shape[1]
                        map_resize = cv2.resize(map_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        
                        # Ensure cv_image matches the resized map
                        cv_image_crop = cv_image[0:target_height, 0:target_width, :]
                        cv_image_crop = 255 - cv_image_crop
                        
                        # Blend images
                        overlay = cv2.addWeighted(cv_image_crop, 0.65, map_resize, 0.35, 0)
                        overlay = cv2.resize(overlay, (0, 0), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
                        overlay = np.clip(overlay * 1.4, 0, 255).astype(np.uint8)
                        return overlay
            except Exception as blend_error:
                print(f"Warning: Could not blend map image: {blend_error}")
            
            # If blending fails, just return the original image
            return cv_image
            
        except Exception as e:
            print(f"Error preparing image: {e}")
            import traceback
            traceback.print_exc()
            return cv_image


if __name__ == "__main__":
    rospy.init_node("graph_search_create_global_path")
    server = GraphSearchGlobalPath()
    print("Starting graph search with global path creation...\n")
    rospy.spin()
