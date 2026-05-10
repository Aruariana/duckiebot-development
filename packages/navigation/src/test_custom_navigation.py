#!/usr/bin/env python3
"""
Test and Example Script for Custom Navigation Stack

This script demonstrates how to test and interact with the obstacle memory
and DWA planner nodes without requiring a physical Duckiebot.

Usage:
    python3 test_custom_navigation.py _veh:=robot_name
    
Or run individual tests:
    python3 test_custom_navigation.py _veh:=robot_name --test obstacle_memory
    python3 test_custom_navigation.py _veh:=robot_name --test dwa_planner
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from duckiebot_msgs.msg import DetectedObstacle, DetectedObject
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sys


class NavigationStackTester:
    """Test harness for custom navigation stack."""
    
    def __init__(self):
        rospy.init_node('navigation_test_node', anonymous=True)
        
        # Get robot vehicle name and construct frame names
        self.veh = rospy.get_param("~veh", "unknown_vehicle")
        self.frame_map = f"{self.veh}/map"
        self.frame_footprint = f"{self.veh}/footprint"
        
        # Publishers for testing
        self.pub_obstacle = rospy.Publisher(
            'object_detection_node/detected_obstacle',
            DetectedObstacle,
            queue_size=1
        )
        
        self.pub_path = rospy.Publisher(
            'graph_search_server_node/path',
            Path,
            queue_size=1
        )
        
        # TF broadcaster for odometry
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscriber for command velocity
        self.sub_cmd_vel = rospy.Subscriber(
            'custom_dwa_planner_node/car_cmd',
            None,  # Will be filled
            self.cb_cmd_vel
        )
        
        self.last_cmd_vel = None
        
    def test_obstacle_memory_basic(self):
        """Test 1: Basic obstacle detection and memory."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 1: Obstacle Memory - Basic Detection")
        rospy.loginfo("=" * 60)
        
        # Publish a single obstacle directly in front
        duckie_obj = DetectedObject()
        duckie_obj.object_type = "duckie"
        duckie_obj.distance = 0.3
        duckie_obj.position.x = 0.3  # 30cm ahead
        duckie_obj.position.y = 0.0  # Centered
        duckie_obj.position.z = 0.0
        duckie_obj.confidence = 0.95
        
        obstacle = DetectedObstacle()
        obstacle.detected = True
        obstacle.objects = [duckie_obj]
        
        rospy.loginfo("Publishing obstacle at (0.3, 0.0) in local frame")
        self.pub_obstacle.publish(obstacle)
        
        rospy.sleep(1.0)
        rospy.loginfo("✓ Test 1 complete. Check RViz for obstacle marker in map frame.")
    
    def test_obstacle_memory_jitter(self):
        """Test 2: YOLO jitter filtering with moving average."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 2: Obstacle Memory - Jitter Filtering")
        rospy.loginfo("=" * 60)
        
        # Publish same obstacle multiple times with small position variations
        # (simulating YOLO detection noise)
        base_x, base_y = 0.3, 0.0
        
        for i in range(5):
            # Add small random noise
            noise_x = np.random.normal(0, 0.02)  # ±2cm noise
            noise_y = np.random.normal(0, 0.02)
            
            duckie_obj = DetectedObject()
            duckie_obj.object_type = "duckie"
            duckie_obj.distance = 0.3
            duckie_obj.position.x = base_x + noise_x
            duckie_obj.position.y = base_y + noise_y
            duckie_obj.position.z = 0.0
            duckie_obj.confidence = 0.95
            
            obstacle = DetectedObstacle()
            obstacle.detected = True
            obstacle.objects = [duckie_obj]
            
            rospy.loginfo(f"  Iteration {i+1}: Publishing at ({base_x + noise_x:.3f}, {base_y + noise_y:.3f})")
            self.pub_obstacle.publish(obstacle)
            
            rospy.sleep(0.2)
        
        rospy.sleep(1.0)
        rospy.loginfo("✓ Test 2 complete. Moving average should smooth the jitter.")
    
    def test_obstacle_memory_persistence(self):
        """Test 3: Obstacle persistence when not re-detected."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 3: Obstacle Memory - Persistence")
        rospy.loginfo("=" * 60)
        
        # Publish obstacle once
        duckie_obj = DetectedObject()
        duckie_obj.object_type = "duckie"
        duckie_obj.distance = 0.3
        duckie_obj.position.x = 0.3
        duckie_obj.position.y = 0.0
        duckie_obj.position.z = 0.0
        duckie_obj.confidence = 0.95
        
        obstacle = DetectedObstacle()
        obstacle.detected = True
        obstacle.objects = [duckie_obj]
        
        rospy.loginfo("Publishing obstacle at (0.3, 0.0)")
        self.pub_obstacle.publish(obstacle)
        
        # Stop publishing (simulates obstacle leaving FOV)
        rospy.loginfo("Stopping detection (obstacle leaves camera FOV)...")
        
        # Publish "not detected" messages for a few seconds
        for i in range(5):
            not_detected = DetectedObstacle()
            not_detected.detected = False
            not_detected.objects = []
            self.pub_obstacle.publish(not_detected)
            rospy.sleep(0.5)
        
        rospy.loginfo("✓ Test 3 complete. Obstacle should persist in memory for ~5 seconds.")
    
    def test_dwa_planner_basic(self):
        """Test 4: DWA planner with simple path."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 4: DWA Planner - Basic Path Following")
        rospy.loginfo("=" * 60)
        
        # Publish simple straight-line path
        path = Path()
        path.header.frame_id = self.frame_map
        path.header.stamp = rospy.Time.now()
        
        # 5 waypoints in a straight line
        for i in range(5):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_map
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = float(i) * 0.5  # 0, 0.5, 1.0, 1.5, 2.0 meters
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        rospy.loginfo("Publishing path with 5 waypoints (0→2m along X-axis)")
        self.pub_path.publish(path)
        
        rospy.sleep(1.0)
        rospy.loginfo("✓ Test 4 complete. DWA should generate forward command.")
    
    def test_dwa_planner_obstacle_avoidance(self):
        """Test 5: DWA planner avoiding obstacle."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 5: DWA Planner - Obstacle Avoidance")
        rospy.loginfo("=" * 60)
        
        # Publish path
        path = Path()
        path.header.frame_id = self.frame_map
        path.header.stamp = rospy.Time.now()
        
        for i in range(5):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_map
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = float(i) * 0.5
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        self.pub_path.publish(path)
        
        # Publish obstacle in the middle of the path
        rospy.sleep(0.5)
        
        duckie_obj = DetectedObject()
        duckie_obj.object_type = "duckie"
        duckie_obj.distance = 0.75
        duckie_obj.position.x = 0.75
        duckie_obj.position.y = 0.0
        duckie_obj.position.z = 0.0
        duckie_obj.confidence = 0.95
        
        obstacle = DetectedObstacle()
        obstacle.detected = True
        obstacle.objects = [duckie_obj]
        
        rospy.loginfo("Publishing obstacle directly on planned path at (0.75, 0.0)")
        self.pub_obstacle.publish(obstacle)
        
        rospy.sleep(2.0)
        rospy.loginfo("✓ Test 5 complete. DWA should generate avoidance maneuver (lateral velocity).")
    
    def test_obstacle_memory_decay(self):
        """Test 6: Obstacle timeout and removal."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("TEST 6: Obstacle Memory - Timeout and Decay")
        rospy.loginfo("=" * 60)
        
        # Publish obstacle
        duckie_obj = DetectedObject()
        duckie_obj.object_type = "duckie"
        duckie_obj.distance = 0.3
        duckie_obj.position.x = 0.3
        duckie_obj.position.y = 0.0
        duckie_obj.position.z = 0.0
        duckie_obj.confidence = 0.95
        
        obstacle = DetectedObstacle()
        obstacle.detected = True
        obstacle.objects = [duckie_obj]
        
        rospy.loginfo("Publishing obstacle...")
        self.pub_obstacle.publish(obstacle)
        
        # Wait for timeout (should be configured as max_obstacle_age, typically 5 seconds)
        rospy.loginfo("Waiting for obstacle timeout (5+ seconds)...")
        for remaining in range(6, 0, -1):
            print(f"  {remaining}...", end=' ', flush=True)
            rospy.sleep(1.0)
        print()
        
        rospy.loginfo("✓ Test 6 complete. Obstacle should be removed after timeout.")
    
    def cb_cmd_vel(self, msg):
        """Callback for velocity commands."""
        self.last_cmd_vel = msg
    
    def print_usage(self):
        """Print usage information."""
        print("""
Custom Navigation Stack - Test Suite
====================================

This test suite verifies the obstacle memory and DWA planner nodes.

Tests:
  1. Obstacle Memory - Basic Detection
     Verify that a single detection appears in the global map
  
  2. Obstacle Memory - Jitter Filtering
     Verify moving average smoothing reduces YOLO noise
  
  3. Obstacle Memory - Persistence
     Verify obstacles stay in memory when not re-detected (FOV exit)
  
  4. DWA Planner - Basic Path Following
     Verify planner generates reasonable forward command for empty path
  
  5. DWA Planner - Obstacle Avoidance
     Verify planner generates lateral command to avoid obstacle on path
  
  6. Obstacle Memory - Timeout and Decay
     Verify old obstacles are forgotten after max_obstacle_age

Requirements:
- Both nodes should be running (roslaunch navigation custom_navigation_stack.launch veh:=robot_name)
- TF tree should be available with {veh}/map → {veh}/footprint
- RViz recommended for visual verification

Usage:
  python3 test_custom_navigation.py          # Run all tests
  python3 test_custom_navigation.py 1        # Run test 1 only
  python3 test_custom_navigation.py 2,4,6    # Run tests 2, 4, 6
        """)


def main():
    """Run tests."""
    tester = NavigationStackTester()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        tester.print_usage()
        return
    
    tests_to_run = [1, 2, 3, 4, 5, 6]
    
    if len(sys.argv) > 1:
        try:
            # Parse test numbers from command line
            tests_str = sys.argv[1]
            if ',' in tests_str:
                tests_to_run = [int(x.strip()) for x in tests_str.split(',')]
            else:
                tests_to_run = [int(tests_str)]
        except ValueError:
            print("ERROR: Invalid test specification")
            tester.print_usage()
            return
    
    rospy.loginfo(f"Running tests: {tests_to_run}")
    
    try:
        if 1 in tests_to_run:
            tester.test_obstacle_memory_basic()
            rospy.sleep(2.0)
        
        if 2 in tests_to_run:
            tester.test_obstacle_memory_jitter()
            rospy.sleep(2.0)
        
        if 3 in tests_to_run:
            tester.test_obstacle_memory_persistence()
            rospy.sleep(2.0)
        
        if 4 in tests_to_run:
            tester.test_dwa_planner_basic()
            rospy.sleep(2.0)
        
        if 5 in tests_to_run:
            tester.test_dwa_planner_obstacle_avoidance()
            rospy.sleep(2.0)
        
        if 6 in tests_to_run:
            tester.test_obstacle_memory_decay()
            rospy.sleep(2.0)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("ALL TESTS COMPLETE")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Check RViz for visualization of:")
        rospy.loginfo("  - Obstacles (red/yellow spheres)")
        rospy.loginfo("  - Planned trajectory (green line)")
        rospy.loginfo("  - Global path (blue line)")
        
    except KeyboardInterrupt:
        rospy.loginfo("Tests interrupted by user")


if __name__ == '__main__':
    main()
