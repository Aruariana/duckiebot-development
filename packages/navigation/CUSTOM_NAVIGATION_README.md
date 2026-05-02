#!/usr/bin/env python3
"""
Custom Navigation Stack for Duckiebot DB21M
=============================================

This module provides a complete lightweight navigation system with obstacle avoidance
designed specifically for Duckiebot constraints:
- Limited onboard compute (no complex 3D costmap)
- Mono camera (depth estimation from object class and bounding box size)
- Small footprint (can use discrete trajectory sampling)

Architecture Overview
====================

1. OBSTACLE MEMORY NODE (obstacle_memory_node.py)
   Purpose: Maintains persistent memory of detected obstacles in global coordinates
   
   Input:
   - DuckieObstacle messages from object_detection_node
   - TF tree (map -> odom -> base_link)
   
   Processing:
   - Converts bounding box bottom point to local ground coordinates using homography
   - Transforms from base_link frame to map frame using TF
   - Applies moving average filtering to smooth YOLO jitter
   - Handles obstacle lifecycle (creation, update, timeout)
   - Matches new detections to existing obstacles if spatially close
   
   Output:
   - MarkerArray for RViz visualization
   - MarkerArray for planner consumption
   
   Key Concepts:
   - Semantic Priors: Dictionary mapping class names to (radius, safety_margin)
   - Global Persistence: Once detected, obstacles stay in memory until timeout
   - Temporal Filtering: Moving average window reduces YOLO detection noise

2. CUSTOM DWA PLANNER NODE (custom_dwa_planner_node.py)
   Purpose: Lightweight local path planning without costmap_2d
   
   Input:
   - Global path from navigation package
   - Current odometry (pose + velocity)
   - Obstacle list from obstacle memory node
   
   Processing:
   - Samples (v, ω) velocity pairs from kinematic limits
   - Simulates trajectory for each velocity pair (1 second horizon)
   - Evaluates 4-component cost function:
     1. Path Cost: Deviation from global path
     2. Obstacle Cost: Proximity to obstacles (∞ if collision)
     3. Velocity Reward: Encourages forward motion
     4. Heading Cost: Smooth heading changes
   - Selects velocity pair with minimum total cost
   
   Output:
   - Twist2DStamped velocity command at 10 Hz
   - MarkerArray showing best trajectory (debug)
   
   Key Concepts:
   - No Costmap: All costs computed on-the-fly from raw obstacle positions
   - Trajectory Rollout: Simulates actual robot motion model
   - Repulsive Potential: Inverse distance to obstacle drives avoidance

Integration with Existing System
================================

1. ROS Topic Connections:

   object_detection_node
   └─> DuckieObstacle (duckie_obstacle)
       └─> obstacle_memory_node
           └─> MarkerArray (obstacles)
               └─> custom_dwa_planner_node
                   ├─> Subscribes: /odom
                   ├─> Subscribes: /graph_search_server_node/path
                   └─> Publishes: car_cmd -> motor_controller

2. TF Tree Requirements:

   Expected TF chain (should already exist on your Duckiebot):
   map
   └─> odom (from odometry node)
       └─> base_link
           └─> camera_optical_frame (for image projection)
   
   Our nodes use:
   - map ← base_link (for obstacle transformation)

3. Launching the System:

   # Option 1: Launch all components together
   roslaunch navigation custom_navigation_stack.launch veh:=duckiebot_name
   
   # Option 2: Launch nodes separately
   roslaunch navigation obstacle_memory_node.launch veh:=duckiebot_name
   roslaunch navigation custom_dwa_planner_node.launch veh:=duckiebot_name

Configuration and Tuning
========================

1. Obstacle Memory Parameters (config/obstacle_memory_node/default.yaml):

   - max_obstacle_age: How long to remember detected obstacles (5s default)
   - moving_avg_window: Size of smoothing filter (5 samples default)
   - semantic_priors: Class-specific properties

   Tuning Tips:
   - If obstacles disappear too quickly: Increase max_obstacle_age
   - If robot crashes into obstacles: Increase safety_margin in semantic_priors
   - If obstacles appear jittery: Increase moving_avg_window

2. DWA Planner Parameters (config/custom_dwa_planner_node/default.yaml):

   - velocity_resolution / omega_resolution: Trajectory quality vs speed
   - prediction_horizon: How far ahead to look (1s good for Duckiebot)
   - weight_*: Relative importance of each cost component

   Tuning Tips:
   - If robot stalls: Reduce weight_obstacle or increase weight_velocity
   - If robot crashes: Increase weight_obstacle or weight_path
   - If oscillations: Increase prediction_horizon
   - If jerky turns: Increase weight_heading

3. Semantic Priors for New Object Classes:

   To add a new detected class (e.g., cones):
   
   a) In config/obstacle_memory_node/default.yaml:
      semantic_priors:
        cone:
          radius: 0.08
          safety_margin: 0.05
          confidence_threshold: 0.5
   
   b) In object_detection_node.py, expand class detection:
      if det['name'] == 'cone':
          class_name = 'cone'

Performance Considerations
==========================

1. Computational Efficiency:

   - No GPU-accelerated costmap needed
   - DWA runs at 10 Hz (not 30 Hz like some planners)
   - Typical computation: <100ms per planning step on Jetson Nano
   - Memory footprint: ~50-100 obstacles can be tracked comfortably

2. Safety Margins:

   - Mono camera depth is unreliable at distance
   - Conservative safety_margin (0.05-0.10m) recommended
   - Cost function has hard collision check (not soft)
   - No trajectory is selected if ANY point is in collision

3. Latency and Responsiveness:

   - 10 Hz planning: 100ms latency to new obstacles
   - YOLO detection: ~50-100ms per frame
   - Total latency: ~200-300ms to react
   - At 0.3 m/s forward speed, this is ~0.06-0.09m of travel before reaction

Debug and Visualization
=======================

1. RViz Visualization:

   Add these displays to RViz:
   - MarkerArray from duckiebot_name/obstacle_memory_node/obstacles_markers
     → Shows detected obstacles as red/yellow spheres
   - MarkerArray from duckiebot_name/custom_dwa_planner_node/debug/best_trajectory
     → Shows planned trajectory (green line)
   - Path from duckiebot_name/graph_search_server_node/path
     → Shows global plan to follow

2. ROS Parameters for Debugging:

   rostopic echo /duckiebot_name/custom_dwa_planner_node/car_cmd
   rostopic echo /duckiebot_name/custom_dwa_planner_node/debug/best_trajectory

3. Logging:

   Both nodes publish debug logs:
   rostopic echo /rosout | grep MemoryNode
   rostopic echo /rosout | grep DWAPlanner

Common Issues and Solutions
===========================

1. Robot doesn't avoid obstacles:
   → Check if obstacle_memory_node is running
   → Verify TF tree: rosrun tf2_tools view_frames.py
   → Increase weight_obstacle in DWA config

2. Robot oscillates or doesn't move:
   → Reduce weight_heading
   → Increase velocity_resolution for smoother trajectories
   → Increase max_velocity to ensure forward motion is always possible

3. Obstacles appear at wrong positions:
   → Verify camera calibration (homography matrix)
   → Check ground_projection_node is working
   → Verify TF transforms with: rosrun tf2_tools view_frames.py

4. Memory usage grows unbounded:
   → Reduce max_obstacle_age
   → Check if obstacle_memory_node is crashing/restarting

Extending the System
====================

1. Add More Object Types:

   a) Train YOLOv5 model with additional classes
   b) Update semantic_priors in config
   c) Modify object_detection_node.py to publish class name in DuckieObstacle
   (currently hardcoded to 'duckie')

2. Implement Global Path Optimization:

   Currently uses simple graph search output. Could add:
   - Hybrid A* for smoother paths
   - Dubins path smoothing for curvature constraints

3. Advanced Obstacle Dynamics:

   Current system assumes static obstacles. For moving obstacles:
   - Track velocity of obstacles in ObstacleMemory
   - Extrapolate future positions in DWA cost function
   - Use velocity-dependent safety margins

4. Multi-Robot Coordination:

   For swarms of Duckiebots:
   - Broadcast obstacle positions to other robots
   - Include other robots as obstacles in planning
   - Add communication latency compensation

References and Papers
====================

1. Dynamic Window Approach (DWA):
   Original: Fox et al., "The Dynamic Window Approach to Collision Avoidance"
   
2. Ground Projection and Camera Calibration:
   Based on Duckietown's complete_image_pipeline package
   
3. Mono Depth Estimation:
   Conservative approach: assume worst-case (farthest) depth consistent with image region

Troubleshooting Script
======================

Save as check_system.sh:

#!/bin/bash
echo "=== Checking Custom Navigation Stack ==="
echo ""
echo "1. ROS Master:"
rosnode list
echo ""
echo "2. Required Topics:"
rostopic list | grep -E "(obstacle|path|car_cmd|odom)"
echo ""
echo "3. TF Tree:"
rosrun tf2_tools view_frames.py 2>/dev/null | head -20
echo ""
echo "4. Node Status:"
rosnode info obstacle_memory_node 2>/dev/null | head -10
rosnode info custom_dwa_planner_node 2>/dev/null | head -10
