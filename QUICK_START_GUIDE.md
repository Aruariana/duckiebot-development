# Quick Start Guide - Custom Navigation Stack

## 5-Minute Setup

### 1. Verify Prerequisites ✓
```bash
# Check ROS is running
roscore  # Should be running on your system

# Verify files exist
ls -la ~/duckiebot-development/packages/navigation/src/obstacle_memory_node.py
ls -la ~/duckiebot-development/packages/navigation/src/custom_dwa_planner_node.py

# Verify TF tree
rosrun tf2_tools view_frames.py
# Should show: map → odom → base_link
```

### 2. Launch System
```bash
# Terminal 1: ROS Master
roscore

# Terminal 2: Start your Duckiebot drivers
# (camera, motor_controller, odometry - as per your setup)
roslaunch duckiebot_driver.launch veh:=duckiebot_name

# Terminal 3: Object detection
roslaunch object_detection object_detection_node.launch veh:=duckiebot_name

# Terminal 4: Navigation stack
roslaunch navigation custom_navigation_stack.launch veh:=duckiebot_name
```

### 3. Verify It's Running
```bash
# Terminal 5: Check nodes
rosnode list
# Should show:
# /duckiebot_name/obstacle_memory_node
# /duckiebot_name/custom_dwa_planner_node

# Check topics
rostopic list | grep -E "(obstacle|cmd_vel|path)"
```

### 4. Visualize in RViz
```bash
# Terminal 6: RViz
rosrun rviz rviz
# Or: roslaunch navigation view_navigation.launch
```

**In RViz, add displays**:
1. **Global Path** - Topic: `/duckiebot_name/graph_search_server_node/path`
2. **Obstacles** - Topic: `/duckiebot_name/obstacle_memory_node/obstacles_markers`
3. **Trajectory** - Topic: `/duckiebot_name/custom_dwa_planner_node/debug/best_trajectory`
4. **Robot** - TF frame: `base_link`

### 5. Send Navigation Goal
```bash
# Terminal 7: Request path to target
rostopic pub -1 /duckiebot_name/graph_search_server_node/plan \
  navigation/GraphSearch "{source_node: 'A', target_node: 'B', target_x: 1.0, target_y: 2.0}"

# Or use RViz: 2D Nav Goal button
```

---

## Troubleshooting Quick Reference

### Problem 1: "No module named obstacle_memory_node"
```bash
# Solution: Make files executable
chmod +x ~/duckiebot-development/packages/navigation/src/*.py

# Or: Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/duckiebot-development/packages/navigation/src
```

### Problem 2: "TF transform timeout"
```bash
# Check TF tree
rosrun tf2_tools view_frames.py

# Ensure map → odom → base_link chain exists
# If missing, your odometry node may not be publishing

# Temporary fix (for testing):
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 map odom 50
```

### Problem 3: "Plugin not loaded"
```bash
# Check YOLO model is present
ls -la packages/object_detection/config/model_weights/

# If missing, check object_detection_node logs
rostopic echo /rosout | grep object_detection
```

### Problem 4: Robot doesn't move
```bash
# Check cmd_vel is being published
rostopic echo /duckiebot_name/cmd_vel | head

# Check planner is receiving path
rostopic echo -n 1 /duckiebot_name/graph_search_server_node/path

# Check planner is receiving odometry
rostopic echo -n 1 /duckiebot_name/odometry/filtered

# If any are empty, fix that input before proceeding
```

### Problem 5: Robot crashes into obstacles
**See**: [DWA_TUNING_GUIDE.md](DWA_TUNING_GUIDE.md) - Scenario 2

---

## Terminal Command Template

```bash
#!/bin/bash
# save as run_navigation.sh, then: chmod +x run_navigation.sh && ./run_navigation.sh

DUCKIEBOT_NAME="duckiebot_name"  # Change this!

# Terminal 1
xterm -e "roscore" &

# Wait for roscore
sleep 2

# Terminal 2: Duckiebot drivers
xterm -e "roslaunch duckiebot_driver.launch veh:=$DUCKIEBOT_NAME" &

# Terminal 3: Object detection  
xterm -e "roslaunch object_detection object_detection_node.launch veh:=$DUCKIEBOT_NAME" &

# Terminal 4: Navigation (custom)
xterm -e "roslaunch navigation custom_navigation_stack.launch veh:=$DUCKIEBOT_NAME" &

# Terminal 5: RViz
xterm -e "rosrun rviz rviz" &

echo "Started all nodes. Ready for navigation!"
```

---

## Video/Monitoring Commands

```bash
# Monitor in real-time (similar to top)
# Install: pip install ros-monitor
ros-monitor

# Log all traffic for debugging
rosbag record -a -o navigation_test.bag

# Playback recording
rosbag play navigation_test.bag

# Analyze what was recorded
rosbag info navigation_test.bag
```

---

## Quick Parameter Changes

```bash
# Increase obstacle avoidance (prevent crashes)
rosparam set /duckiebot_name/custom_dwa_planner_node/weight_obstacle 15.0

# Increase speed
rosparam set /duckiebot_name/custom_dwa_planner_node/max_velocity 0.5

# Reduce memory jitter (faster response)
rosparam set /duckiebot_name/obstacle_memory_node/max_obstacle_age 3.0

# Verify changes
rosparam get /duckiebot_name/custom_dwa_planner_node/weight_obstacle
```

---

## Performance Monitoring

```bash
# CPU usage
watch -n 1 'ps aux | grep -E "(obstacle|dwa)"'

# Memory usage
ps aux | grep -E "(obstacle|dwa)" | awk '{print $2, $3"% CPU", $4"% MEM"}'

# Planning latency (add to code to see in logs)
rostopic echo /rosout | grep "Planning took"

# Check package stats
rostopic bw /duckiebot_name/cmd_vel
rostopic hz /duckiebot_name/cmd_vel
```

---

## File Locations (for Reference)

```
packages/navigation/
├── src/
│   ├── obstacle_memory_node.py              ← Main obstacle tracker
│   ├── custom_dwa_planner_node.py           ← Main planner
│   ├── navigation_utils.py                  ← Helper functions
│   └── test_custom_navigation.py            ← Test suite
├── launch/
│   ├── obstacle_memory_node.launch          ← Obstacle node launcher
│   ├── custom_dwa_planner_node.launch       ← Planner node launcher
│   └── custom_navigation_stack.launch       ← Complete system launcher
├── config/
│   ├── obstacle_memory_node/default.yaml    ← Obstacle params
│   └── custom_dwa_planner_node/default.yaml← Planner params
├── CUSTOM_NAVIGATION_README.md              ← Full documentation
├── DWA_TUNING_GUIDE.md                      ← Detailed tuning guide
└── CUSTOM_NAVIGATION_IMPLEMENTATION_SUMMARY.md

CUSTOM_NAVIGATION_IMPLEMENTATION_SUMMARY.md  ← This summary file
```

---

## Next Steps

1. ✅ Get it running with defaults
2. 📖 Read [DWA_TUNING_GUIDE.md](packages/navigation/DWA_TUNING_GUIDE.md)
3. 🎯 Tune for your specific use case
4. 🧪 Run [test suite](packages/navigation/src/test_custom_navigation.py)
5. 🚀 Deploy on physical Duckiebot

---

## Support & Debugging

**Check logs**:
```bash
rostopic echo /rosout | grep -E "(WARN|ERROR)" | head -20
```

**Enable verbose logging**:
```bash
# In launch file, add: output="screen"
# Or monitor nodes directly:
rosnode info /duckiebot_name/obstacle_memory_node
```

**Get help**:
- See [CUSTOM_NAVIGATION_README.md](packages/navigation/CUSTOM_NAVIGATION_README.md) - Common Issues section
- See [DWA_TUNING_GUIDE.md](packages/navigation/DWA_TUNING_GUIDE.md) - Scenario-specific help

---

**You're ready! Good luck!** 🚀🦆
