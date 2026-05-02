# Integration Architecture - Custom Navigation Stack with Existing Duckiebot Systems

## System Overview

This document describes how the new custom navigation stack integrates with your existing Duckiebot system.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUCKIEBOT NAVIGATION SYSTEM                   │
└─────────────────────────────────────────────────────────────────┘

PERCEPTION LAYER:
├─ 📷 Camera Driver
├─ 🧠 YOLOv5 Object Detector
├─ 🔍 Ground Projection (Image → Ground Plane)
└─ 📍 Odometry Node

MEMORY & PLANNING LAYER (NEW):
├─ **Obstacle Memory Node** (CUSTOM)
│  └─ Input: Object detections
│  └─ Output: Global obstacle map
├─ **Custom DWA Planner** (CUSTOM)
│  ├─ Input: Path, Odometry, Obstacles
│  └─ Output: Velocity commands
└─ Graph Search (Existing global planner)

CONTROL LAYER:
├─ Motor Controller
├─ LED Controller
└─ FSM (Finite State Machine)

MIDDLEWARE:
├─ TF Transforms (map → odom → base_link)
├─ ROS Parameter Server
└─ ROS Services/Actions
```

---

## Data Flow

### Current System (Without Custom Navigation)

```
Camera Frame
    ↓
[Line Detector] → Segments
    ↓
[Ground Projection] → Ground segments
    ↓
[Lane Filter] → Lane pose estimate
    ↓
[Lane Controller] → Motor commands
```

### With Custom Navigation Stack

```
Camera Frame
    ├─→ [Line Detector] → Segments → [Ground Projection] → [Lane Filter] → Lane pose
    │
    └─→ [YOLOv5 Detector] → Detections
                  ↓
           [Ground Projection] 
                  ↓
           [Obstacle Memory] → Global obstacle map
                  ↓
           [Custom DWA Planner] ←─────────────┐
                  ↓                           │
           Motor commands                Global path
                  ↓
           [Motor Controller]

Note: Global path comes from nav stack (graph_search) or manually specified
```

---

## How Nodes Communicate

### 1. Object Detection → Obstacle Memory

**What object_detection_node publishes**:
```
Topic: /duckiebot_name/object_detection_node/duckie_obstacle
Type: duckiebot_msgs/DuckieObstacle
Message:
  bool detected
  float32 distance
  geometry_msgs/Point position  (normalized: [0-1] in camera frame)
```

**What obstacle_memory_node does**:
1. Subscribes to DuckieObstacle
2. Receives bounding box bottom point (in normalized image coordinates)
3. Converts to local ground coords using homography (pixel → ground plane)
4. Transforms to global coordinates using TF (base_link → map)
5. Publishes on:
   - `~obstacles_markers`: MarkerArray (for RViz)
   - `~obstacles`: MarkerArray (for planner)

### 2. Path + Odometry + Obstacles → DWA Planner

**Custom DWA Planner subscribes to**:
- `/duckiebot_name/graph_search_server_node/path` (nav_msgs/Path)
  - Global path from navigation graph search
- `/duckiebot_name/odometry/filtered` (nav_msgs/Odometry)
  - Current robot pose and velocity
- `/duckiebot_name/obstacle_memory_node/obstacles` (visualization_msgs/MarkerArray)
  - Global obstacle list from obstacle memory

**Custom DWA Planner publishes**:
- `~cmd_vel` (duckietown_msgs/Twist2DStamped)
  - Velocity command for motor controller

### 3. Motor Controller Receives cmd_vel

Your motor controller should be subscribed to:
```
Topic: /duckiebot_name/custom_dwa_planner_node/cmd_vel
Type: duckietown_msgs/Twist2DStamped
Message:
  v (linear velocity in m/s)
  omega (angular velocity in rad/s)
```

---

## Integration Points with Existing Nodes

### Lane Following (Optional - Can coexist)

**Scenario A: Use ONLY Custom Navigation**
- Disable lane controller
- Let custom DWA follow the global path
- Pro: Unified system, easier debugging
- Con: No lane-level control precision

**Scenario B: Hybrid - Lane + Obstacle Avoidance**
- Keep lane controller active
- Add light obstacle avoidance in lane controller
- Use custom DWA when global path is available
- Pro: Best of both worlds
- Con: More complex, more configuration

**Recommended**: Start with Scenario A (Scenario B for advanced users)

### Graph Search Integration

Your graph_search_server publishes paths in map frame:
- Obstacle Memory → Processes detections in this same map frame
- Custom DWA → Plans trajectories relative to same map frame
- **Perfect integration** ✓

### Odometry Integration

Must provide odometry in correct format:

**Expected**:
```
Topic: /duckiebot_name/odometry/filtered
Type: nav_msgs/Odometry
Contains:
  pose.pose.position (x, y, z)
  pose.pose.orientation (quaternion)
  twist.twist.linear.x (linear velocity)
  twist.twist.angular.z (angular velocity)
```

**Verify**:
```bash
rostopic hz /duckiebot_name/odometry/filtered  # Should be 20-50 Hz
rostopic echo -n 1 /duckiebot_name/odometry/filtered
```

If your odometry topic has different name, remap in launch file:
```xml
<remap from="~odom" to="/duckiebot_name/odometry/your_name"/>
```

---

## TF Tree Integration

### Expected TF Tree

```
map (global frame)
  ├─ frame_id: "map"
  └─ child_frame_id: "odom"
     ├─ broadcaster: odometry_node
     └─ child_frame_id: "base_link"
        ├─ broadcaster: odometry node or tf_static
        └─ child_frame_id: "camera_optical_frame"
           └─ broadcaster: camera info or calibration
```

### Verification

```bash
# View TF tree
rosrun tf2_tools view_frames.py
# Or: rostopic echo /tf

# Test transform lookup
rosrun tf2_tools echo.py map base_link
# Should show: [x, y, z] and quaternion [qx, qy, qz, qw]
```

### Issues and Fixes

**Problem**: "Lookup would require extrapolation into the future"
```bash
# Cause: Clock skew between systems
# Fix: Ensure all systems use same ROS time source
rosparam get /use_sim_time  # Should be false unless using bag files
```

**Problem**: "No transform from base_link to map"
```bash
# Cause: Odometry node not running or TF publisher missing
# Fix: 
rosnode list | grep odometry
# If missing, launch odometry node
```

---

## Configuration Changes Needed

### 1. Object Detection Configuration

**Current object_detection_node.py** already:
- ✅ Uses ground projection ✓
- ✅ Publishes DuckieObstacle ✓
- ✅ Computes ground plane coordinates ✓

**No changes needed** - it already outputs in correct format

### 2. Graph Search Configuration

**Current graph_search_server** already:
- ✅ Publishes path in map frame ✓
- ✅ Available as service ✓
- ✅ Exports path as nav_msgs/Path ✓

**No changes needed** - compatible as-is

### 3. Motor Controller Integration

**Ensure your motor controller subscribes to**:
```
/duckiebot_name/custom_dwa_planner_node/cmd_vel
(or remap this topic in launch file)
```

**Old method** (lane controller):
```
/duckiebot_name/lane_controller_node/car_cmd  ← DISABLE if using custom DWA
```

### 4. Launch File Changes

**Old system**:
```xml
<node pkg="lane_control" type="lane_controller_node.py" name="lane_controller_node"/>
```

**New system** - keep object detection but add:
```xml
<!-- Custom Navigation -->
<include file="$(find navigation)/launch/custom_navigation_stack.launch">
    <arg name="veh" value="$(arg veh)"/>
</include>
```

---

## Mode Switching Strategy

### Three Operating Modes

#### Mode 1: Pure Lane Following (Existing)
- Lane controller active
- No global path
- No obstacle avoidance
- **Use when**: Simple corridor following

```bash
roslaunch lane_control lane_controller_node.launch veh:=duckiebot_name
```

#### Mode 2: Custom Navigation (New)
- Custom DWA active
- Global path provided
- Full obstacle avoidance
- **Use when**: Mission with waypoints, dynamic obstacles

```bash
roslaunch navigation custom_navigation_stack.launch veh:=duckiebot_name
```

#### Mode 3: Hybrid (Advanced)
- Both systems active
- Lane controller provides baseline
- DWA layer adds obstacle avoidance
- **Use when**: Complex scenarios, need robustness

Requires:
- Lane controller outputting lower priority command
- DWA overlaying when obstacles detected
- Arbitration logic (e.g., FSM)

---

## Testing Integration

### Test 1: Can nodes communicate?

```bash
# Terminal 1: Launch navigation
roslaunch navigation custom_navigation_stack.launch veh:=duckiebot_name

# Terminal 2: Check topics exist
rostopic list
# Should include:
# /duckiebot_name/obstacle_memory_node/obstacles
# /duckiebot_name/custom_dwa_planner_node/cmd_vel

# Terminal 3: Send test path
rostopic pub -1 /duckiebot_name/graph_search_server_node/path nav_msgs/Path "
header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'map'
poses:
- pose:
    position:
      x: 0.0
      y: 0.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0"

# Terminal 4: Check cmd_vel is published
rostopic echo /duckiebot_name/custom_dwa_planner_node/cmd_vel
# Should show velocity updates
```

### Test 2: Obstacle detection flow

```bash
# Terminal 1: Launch system
roslaunch navigation custom_navigation_stack.launch veh:=duckiebot_name

# Terminal 2: Check object detection
rostopic echo -n 1 object_detection_node/duckie_obstacle

# Terminal 3: Obstacle memory processing
rostopic echo /duckiebot_name/obstacle_memory_node/obstacles

# Should see obstacles appear in global coordinates
```

### Test 3: Full integration test

```bash
# Use provided test script
python3 packages/navigation/src/test_custom_navigation.py
```

---

## Performance Expectations

### CPU/Memory After Integration

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| Object Detection | 20-30% | 300-400 MB | Already running |
| Obstacle Memory | 5-10% | 100-150 MB | LOW COST |
| DWA Planner | 15-25% | 50-100 MB | LOW COST |
| Total Added | 20-35% | 150-250 MB | Still room for other systems |

### Latency Contribution

| Stage | Latency | Notes |
|-------|---------|-------|
| YOLO Detection | 50-100 ms | Fixed |
| Ground Projection | 2-5 ms | Fast |
| TF Transform | 1-3 ms | Fast |
| DWA Planning | 30-50 ms | Fixed 10 Hz cycle |
| Total | 80-160 ms | Depends on YOLOv5 model |

---

## Rollback / Disable Custom Navigation

If you need to disable custom navigation and use old system:

```bash
# Option 1: Don't launch custom nodes
# Just launch lane controller as before
roslaunch lane_control lane_controller_node.launch veh:=duckiebot_name

# Option 2: Keep nodes running but disable inputs
# Don't publish path to DWA planner
# Sets cmd_vel to zero, route to lane controller instead

# Option 3: Kill nodes
rosnode kill /duckiebot_name/obstacle_memory_node
rosnode kill /duckiebot_name/custom_dwa_planner_node
```

---

## Troubleshooting Integration

### Problem: Multiple cmd_vel publishers (conflict)

**Symptom**: Motor controller receives contradictory commands

**Solution**:
1. Ensure only ONE cmd_vel publisher is active
2. Use topic remapping to route commands through single controller
3. Implement arbitration logic if multiple planners needed

```xml
<!-- Make sure only DWA publishes to motor -->
<remap from="/duckiebot_name/lane_controller_node/car_cmd" 
       to="/duckiebot_name/lane_cmd_vel"/>  <!-- Different topic -->
<remap from="/duckiebot_name/custom_dwa_planner_node/cmd_vel" 
       to="/duckiebot_name/cmd_vel"/>       <!-- Motor controller subscribes here -->
```

### Problem: Object detections not reaching planner

**Debug flow**:
```bash
# Check 1: Object detection running
rostopic list | grep object_detection

# Check 2: Obstacle memory receiving detections
rostopic hz object_detection_node/duckie_obstacle

# Check 3: Obstacle memory publishing obstacles
rostopic echo /duckiebot_name/obstacle_memory_node/obstacles

# Check 4: DWA receiving obstacles
# (Would be visible in DWA debug output)

# If any step fails, check node logs
rosnode info obstacle_memory_node
```

### Problem: DWA planning but motor not responding

**Debug**:
```bash
# Check cmd_vel is published
rostopic echo /duckiebot_name/custom_dwa_planner_node/cmd_vel

# Check motor controller is subscribed
# (Check your motor controller code or logs)

# Verify topic name matches what motor controller expects
# May need remap in launch file
```

---

## Next Steps

1. ✅ **Start**: Launch custom navigation stack with defaults
2. 📊 **Monitor**: Check all nodes are running without errors
3. 🎯 **Test**: Run test suite to verify integration
4. ⚙️ **Configure**: Tune parameters for your environment (see DWA_TUNING_GUIDE.md)
5. 🚀 **Deploy**: Use on physical Duckiebot for real missions

---

For more details, see:
- [CUSTOM_NAVIGATION_README.md](packages/navigation/CUSTOM_NAVIGATION_README.md)
- [DWA_TUNING_GUIDE.md](packages/navigation/DWA_TUNING_GUIDE.md)
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
