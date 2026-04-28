# Graph Search + Move Base Integration Guide

## Overview
This setup integrates your custom graph search with ROS's `move_base` stack for:
- **Global planning**: A* search on your Duckietown graph
- **Local planning**: DWA (Dynamic Window Approach) for obstacle avoidance
- **Smooth trajectories**: Continuous motion between graph waypoints

## Installation

### 1. Install Required ROS Packages
```bash
sudo apt-get install ros-<distro>-move-base
sudo apt-get install ros-<distro>-dwa-local-planner
sudo apt-get install ros-<distro>-navfn
sudo apt-get install ros-<distro>-map-server
sudo apt-get install ros-<distro>-amcl  # Optional: for localization
```

### 2. Make Python Script Executable
```bash
chmod +x packages/navigation/src/graph_search_move_base_node.py
```

### 3. Catkin Build
```bash
cd ~/DTCORE/dt-core
catkin_make
source devel/setup.bash
```

## Configuration Files

The setup includes several YAML config files in `packages/navigation/config/`:

- **move_base_params.yaml**: Main move_base parameters
- **costmap_common_params.yaml**: Shared costmap settings
- **local_costmap_params.yaml**: Local costmap (rolling window)
- **global_costmap_params.yaml**: Global costmap (static map)
- **dwa_local_planner_params.yaml**: DWA planner tuning

## Usage

### Launch the System
```bash
roslaunch navigation graph_search_move_base.launch map_name:=<your_map>
```

This will start:
1. `map_server` - Loads your map
2. `move_base` - Navigation stack with global + local planners
3. `graph_search_move_base_node` - Custom graph search service
4. `rviz` - Visualization

### Call the Service
```bash
rosservice call /graph_search "source_node: 'node_1' target_node: 'node_5'"
```

## Important Notes

### 1. **Coordinate Transformation**
You **MUST** implement `_get_waypoints_from_path()` to extract actual coordinates from your graph:
```python
def _get_waypoints_from_path(self, path):
    # Convert graph actions to (x, y, theta) coordinates
    # using your graph's position data
```

### 2. **TF Tree**
Ensure these transforms are published:
- `map` → `odom` (from AMCL or odometry)
- `odom` → `base_link` (from your odometry node)
- `base_link` → `laser` (from your sensor setup)

### 3. **Sensor Topics**
Remap these to match your robot:
```xml
<remap from="cmd_vel" to="your_cmd_topic" />
<remap from="odom" to="your_odom_topic" />
<remap from="scan" to="your_laser_topic" />
```

### 4. **Parameters Tuning**
Adjust in `dwa_local_planner_params.yaml`:
- `max_vel_x`: Robot's maximum speed
- `acc_lim_x`: Acceleration limits
- `inflation_radius`: Buffer around obstacles

## Architecture Comparison

| Aspect | Original | New with Move Base |
|--------|----------|-------------------|
| Global Planning | A* on graph | A* on graph (same) |
| Local Planning | None | DWA local planner |
| Obstacle Avoidance | No | Yes |
| Trajectory Type | Discrete actions | Continuous path |
| Complexity | Simple | Moderate |
| Requires TF | No | Yes |
| Requires Odometry | No | Yes |

## Troubleshooting

### Robot doesn't move
- Check TF tree: `rosrun tf tf_monitor`
- Verify cmd_vel remap is correct
- Check move_base is publishing `/move_base/status`

### Path planning fails
- Verify `/map` topic is being published
- Check that source/target nodes exist in graph
- Ensure waypoint extraction returns valid coordinates

### Local planner fails
- Check costmaps are being updated: `rostopic list | grep costmap`
- Verify sensor data (laser scans) are being received
- Tune DWA parameters for your robot's speed/size

### High CPU usage
- Reduce `update_frequency` in costmap config
- Decrease costmap resolution
- Lower `sim_periods` in DWA params

## Hybrid Approach
You can also keep the original node and run both:
```bash
# Terminal 1: Original graph search only
rosrun navigation graph_search_server_node.py

# Terminal 2: Graph search + Move Base
roslaunch navigation graph_search_move_base.launch
```

Then choose which service to use based on your needs.
