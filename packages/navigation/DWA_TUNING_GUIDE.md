# DWA Planner Tuning Guide for Duckiebot DB21M

## Overview

This guide explains how to tune the custom DWA planner for your specific Duckiebot, environment, and use case. The planner has many parameters that affect behavior, and finding the right balance requires experimentation.

## Key Concepts

### Cost Function

The planner selects velocities (v, ω) to minimize:

```
total_cost = w_path × path_cost + w_obstacle × obstacle_cost + w_velocity × velocity_cost + w_heading × heading_cost
```

Where:
- **path_cost**: How far trajectory deviates from planned path (lane-keeping)
- **obstacle_cost**: Distance to obstacles (∞ if collision) (safety)
- **velocity_cost**: Discourages stalling (forward progress)
- **heading_cost**: Smooth turns (ride comfort)

### Velocity Space & Dynamic Window

The planner samples discrete velocities within the **dynamic window**:

**Key Insight**: Unlike static sampling (0 → max_velocity), DWA constrains velocities to what's **physically reachable** from the current velocity given acceleration limits.

```
Reachable linear velocities:
  [max(0, v_current - max_accel_v × dt), min(max_velocity, v_current + max_accel_v × dt)]

Reachable angular velocities:
  [max(-max_omega, ω_current - max_accel_ω × dt), min(max_omega, ω_current + max_accel_ω × dt)]
```

**Example**: If current v=0.2 m/s, max_accel_v=0.5 m/s², dt=0.1s:
- Without dynamic window: sample [0, 0.1, 0.2, ..., 0.4] (unrealistic)
- With dynamic window: sample [0.15, 0.175, 0.2, 0.225, 0.25] (reachable!)

**Benefit**: Ensures velocity commands match robot's actual acceleration capabilities, enabling smoother motion and better emergency braking response.

More samples = better quality but slower computation.

## Common Tuning Scenarios

### Scenario 1: Robot Stalls or Doesn't Move

**Symptom**: Robot stands still even with clear path ahead

**Root Cause**: Obstacle cost is too high or path cost prevents forward motion

**Solutions**:
```yaml
# Decrease obstacle weight or increase velocity preference
weight_obstacle: 5.0          # Reduce from 10.0
weight_velocity: 0.5          # Increase from 0.1

# Or: Ensure no phantom obstacles
max_obstacle_age: 3.0          # Forget obstacles faster
moving_avg_window: 3           # Less smoothing = faster detection removal

# Or: Increase velocity sampling to find forward option
velocity_resolution: 7         # Increase from 5
```

**Debug**:
```bash
# Monitor if any velocity is selected as "best"
rostopic echo /duckiebot_name/custom_dwa_planner_node/car_cmd | grep "v:"

# Check if obstacles are blocking (view_frames should show correct TF)
rostopic echo /duckiebot_name/obstacle_memory_node/obstacles
```

---

### Scenario 2: Robot Crashes Into Obstacles

**Symptom**: Robot doesn't avoid detected obstacles

**Root Cause**: Obstacle weight too low or safety margins too small

**Solutions**:
```yaml
# Increase obstacle avoidance priority
weight_obstacle: 20.0          # Increase from 10.0

# Increase safety margins around obstacles
semantic_priors:
  duckie:
    safety_margin: 0.10        # Increase from 0.05

# Shorter prediction horizon might miss distant obstacles
prediction_horizon: 1.5        # Increase from 1.0

# More angular samples for evasive maneuvers
omega_resolution: 15           # Increase from 9
```

**Debug**:
```bash
# Verify obstacle detection and transformation
rosrun tf2_tools view_frames.py
# Check if obstacles appear at expected global positions

# Monitor cost breakdown
rostopic echo /duckiebot_name/custom_dwa_planner_node/debug/best_trajectory
```

---

### Scenario 3: Robot Oscillates or Jerky Movement

**Symptom**: Robot weaves left-right or makes abrupt turns

**Root Cause**: Heading cost too low or angular sample rate causing instability

**Solutions**:
```yaml
# Smooth heading changes
weight_heading: 0.2            # Increase from 0.05

# Longer prediction horizon reduces jerky decisions
prediction_horizon: 1.5        # Increase from 1.0

# More velocity samples for smoother interpolation
velocity_resolution: 7         # Increase from 5
omega_resolution: 15           # Increase from 9

# Finer trajectory simulation
simulation_dt: 0.05            # Decrease from 0.1
```

---

### Scenario 4: Robot Doesn't Follow Global Path

**Symptom**: Robot moves forward but drifts away from intended route

**Root Cause**: Path cost weight too low

**Solutions**:
```yaml
# Increase lane-keeping priority
weight_path: 2.0               # Increase from 1.0

# Ensure global path is being published
# In custom_dwa_planner_node.py:
# rospy.loginfo(f"Path has {len(self.global_path)} waypoints")
```

---

### Scenario 5: Slow Response to Obstacles (Sluggish Reaction)

**Symptom**: Detects obstacle but takes too long to avoid it

**Root Cause**: Non-real-time planner rate or moving average too large

**Solutions**:
```yaml
# Reduce moving average window for faster position updates
moving_avg_window: 3           # Decrease from 5

# Reduce obstacle memory age
max_obstacle_age: 3.0          # Decrease from 5.0

# In code: Planner runs at 10 Hz (100ms cycle time)
# This is set in custom_dwa_planner_node.py:
# self.plan_timer = rospy.Timer(rospy.Duration(0.1), self.plan_step)

# To increase rate to 20 Hz:
# self.plan_timer = rospy.Timer(rospy.Duration(0.05), self.plan_step)
# But monitor CPU usage on Jetson Nano
```

---

### Scenario 6: Unrealistic Velocity Commands or Jerky Acceleration

**Symptom**: Robot commanded velocities that it can't physically achieve; motion stutters or jerks

**Root Cause**: Acceleration limits (`max_accel_v`, `max_accel_omega`) don't match robot's actual capabilities or are not set

**Solutions**:
```yaml
# Measure actual acceleration capability
# If motor can accelerate 0→0.4 m/s in 1 second: max_accel_v ≈ 0.4 m/s²

max_accel_v: 0.5               # Conservative estimate
max_accel_omega: 2.0           # For steering acceleration

# Too aggressive (commands exceed motor capability):
# max_accel_v: 2.0             # DON'T - unrealistic
# max_accel_omega: 5.0         # DON'T - unrealistic

# Too conservative (overly smooth but sluggish):
# max_accel_v: 0.1             # May cause slow responses
```

**Why this matters**:
- Without proper acceleration limits, simulated trajectories don't match reality
- Robot can't brake as fast as simulation predicts → collision risk
- Creates jerkiness as planner constantly changes target velocities

**Debug**:
```bash
# Check if velocity commands are realistic
rostopic echo /duckiebot_name/car_cmd

# Monitor actual vs. commanded velocity
# rostopic echo /duckiebot_name/odom | grep -A2 "linear"

# If actual velocity lags far behind commanded velocity:
# → acceleration limits are too aggressive
```

---



### Velocity Limits

```yaml
max_velocity: 0.4              # Physical limit of Duckiebot motors
max_omega: 1.5                 # Steering rate limit of Duckiebot

# Tips:
# - Reduce if robot overshoots turns
# - Increase if robot feels sluggish
# - Must match actual motor calibration
```

### Acceleration Limits (Dynamic Window)

**These are critical for realistic DWA behavior!**

```yaml
max_accel_v: 0.5               # Linear acceleration (m/s²)
max_accel_omega: 2.0           # Angular acceleration (rad/s²)
```

**What these do**:
- Constrain velocity sampling to reachable values
- Prevent unrealistic instant velocity jumps
- Enable smoother trajectory changes
- Improve emergency braking response

**How to tune**:

1. **If robot motion feels jerky or makes abrupt velocity changes**:
   ```yaml
   max_accel_v: 0.3            # Conservative - smoother acceleration
   max_accel_omega: 1.0        # Conservative - gentler turns
   ```
   - Smaller values → smoother but less responsive
   - Good for: Smooth lane-following, navigation with delicate cargo

2. **If robot seems unresponsive or slow to avoid obstacles**:
   ```yaml
   max_accel_v: 1.0            # Aggressive - quick direction changes
   max_accel_omega: 3.0        # Aggressive - sharp evasive turns
   ```
   - Larger values → more responsive but rougher motion
   - Good for: Crowded environments, emergency avoidance

3. **To measure actual robot capability**:
   ```bash
   # Command full forward: v=0.4, ω=0
   # Measure how long to reach from v=0 (should be ~0.8s at 0.5 m/s²)
   # actual_accel = Δv / Δt
   ```

**Rule of thumb**:
- `max_accel_v` should be 1-3× slower than peak motor acceleration
- Conservative values (0.3-0.5) work well for small robots
- Aggressive values (1.0+) only if motors tested at high power

### Sampling Resolution

```yaml
velocity_resolution: 5         # Typical: 3-7 (more = better quality, slower)
omega_resolution: 9            # Typical: 5-15

# Trade-off:
# 3×5 = 15 trajectories (very fast, low quality)
# 7×15 = 105 trajectories (slower, high quality)
# 5×9 = 45 trajectories (good balance)
```

### Prediction Horizon

```yaml
prediction_horizon: 1.0        # How far into future to predict (seconds)

# Typical values:
# 0.5 s: Very reactive, may miss distant obstacles
# 1.0 s: Good balance, TRadeooof is typical DWA horizon for Duckiebot-scale
# 2.0 s: Very predictive but slow, might be overkill

# At v=0.3 m/s, 1s horizon = 0.3m of visible future
# This is 3-4 Duckiebot lengths - reasonable for reactive planning
```

---

## Scenario-Specific Configurations

### Configuration A: Lane Following (Loop Closure)

```yaml
# Prioritize staying on road, minimal dynamic obstacles expected
weight_path: 3.0
weight_velocity: 0.05
weight_obstacle: 5.0
weight_heading: 0.1

velocity_resolution: 5
omega_resolution: 9

# Conservative acceleration for smooth lane-following
max_accel_v: 0.3               # Gentle acceleration
max_accel_omega: 1.0           # Smooth steering

# Faster forgetting of transient obstacles
max_obstacle_age: 2.0
moving_avg_window: 3
```

### Configuration B: Crowded Environment (Many Obstacles)

```yaml
# Prioritize not crashing, less emphasis on exact path
weight_path: 0.5
weight_velocity: 0.2
weight_obstacle: 15.0          # Safety first
weight_heading: 0.15

velocity_resolution: 7         # Higher quality to find gaps
omega_resolution: 15

# Aggressive acceleration for evasive maneuvers
max_accel_v: 0.8               # Quick direction changes
max_accel_omega: 2.5           # Responsive steering

# Hold obstacles longer to avoid re-collisions
max_obstacle_age: 8.0
moving_avg_window: 7           # Smooth out noise for stable decisions
```

### Configuration C: Fast Indoor Navigation

```yaml
# Assume relatively clean environment, emphasis on speed
weight_path: 1.5
weight_velocity: 0.5           # Encourage speed
weight_obstacle: 8.0
weight_heading: 0.05

velocity_resolution: 5
omega_resolution: 7

max_velocity: 0.5              # Push speed limit
max_omega: 2.0

# Moderate acceleration for responsive but controlled motion
max_accel_v: 0.6               # Balanced responsiveness
max_accel_omega: 2.0

prediction_horizon: 0.8        # Shorter horizon = faster decisions

max_obstacle_age: 2.0          # Forget quickly
moving_avg_window: 3
```

### Configuration D: Careful Motion (Delivery, Fragile Cargo)

```yaml
# Maximize comfort and safety
weight_path: 1.0
weight_velocity: 0.05          # Gentle acceleration
weight_obstacle: 12.0
weight_heading: 0.3            # Smooth turns

velocity_resolution: 7         # High quality
omega_resolution: 15

simulation_dt: 0.05            # Fine-grained simulation

max_velocity: 0.25             # Conservative speed
max_omega: 0.8                 # Gentle steering

# Very conservative acceleration for ultra-smooth motion
max_accel_v: 0.2               # Gradual acceleration
max_accel_omega: 0.5           # Minimal steering jerk

max_obstacle_age: 6.0
moving_avg_window: 7
```

---

## Systematic Tuning Process

1. **Baseline Test**: Record behavior with default parameters
   ```bash
   rosrun rosbag record /duckiebot_name/odom /duckiebot_name/car_cmd
   ```

2. **Change One Parameter**: Modify single parameter by ±20%

3. **Test in Same Scenario**: Ensure fair comparison

4. **Measure**: Document
   - Success rate (crash-free runs)
   - Path deviation (lateral error from intended route)
   - Speed (average velocity achieved)
   - Smoothness (acceleration jerks)

5. **Iterate**: Repeat for each parameter

---

## Performance Metrics

### CPU Usage

Monitor on Jetson Nano:
```bash
# Terminal 1
watch -n 0.5 'ps aux | grep custom_dwa'

# Typical:
# 5-10% CPU for obstacle_memory_node
# 15-25% CPU for custom_dwa_planner_node
# Total: ~30-35% of one core
```

### Latency

Measure planning delay:
```bash
# In custom_dwa_planner_node.py, add timing:
import time
start = time.time()
best_velocity = self.planner.plan(...)
elapsed = time.time() - start
rospy.loginfo(f"Planning took {elapsed*1000:.1f}ms")

# Typical: 30-50ms on Jetson Nano
# Goal: <100ms to plan at 10 Hz
```

---

## Troubleshooting Checklist

- [ ] Are both nodes running without errors?
  ```bash
  rosnode list | grep -E "(obstacle|dwa)"
  ```

- [ ] Is TF tree valid?
  ```bash
  rosrun tf2_tools view_frames.py
  ```

- [ ] Are detections being converted to global coordinates?
  ```bash
  rostopic echo /duckiebot_name/obstacle_memory_node/obstacles
  ```

- [ ] Is planner receiving path?
  ```bash
  rostopic echo /duckiebot_name/graph_search_server_node/path
  ```

- [ ] Is car_cmd being published?
  ```bash
  rostopic echo /duckiebot_name/car_cmd
  ```

- [ ] Are parameters loading correctly?
  ```bash
  rosparam get /duckiebot_name/obstacle_memory_node
  rosparam get /duckiebot_name/custom_dwa_planner_node
  ```

---

## Performance Optimization

If nodes are too slow for your system:

1. **Reduce sampling**:
   ```yaml
   velocity_resolution: 3    # From 5
   omega_resolution: 7       # From 9
   ```

2. **Shorter horizon**:
   ```yaml
   prediction_horizon: 0.5   # From 1.0
   ```

3. **Coarser simulation**:
   ```yaml
   simulation_dt: 0.2        # From 0.1
   ```

4. **Less frequent planning**:
   Edit `custom_dwa_planner_node.py`, change timer duration from 0.1 to 0.2

---

## Next Steps

1. Start with default configuration
2. Test in controlled environment
3. Note specific issues
4. Refer to appropriate scenario above
5. Adjust one parameter at a time
6. Retune if environment significantly changes

Good luck! 🦆
