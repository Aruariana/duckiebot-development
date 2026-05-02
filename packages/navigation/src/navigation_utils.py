#!/usr/bin/env python3
"""
Utility functions for custom navigation stack on Duckiebot.

This module provides helper functions for:
- DWA planning utilities (trajectory visualization, cost visualization)
- Obstacle manipulation (detection merging, motion prediction)
- Performance metrics and debugging
"""

import numpy as np
from typing import List, Tuple, Optional
import rospy
from geometry_msgs.msg import Point as PointMsg
from visualization_msgs.msg import Marker, MarkerArray
from duckietown.dtros import DTROS


def trajectory_to_marker_array(trajectory_points: List, 
                              frame_id: str = "map",
                              color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                              ns: str = "trajectory") -> MarkerArray:
    """
    Convert a list of trajectory points to a RViz MarkerArray for visualization.
    
    Args:
        trajectory_points: List of TrajectoryPoint objects with x, y, theta
        frame_id: TF frame for visualization
        color: RGB color tuple (0.0-1.0 range)
        ns: Marker namespace
        
    Returns:
        visualization_msgs.msg.MarkerArray
    """
    marker_array = MarkerArray()
    
    # Line strip connecting all points
    line_marker = Marker()
    line_marker.header.frame_id = frame_id
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = ns
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    
    line_marker.pose.orientation.w = 1.0
    line_marker.scale.x = 0.01  # Line width
    
    line_marker.color.r = color[0]
    line_marker.color.g = color[1]
    line_marker.color.b = color[2]
    line_marker.color.a = 0.8
    
    for point in trajectory_points:
        msg_point = PointMsg()
        msg_point.x = point.x
        msg_point.y = point.y
        msg_point.z = 0.0
        line_marker.points.append(msg_point)
    
    marker_array.markers.append(line_marker)
    
    # Start point (green circle)
    if trajectory_points:
        start_marker = Marker()
        start_marker.header.frame_id = frame_id
        start_marker.header.stamp = rospy.Time.now()
        start_marker.ns = ns
        start_marker.id = 1
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        
        start_marker.pose.position.x = trajectory_points[0].x
        start_marker.pose.position.y = trajectory_points[0].y
        start_marker.pose.position.z = 0.0
        start_marker.pose.orientation.w = 1.0
        
        start_marker.scale.x = 0.05
        start_marker.scale.y = 0.05
        start_marker.scale.z = 0.05
        
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        
        marker_array.markers.append(start_marker)
        
        # End point (red circle)
        end_marker = Marker()
        end_marker.header.frame_id = frame_id
        end_marker.header.stamp = rospy.Time.now()
        end_marker.ns = ns
        end_marker.id = 2
        end_marker.type = Marker.SPHERE
        end_marker.action = Marker.ADD
        
        end_marker.pose.position.x = trajectory_points[-1].x
        end_marker.pose.position.y = trajectory_points[-1].y
        end_marker.pose.position.z = 0.0
        end_marker.pose.orientation.w = 1.0
        
        end_marker.scale.x = 0.05
        end_marker.scale.y = 0.05
        end_marker.scale.z = 0.05
        
        end_marker.color.r = 1.0
        end_marker.color.g = 0.0
        end_marker.color.b = 0.0
        end_marker.color.a = 1.0
        
        marker_array.markers.append(end_marker)
    
    return marker_array


def compute_path_length(trajectory_points: List) -> float:
    """
    Compute total path length of a trajectory.
    
    Args:
        trajectory_points: List of TrajectoryPoint objects
        
    Returns:
        Total path length in meters
    """
    if len(trajectory_points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(trajectory_points)):
        dx = trajectory_points[i].x - trajectory_points[i-1].x
        dy = trajectory_points[i].y - trajectory_points[i-1].y
        total_length += np.sqrt(dx*dx + dy*dy)
    
    return total_length


def compute_curvature(trajectory_points: List) -> float:
    """
    Compute average curvature of a trajectory (average heading change rate).
    
    Lower curvature = smoother path
    
    Args:
        trajectory_points: List of TrajectoryPoint objects with theta
        
    Returns:
        Average curvature in 1/meters
    """
    if len(trajectory_points) < 2:
        return 0.0
    
    heading_changes = []
    for i in range(1, len(trajectory_points)):
        dtheta = trajectory_points[i].theta - trajectory_points[i-1].theta
        # Normalize to [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        heading_changes.append(abs(dtheta))
    
    if heading_changes:
        path_length = compute_path_length(trajectory_points)
        if path_length > 0:
            total_heading_change = sum(heading_changes)
            return total_heading_change / path_length
    
    return 0.0


def get_trajectory_statistics(trajectory_points: List) -> dict:
    """
    Compute statistics about a trajectory for analysis and debugging.
    
    Args:
        trajectory_points: List of TrajectoryPoint objects
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'length': compute_path_length(trajectory_points),
        'curvature': compute_curvature(trajectory_points),
        'num_points': len(trajectory_points),
        'min_x': min([p.x for p in trajectory_points]) if trajectory_points else 0,
        'max_x': max([p.x for p in trajectory_points]) if trajectory_points else 0,
        'min_y': min([p.y for p in trajectory_points]) if trajectory_points else 0,
        'max_y': max([p.y for p in trajectory_points]) if trajectory_points else 0,
    }
    
    if trajectory_points:
        heading_changes = []
        for i in range(1, len(trajectory_points)):
            dtheta = trajectory_points[i].theta - trajectory_points[i-1].theta
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            heading_changes.append(abs(dtheta))
        
        if heading_changes:
            stats['max_heading_change'] = max(heading_changes)
            stats['avg_heading_change'] = np.mean(heading_changes)
    
    return stats


def merge_nearby_obstacles(obstacles: List[Tuple], 
                          merge_distance: float = 0.1) -> List[Tuple]:
    """
    Merge obstacles that are very close to each other (likely same obstacle with noise).
    
    Args:
        obstacles: List of (x, y, radius) tuples
        merge_distance: Max distance to consider obstacles as same
        
    Returns:
        List of merged obstacles
    """
    if not obstacles:
        return []
    
    merged = []
    used = set()
    
    for i, (x1, y1, r1) in enumerate(obstacles):
        if i in used:
            continue
        
        # Find all nearby obstacles
        nearby_group = [(x1, y1, r1)]
        used.add(i)
        
        for j, (x2, y2, r2) in enumerate(obstacles):
            if j <= i or j in used:
                continue
            
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < merge_distance:
                nearby_group.append((x2, y2, r2))
                used.add(j)
        
        # Average the nearby obstacles
        avg_x = np.mean([obs[0] for obs in nearby_group])
        avg_y = np.mean([obs[1] for obs in nearby_group])
        avg_r = np.mean([obs[2] for obs in nearby_group])
        
        merged.append((avg_x, avg_y, avg_r))
    
    return merged


def compute_obstacle_distance_field(obstacles: List[Tuple],
                                   grid_origin: Tuple[float, float],
                                   grid_size: Tuple[int, int],
                                   cell_size: float) -> np.ndarray:
    """
    Compute a 2D distance field from obstacles for visualization.
    
    Distance field shows minimum distance to any obstacle at each grid cell.
    Can be used to visualize obstacle repulsion potential.
    
    Args:
        obstacles: List of (x, y, radius) tuples
        grid_origin: (x, y) of grid bottom-left corner
        grid_size: (width, height) of grid in cells
        cell_size: Size of each grid cell
        
    Returns:
        2D numpy array with distances
    """
    distance_field = np.full(grid_size, float('inf'))
    
    ox, oy = grid_origin
    gw, gh = grid_size
    
    for ix in range(gw):
        for iy in range(gh):
            # Cell position in world coords
            cx = ox + ix * cell_size
            cy = oy + iy * cell_size
            
            # Minimum distance to any obstacle
            min_dist = float('inf')
            for obs_x, obs_y, obs_r in obstacles:
                dist = np.sqrt((cx - obs_x)**2 + (cy - obs_y)**2)
                min_dist = min(min_dist, dist - obs_r)
            
            distance_field[iy, ix] = min_dist
    
    return distance_field
