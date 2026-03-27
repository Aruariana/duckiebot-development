"""Projection helper utilities for image -> ground conversions.

This module provides small, reusable functions used by nodes that need
to load extrinsic homographies and convert normalized image points to
ground coordinates.

Place this file under the `image_processing` package so it can be
imported as `image_processing.projection_utils` by consumers.
"""
from typing import Optional
import os

import yaml
import rospy
import numpy as np
from geometry_msgs.msg import Point as PointMsg

from .ground_projection_geometry import Point as GPPoint


def load_extrinsics(namespace: Optional[str] = None, cali_file_folder: str = "/data/config/calibrations/camera_extrinsic/") -> np.ndarray:
    """
    Load homography matrix for the given namespace.

    Args:
        namespace: ROS namespace (defaults to `rospy.get_namespace()`)
        cali_file_folder: folder where calibration yamls are stored

    Returns:
        homography as a numpy array (3x3)

    Behavior mirrors existing node logic: if a specific file is not
    found, it falls back to `default.yaml`. If no calibration is
    found, it will log an error and trigger a ROS shutdown.
    """
    ns = namespace if namespace is not None else rospy.get_namespace().strip("/")
    cali_file = os.path.join(cali_file_folder, f"{ns}.yaml")

    if not os.path.isfile(cali_file):
        rospy.logwarn(f"Can't find calibration file: {cali_file}. Using default calibration instead.")
        cali_file = os.path.join(cali_file_folder, "default.yaml")

    if not os.path.isfile(cali_file):
        msg = "Found no calibration file ... aborting"
        rospy.logerr(msg)
        rospy.signal_shutdown(msg)

    try:
        with open(cali_file, "r") as stream:
            calib_data = yaml.load(stream, Loader=yaml.Loader)
    except yaml.YAMLError:
        msg = f"Error in parsing calibration file {cali_file} ... aborting"
        rospy.logerr(msg)
        rospy.signal_shutdown(msg)

    return np.array(calib_data["homography"]).reshape((3, 3))


def pixel_msg_to_ground_msg(point_msg, ground_projector, rectifier) -> PointMsg:
    """Convert a normalized point message to a ground-coordinate `Point` message.

    The function performs the same sequence used across existing nodes:
    1. Convert normalized message to internal `Point` representation
    2. Convert normalized vector to absolute pixel coordinates
    3. Rectify (undistort) the pixel
    4. Project the rectified pixel to ground using the homography

    Args:
        point_msg: a geometry_msgs-like message with `x` and `y` normalized coords
        ground_projector: instance of `GroundProjectionGeometry`
        rectifier: instance of `Rectify`

    Returns:
        geometry_msgs.msg.Point populated with ground coordinates
    """
    # normalized coordinates to absolute pixel
    norm_pt = GPPoint.from_message(point_msg)
    pixel = ground_projector.vector2pixel(norm_pt)

    # rectify
    rect = rectifier.rectify_point(pixel)
    rect_pt = GPPoint.from_message(rect)

    # project on ground
    ground_pt = ground_projector.pixel2ground(rect_pt)

    # convert to ROS message
    ground_pt_msg = PointMsg()
    ground_pt_msg.x = ground_pt.x
    ground_pt_msg.y = ground_pt.y
    ground_pt_msg.z = ground_pt.z

    return ground_pt_msg
