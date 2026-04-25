#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tr
from duckietown_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

class AprilTagLocalization(object):
    def __init__(self):
        self.node_name = "apriltag_localization_node"
        rospy.init_node(self.node_name, anonymous=False)

        # Camera to Vehicle (Base Footprint) extrinsics
        self.camera_x = rospy.get_param("~camera_x", 0.065)
        self.camera_y = rospy.get_param("~camera_y", 0.0)
        self.camera_z = rospy.get_param("~camera_z", 0.11)
        self.camera_theta = rospy.get_param("~camera_theta", 19.0)

        # KNOWN MAP LOCATIONS: Dictionary mapping tag_id to (x, y, z, yaw_in_radians)
        self.known_tags = {
            # Example: Tag 11 is at X=0.25m, Y=0.0m, Z=0.08m (8cm off ground), facing forward
            11: (0.25, 0.0, 0.08, 3.14159)    
        }

        # Publish tag markers for RVIZ visualization
        self.pub_markers = rospy.Publisher("~tag_markers", MarkerArray, queue_size=1, latch=True)
        self._publish_tag_markers()

        # Subscribers and Publishers
        self.sub_detections = rospy.Subscriber(
            "~detections", AprilTagDetectionArray, self.callback, queue_size=1
        )
        self.pub_pose = rospy.Publisher(
            "~pose", PoseStamped, queue_size=1
        )

        rospy.loginfo("[%s] ready to correct map position.", self.node_name)

    def callback(self, msg):
        for detection in msg.detections:
            tag_id = int(detection.tag_id)

            rospy.loginfo("[%s] Detected tag ID: %d", self.node_name, tag_id)

            # Ignore tags we don't know the absolute map location of
            if tag_id not in self.known_tags:
                continue

            # 1. Transform: Vehicle to Camera
            veh_t_cam = tr.translation_matrix((self.camera_x, self.camera_y, self.camera_z))
            veh_R_cam = tr.euler_matrix(0, self.camera_theta * np.pi / 180, 0, "rxyz")
            veh_T_cam = tr.concatenate_matrices(veh_t_cam, veh_R_cam)
            
            # Duckietown camera frame correction (z out, x right, y down)
            cam_T_optical = tr.euler_matrix(-np.pi / 2, 0, -np.pi / 2, "rzyx")
            veh_T_optical = tr.concatenate_matrices(veh_T_cam, cam_T_optical)

            # 2. Transform: Optical to Tag
            trans = detection.transform.translation
            rot = detection.transform.rotation
            opt_t_tag = tr.translation_matrix((trans.x, trans.y, trans.z))
            opt_R_tag = tr.quaternion_matrix((rot.x, rot.y, rot.z, rot.w))
            opt_T_tag = tr.concatenate_matrices(opt_t_tag, opt_R_tag)

            # Converts from Apriltag (Z-in, X-right, Y-down) to Map (X-out, Y-right, Z-up)
            tagzout_T_tagxout = np.array([
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0]
            ])

            # Calculate: Vehicle -> Tag 
            veh_T_tag = tr.concatenate_matrices(veh_T_optical, opt_T_tag, tagzout_T_tagxout)

            # 3. Transform: Map to Tag (from our known dictionary)
            map_x, map_y, map_z, map_yaw = self.known_tags[tag_id]
            map_t_tag = tr.translation_matrix((map_x, map_y, map_z)) 
            map_R_tag = tr.euler_matrix(0, 0, map_yaw, "sxyz")
            map_T_tag = tr.concatenate_matrices(map_t_tag, map_R_tag)

            # 4. Calculate: Map -> Vehicle
            # If map->tag = map->veh * veh->tag, then map->veh = map->tag * inverse(veh->tag)
            veh_T_tag_inv = tr.inverse_matrix(veh_T_tag)
            map_T_veh = tr.concatenate_matrices(map_T_tag, veh_T_tag_inv)

            # Extract final position and rotation
            (final_x, final_y, final_z) = tr.translation_from_matrix(map_T_veh)
            (qx, qy, qz, qw) = tr.quaternion_from_matrix(map_T_veh)

            # 5. Publish Pose
            pose_msg = PoseStamped()

            pose_msg.header.stamp = msg.header.stamp 
            pose_msg.header.frame_id = rospy.get_namespace().strip('/') + "/map"

            pose_msg.pose.position.x = final_x
            pose_msg.pose.position.y = final_y
            pose_msg.pose.position.z = 0.0 # Force 2D

            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            self.pub_pose.publish(pose_msg)
    
    def _publish_tag_markers(self):
        marker_array = MarkerArray()
        for tag_id, (x, y, z, yaw) in self.known_tags.items():
            marker = Marker()
            marker.header.frame_id = rospy.get_namespace().strip('/') + "/map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "known_tags"
            marker.id = tag_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            
            # Orientation
            q = tr.quaternion_from_euler(0, 0, yaw)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            
            # Size (e.g., 6.5cm tag)
            marker.scale.x = 0.01
            marker.scale.y = 0.065
            marker.scale.z = 0.065
            
            # Color (Cyan)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
        self.pub_markers.publish(marker_array)

if __name__ == "__main__":
    node = AprilTagLocalization()
    rospy.spin()