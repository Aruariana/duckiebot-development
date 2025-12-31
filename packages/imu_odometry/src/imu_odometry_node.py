#!/usr/bin/env python3

import time
import math
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, Pose, Point, Vector3, TransformStamped, Transform

# For Rviz path visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from tf import transformations as tr


class ImuOdometryNode(DTROS):
    """Performs odometry using IMU.
    The node performs odometry estimation based upon IMU values.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~veh (:obj:`str`): Robot name
        ~publish_hz (:obj:`float`): Frequency at which to publish odometry
        ~imu_stale_dt (:obj:`float`): Time in seconds after IMU is considered stale
        ~debug (:obj: `bool`): Enable/disable debug output

    Publisher:
        ~odom (:obj:`Odometry`): The computed odometry

    Subscribers:
        ~imu_node/data (:obj:`Imu`): IMU data
    """

    def __init__(self, node_name):
        super(ImuOdometryNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.node_name = node_name

        self.veh = rospy.get_param("~veh")
        self.publish_hz = rospy.get_param("~publish_hz")
        self.imu_stale_dt = rospy.get_param("~imu_stale_dt")
        self.origin_frame = rospy.get_param("~origin_frame").replace("~", self.veh)
        self.target_frame = rospy.get_param("~target_frame").replace("~", self.veh)
        self.debug = rospy.get_param("~debug", False)

        self.imu_last = None
        self.imu_timestamp_last = None
        self.imu_timestamp_last_local = None

        # Current pose, forward velocity, and angular rate
        self.timestamp = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.q = [0.0, 0.0, 0.0, 1.0]
        self.tv = 0.0
        self.rv = 0.0

        # Velocity components
        self.vx = 0.0
        self.vy = 0.0

        # Used for debugging
        self.x_trajectory = []
        self.y_trajectory = []
        self.yaw_trajectory = []
        self.time = []

        self.total_dist = 0

        # Setup subscriber
        self.sub_imu = rospy.Subscriber("~imu", Imu, self.cb_imu)

        # Setup publishers
        self.pub = rospy.Publisher("~odom", Odometry, queue_size=10)

        # Path publisher for Rviz visualization
        self.path_pub = rospy.Publisher("~path", Path, queue_size=1, latch=True)
        self.path = Path()

        # Setup timer
        self.timer = rospy.Timer(rospy.Duration(1 / self.publish_hz), self.cb_timer)
        self._print_time = 0
        self._print_every_sec = 30

        # tf broadcaster for odometry TF
        self._tf_broadcaster = TransformBroadcaster()

        self.loginfo("Initialized")

    def cb_imu(self, imu_msg):
        timestamp_now = rospy.get_time()
        timestamp = imu_msg.header.stamp.to_sec()

        if not self.imu_last:
            self.imu_last = imu_msg
            self.imu_timestamp_last = timestamp
            self.imu_timestamp_last_local = timestamp_now
            return

        # Skip if older message
        dt_stamp = imu_msg.header.stamp - self.imu_last.header.stamp
        if dt_stamp.to_sec() < 0:
            self.loginfo("Ignoring stale IMU message")
            return

        dt = timestamp - self.imu_timestamp_last

        if dt < 1e-6:
            self.logwarn("Time since last IMU message (%f) is too small. Ignoring" % dt)
            return

        # Angular velocity around z-axis (yaw rate)
        rv_z = imu_msg.angular_velocity.z

        # Linear accelerations
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        # Integrate angular velocity for yaw
        dyaw = rv_z * dt
        self.yaw = self.angle_clamp(self.yaw + dyaw)

        # Integrate linear acceleration for velocity (simple integration, assuming gravity compensated)
        self.vx += ax * dt
        self.vy += ay * dt

        # Integrate velocity for position
        dx = self.vx * dt
        dy = self.vy * dt

        self.x += dx
        self.y += dy

        # Update quaternion
        self.q = tr.quaternion_from_euler(0, 0, self.yaw)

        # Translational velocity (magnitude)
        self.tv = math.sqrt(self.vx**2 + self.vy**2)
        self.rv = rv_z

        if self.debug:
            self.loginfo(
                "IMU:\t Time = %.4f\t AX = %.4f\t AY = %.4f\t RV_Z = %.4f"
                % (timestamp, ax, ay, rv_z)
            )
            self.loginfo(
                "TV = %.2f m/s\t RV = %.2f deg/s\t DT = %.4f" % (self.tv, self.rv * 180 / math.pi, dt)
            )

        self.timestamp = timestamp
        self.imu_last = imu_msg
        self.imu_timestamp_last = timestamp
        self.imu_timestamp_last_local = timestamp_now

    def cb_timer(self, _):
        need_print = time.time() - self._print_time > self._print_every_sec
        if self.imu_timestamp_last:
            dt = rospy.get_time() - self.imu_timestamp_last_local
            if abs(dt) > self.imu_stale_dt:
                if need_print:
                    self.logwarn(
                        "No IMU messages received for %.2f seconds. "
                        "Setting velocities to zero" % dt
                    )
                self.rv = 0.0
                self.tv = 0.0
                self.vx = 0.0
                self.vy = 0.0
        else:
            if need_print:
                self.logwarn(
                    "No IMU messages received. " "Setting velocities to zero"
                )
            self.rv = 0.0
            self.tv = 0.0
            self.vx = 0.0
            self.vy = 0.0

        # Publish the odometry message
        self.publish_odometry()
        if need_print:
            self._print_time = time.time()

    def publish_odometry(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.origin_frame
        odom.pose.pose = Pose(Point(self.x, self.y, self.z), Quaternion(*self.q))
        odom.child_frame_id = self.target_frame
        odom.twist.twist = Twist(Vector3(self.vx, self.vy, 0.0), Vector3(0.0, 0.0, self.rv))

        self.pub.publish(odom)

        self._tf_broadcaster.sendTransform(
            TransformStamped(
                header=odom.header,
                child_frame_id=self.target_frame,
                transform=Transform(
                    translation=Vector3(self.x, self.y, self.z), rotation=Quaternion(*self.q)
                ),
            )
        )
        
        # Publish path for Rviz visualization
        self.path.header = odom.header
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)    

    @staticmethod
    def angle_clamp(theta):
        if theta > 2 * math.pi:
            return theta - 2 * math.pi
        elif theta < -2 * math.pi:
            return theta + 2 * math.pi
        else:
            return theta


if __name__ == "__main__":
    # create node
    node = ImuOdometryNode("imu_odometry_node")
    rospy.spin()
    # ---
    rospy.signal_shutdown("done")