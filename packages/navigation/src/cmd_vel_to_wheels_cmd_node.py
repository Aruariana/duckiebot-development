#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import WheelsCmdStamped
from rosgraph_msgs.msg import Clock


class CmdVelToWheelsCmdNode:
    def __init__(self):
        self.node_name = rospy.get_name()
        
        # Parameters
        self.baseline = rospy.get_param("~baseline", 0.1)  # Distance between wheels (meters)
        self.radius = rospy.get_param("~radius", 0.0318)    # Wheel radius (meters)
        self.max_wheel_speed = rospy.get_param("~max_wheel_speed", 1.0)  # Max wheel speed
        
        # Subscriber
        self.sub_cmd_vel = rospy.Subscriber("~cmd_vel", Twist, self.cmd_vel_callback)
        
        # Publisher
        self.pub_wheels_cmd = rospy.Publisher("~wheels_cmd", WheelsCmdStamped, queue_size=1)
        
        rospy.loginfo(f"[{self.node_name}] Initialized. baseline={self.baseline}, radius={self.radius}")
    
    def cmd_vel_callback(self, msg):
        """
        Convert geometry_msgs/Twist to duckietown_msgs/WheelsCmdStamped
        
        Twist contains:
          - linear.x: forward velocity (m/s)
          - angular.z: rotational velocity (rad/s)
        
        WheelsCmdStamped contains:
          - vel_left: left wheel velocity
          - vel_right: right wheel velocity
        """
        # Extract velocities from Twist
        v_linear = msg.linear.x      # Forward velocity
        v_angular = msg.angular.z    # Rotational velocity
        
        # Compute wheel velocities using differential drive kinematics
        # For a differential drive robot:
        # v_left = (v_linear - (baseline/2) * v_angular) / radius
        # v_right = (v_linear + (baseline/2) * v_angular) / radius
        
        v_left = (v_linear - (self.baseline / 2.0) * v_angular) / self.radius
        v_right = (v_linear + (self.baseline / 2.0) * v_angular) / self.radius
        
        # Clamp to max speed
        v_left = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_left))
        v_right = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_right))
        
        # Create WheelsCmdStamped message
        wheels_cmd = WheelsCmdStamped()
        wheels_cmd.header.stamp = rospy.Time.now()
        wheels_cmd.vel_left = v_left
        wheels_cmd.vel_right = v_right
        
        self.pub_wheels_cmd.publish(wheels_cmd)
    
    def onShutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutdown.")


if __name__ == "__main__":
    rospy.init_node("cmd_vel_to_wheels_cmd_node")
    node = CmdVelToWheelsCmdNode()
    rospy.on_shutdown(node.onShutdown)
    rospy.spin()
