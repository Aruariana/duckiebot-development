#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped
from duckiebot_msgs.msg import DetectedObstacle

# Orijinal lane_control paketindeki kütüphaneyi kullanıyoruz!
from lane_controller.controller import LaneController

class PlatooningNode(DTROS):
    def __init__(self, node_name):
        super(PlatooningNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # --- Load Lane Control Parameters (tunable via DTParam) ---
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_theta"] = DTParam("~k_theta", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Iphi"] = DTParam("~k_Iphi", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~theta_thres_min"] = DTParam("~theta_thres_min", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0) 
        self.params["~theta_thres_max"] = DTParam("~theta_thres_max", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0) 
        
        # Static parameters (from rosparam)
        self.params["~d_thres"] = rospy.get_param("~d_thres", 0.2615)
        self.params["~d_offset"] = rospy.get_param("~d_offset", 0.0)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", {"d": {"top": 0.3, "bot": -0.3}, "phi": {"top": 1.2, "bot": -1.2}})
        self.params["~d_resolution"] = rospy.get_param("~d_resolution", 0.011)
        self.params["~phi_resolution"] = rospy.get_param("~phi_resolution", 0.051)
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", 0.0)
        
        # Platooning-specific parameters
        self.target_dist = rospy.get_param("~target_distance", 0.40)
        self.k_p_dist = rospy.get_param("~k_p_dist", 1.5)
        self.y_offset_enter = rospy.get_param("~y_offset_enter", 0.12)
        self.y_offset_exit = rospy.get_param("~y_offset_exit", 0.20)
        self.timeout_duration = rospy.get_param("~timeout_duration", 0.5)
        self.max_platoon_speed = rospy.get_param("~max_platoon_speed", 0.25)  # Safety limit for platooning
        self.min_platoon_speed = rospy.get_param("~min_platoon_speed", 0.0)

        # Temel Kontrolcü Objesi (lane_control kütüphanesinden)
        self.controller = LaneController(self.params)

        # State variables
        self.is_platooning = False
        self.last_duckiebot_time = rospy.Time.now()
        self.current_target_dist = None
        self.current_target_y = None
        self.prev_target_dist = None  # For stability checking
        self.last_s = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.detection_confidence = 0.0  # Track detection confidence

        # Publishers & Subscribers
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)
        
        self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.cbLanePose, queue_size=1)
        self.sub_detected_obstacle = rospy.Subscriber("~detected_obstacle", DetectedObstacle, self.cbDetectedObstacle, queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1)

        self.log("Platooning Node Initialized! Ready to follow ducks.")

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        self.wheels_cmd_executed = msg_wheels_cmd

    def cbDetectedObstacle(self, msg):
        """Update state machine with target vehicle detection from YOLO.
        
        Implements hysteresis logic with edge case detection:
        - ENTERING platooning: Only when duckiebot is well-centered (y_offset < enter threshold)
        - EXITING platooning: When duckiebot moves too far to side (y_offset > exit threshold)
                              OR when duckiebot distance is unstable (sudden increases)
        """
        if msg.detected and len(msg.objects) > 0:
            duckiebots = [obj for obj in msg.objects if obj.object_type == "duckiebot"]
            
            if duckiebots:
                closest_bot = min(duckiebots, key=lambda obj: obj.distance)
                self.current_target_dist = closest_bot.distance
                self.current_target_y = closest_bot.position.y
                self.last_duckiebot_time = rospy.Time.now()

                abs_y = abs(self.current_target_y)
                
                # Edge case detection: Check if leader robot is turning (moving out of lane)
                # This prevents the follower from leaving the lane during leader's turns
                is_leader_turning = abs_y > self.y_offset_exit
                
                # Edge case detection: Check distance stability
                # If distance suddenly increases significantly, leader might be turning away
                is_distance_unstable = False
                if self.prev_target_dist is not None:
                    dist_increase = self.current_target_dist - self.prev_target_dist
                    # If distance increases by more than 20% in one frame, it's unstable
                    if dist_increase > 0.05 and dist_increase > self.prev_target_dist * 0.2:
                        is_distance_unstable = True
                
                self.prev_target_dist = self.current_target_dist
                
                # --- HYSTERESIS LOGIC ---
                if not self.is_platooning:
                    # ENTER platooning mode: duckiebot must be well-centered in front
                    if abs_y < self.y_offset_enter and not is_distance_unstable:
                        self.is_platooning = True
                        self.log(f"Leader locked: Platooning ON (y_offset={abs_y:.3f}m, dist={self.current_target_dist:.3f}m)")
                else:
                    # EXIT platooning mode: when leader turns or distance becomes unstable
                    if is_leader_turning or is_distance_unstable:
                        self.is_platooning = False
                        reason = "turning" if is_leader_turning else "distance unstable"
                        self.log(f"Leader {reason}: Platooning OFF -> Fallback to Lane Following (y_offset={abs_y:.3f}m)")
            else:
                self.check_timeout()
        else:
            self.check_timeout()

    def check_timeout(self):
        """ Kamera frame kaçırırsa hemen modu bozma, bekle. """
        if self.is_platooning:
            time_since_last_seen = (rospy.Time.now() - self.last_duckiebot_time).to_sec()
            if time_since_last_seen > self.timeout_duration:
                self.is_platooning = False
                self.current_target_dist = None
                self.log("Lider uzun süredir kayıp (Timeout): Platooning OFF")

    def cbLanePose(self, pose_msg):
        """Compute control action at each lane pose update.
        
        The control strategy is:
        1. Always compute steering (omega) for lane centering
        2. Switch velocity (v) based on mode:
           - Lane Following: Use nominal velocity from lane controller
           - Platooning: Use adaptive cruise control based on distance to leader
        """
        current_s = rospy.Time.now().to_sec()
        dt = current_s - self.last_s if self.last_s is not None else None

        # Apply offsets to lane position errors
        d_err = pose_msg.d - self.params["~d_offset"]
        phi_err = pose_msg.phi

        # Clamp error signals to avoid extreme control actions
        if np.abs(d_err) > self.params["~d_thres"]:
            d_err = np.sign(d_err) * self.params["~d_thres"]
        if phi_err > self.params["~theta_thres_max"].value or phi_err < self.params["~theta_thres_min"].value:
            phi_err = np.maximum(self.params["~theta_thres_min"].value, np.minimum(phi_err, self.params["~theta_thres_max"].value))

        wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
        
        # Always compute steering omega to keep robot in lane
        v_nom, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, None)
        
        # --- SELECT VELOCITY BASED ON OPERATING MODE ---
        if self.is_platooning and self.current_target_dist is not None:
            # PLATOONING MODE: Adaptive Cruise Control
            # P-controller: v = v_base + k_p * (distance_error)
            dist_error = self.current_target_dist - self.target_dist
            v_acc = self.params["~v_bar"].value + (self.k_p_dist * dist_error)
            
            # Safety bounds: clamp velocity to safe range
            # Prevent backing up, respect maximum platooning speed
            v = np.clip(v_acc, self.min_platoon_speed, self.max_platoon_speed)
        else:
            # LANE FOLLOWING MODE: Use nominal velocity from controller
            v = v_nom
            
        # Add feedforward term
        omega += self.params["~omega_ff"]

        # Publish control command
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header
        car_control_msg.v = v
        car_control_msg.omega = omega
        
        self.pub_car_cmd.publish(car_control_msg)
        self.last_s = current_s

if __name__ == "__main__":
    node = PlatooningNode(node_name="platooning_node")
    rospy.spin()
