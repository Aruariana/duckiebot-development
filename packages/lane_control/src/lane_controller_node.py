#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)
from duckiebot_msgs.msg import DetectedObstacle

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_theta"] = DTParam(
            "~k_theta", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Iphi"] = DTParam(
            "~k_Iphi", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        #self.params["~theta_thres"] = rospy.get_param("~theta_thres", None)
        #Breaking up the self.params["~theta_thres"] parameter for more finer tuning of phi
        self.params["~theta_thres_min"] = DTParam("~theta_thres_min", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)  #SUGGESTION mandatorizing the use of DTParam inplace of rospy.get_param for parameters in the entire dt-core repository as it allows active tuning while Robot is in action.
        self.params["~theta_thres_max"] = DTParam("~theta_thres_max", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0) 
        self.params["~d_thres"] = rospy.get_param("~d_thres", None)
        self.params["~d_offset"] = rospy.get_param("~d_offset", None)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", None)
        self.params["~d_resolution"] = rospy.get_param("~d_resolution", None)
        self.params["~phi_resolution"] = rospy.get_param("~phi_resolution", None)
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", None)
        self.params["~verbose"] = rospy.get_param("~verbose", None)
        self.params["~stop_line_slowdown"] = rospy.get_param("~stop_line_slowdown", None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        
        # Stop line logic (separate from object detection)
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        # Modular detected objects storage - extensible for new object types
        # Each object type stores: distance, position, detected flag, and two distance thresholds
        self.detected_objects = {
            'duckie': {
                'distance': None,
                'position': None,  # geometry_msgs/Point: x, y, z coordinates
                'detected': False,
                'distance_threshold_stop': 0.1,      # Distance to completely stop
                'distance_threshold_slowdown': 0.3   # Distance to start slowing down
            }
            # Future: Add new object types here
            # 'duckiebot_ahead': {
            #     'distance': None, 'position': None, 'detected': False,
            #     'distance_threshold_stop': 0.3, 'distance_threshold_slowdown': 0.6
            # }
        }

        self.current_pose_source = "lane_filter"

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber(
            "~lane_pose", LanePose, self.cbAllPoses, "lane_filter", queue_size=1
        )
        self.sub_intersection_navigation_pose = rospy.Subscriber(
            "~intersection_navigation_pose",
            LanePose,
            self.cbAllPoses,
            "intersection_navigation",
            queue_size=1,
        )
        self.sub_wheels_cmd_executed = rospy.Subscriber(
            "~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1
        )
        self.sub_stop_line = rospy.Subscriber(
            "~stop_line_reading", StopLineReading, self.cbStopLineReading, queue_size=1
        )
        self.sub_obstacle_stop_line = rospy.Subscriber(
            "~obstacle_distance_reading", StopLineReading, self.cbObstacleStopLineReading, queue_size=1
        )
        self.sub_detected_obstacle = rospy.Subscriber(
            "~detected_obstacle", DetectedObstacle, self.cbDetectedObstacle, queue_size=1
        )

        self.log("Initialized!")

    def cbObstacleStopLineReading(self, msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def cbDetectedObstacle(self, msg):
        """
        Callback processing detected obstacles and updating detection state for each object type.
        
        This is a modular callback that handles any object type present in the detected_objects dict.
        New object types can be added to self.detected_objects, and this callback will automatically
        process them without requiring modification.

        Args:
            msg(:obj:`DetectedObstacle`): Message containing information about detected obstacles.
        """
        # Reset all tracked objects to not detected
        for obj_type in self.detected_objects:
            self.detected_objects[obj_type]['detected'] = False
            self.detected_objects[obj_type]['distance'] = None
            self.detected_objects[obj_type]['position'] = None

        # Process incoming detections if available
        if not msg.detected or len(msg.objects) == 0:
            return

        # Group detected objects by type
        objects_by_type = {}
        for obj in msg.objects:
            if obj.object_type not in objects_by_type:
                objects_by_type[obj.object_type] = []
            objects_by_type[obj.object_type].append(obj)

        # Update tracked objects - store the closest object of each type
        for obj_type, objects in objects_by_type.items():
            if obj_type in self.detected_objects:
                # Find closest object of this type
                closest_obj = min(objects, key=lambda obj: obj.distance)

                self.detected_objects[obj_type]['distance'] = closest_obj.distance
                self.detected_objects[obj_type]['position'] = closest_obj.position
                self.detected_objects[obj_type]['detected'] = True

                if self.params["~verbose"] == 2:
                    self.log(f"Detected {obj_type}: distance={closest_obj.distance:.3f}m at position ({closest_obj.position.x:.3f}, {closest_obj.position.y:.3f}, {closest_obj.position.z:.3f})")


    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state  # String of current FSM state

        if self.fsm_state == "INTERSECTION_CONTROL":
            self.current_pose_source = "intersection_navigation"
        else:
            self.current_pose_source = "lane_filter"

        if self.params["~verbose"] == 2:
            self.log("Pose source: %s" % self.current_pose_source)

    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.

        If the source of the message corresponds with the current wanted pose source, it computes a control command.

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """

        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg

            self.pose_msg = input_pose_msg

            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = current_s - self.last_s

        # === STOP LINE LOGIC (separate from object detection) ===
        # Stop if at stop line (keep original separate logic)
        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        # === DETECTED OBJECTS LOGIC (modular) ===
        # Stop if at any detected object (distance below stop threshold)
        elif any(obj_info['detected'] and obj_info['distance'] < obj_info['distance_threshold_stop']
                 for obj_info in self.detected_objects.values()):
            v = 0
            omega = 0
        else:
            # Compute errors
            d_err = pose_msg.d - self.params["~d_offset"]
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params["~d_thres"]:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"]
            
            if phi_err > self.params["~theta_thres_max"].value or phi_err < self.params["~theta_thres_min"].value:
                self.log("phi_err too large/small, thresholding it!", "error")
                phi_err = np.maximum(self.params["~theta_thres_min"].value, np.minimum(phi_err, self.params["~theta_thres_max"].value))

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            
            # === Determine which obstacle distance to use for control action ===
            distance_for_control = self.stop_line_distance
            using_detected_object = False
            apply_slowdown = False
            
            # Priority 1: Check detected objects (modular - works for any object type)
            closest_detected_distance = float('inf')
            for obj_type, obj_info in self.detected_objects.items():
                if obj_info['detected'] and obj_info['distance'] < closest_detected_distance:
                    closest_detected_distance = obj_info['distance']
                    distance_for_control = obj_info['distance']
                    using_detected_object = True
                    # Check if we should apply slowdown (within slowdown threshold but above stop threshold)
                    if obj_info['distance'] < obj_info['distance_threshold_slowdown']:
                        apply_slowdown = True
                    if self.params["~verbose"] == 2:
                        self.log(f"Using {obj_type} distance for control: {obj_info['distance']:.3f}m")
            
            # Priority 2: If no detected object, check obstacle stop line
            if not using_detected_object and self.obstacle_stop_line_detected:
                distance_for_control = self.obstacle_stop_line_distance
                using_detected_object = True
                apply_slowdown = True
                if self.params["~verbose"] == 2:
                    self.log(f"Using obstacle stop line distance for control: {self.obstacle_stop_line_distance:.3f}m")
            
            # Compute control action
            v, omega = self.controller.compute_control_action(
                d_err, phi_err, dt, wheels_cmd_exec, distance_for_control
            )
            
            # Apply slowdown if within slowdown threshold of any detected object
            # TODO: This could be made configurable per object type in the future
            if apply_slowdown:
                v = v * 0.25
                omega = omega * 0.25
                if self.params["~verbose"] == 2:
                    self.log("Applying slowdown for detected obstacle")

            # For feedforward action (i.e. during intersection navigation)
            omega += self.params["~omega_ff"]

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header

        # Add commands to car message
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name="lane_controller_node")
    # Keep it spinning
    rospy.spin()
