#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from duckietown_msgs.srv import ChangePattern
from duckiebot_msgs.msg import DuckieObstacle


class LEDSupervisorNode(DTROS):
    """Supervisor node for LED patterns based on motion and safe-following state."""

    def __init__(self, node_name):
        super(LEDSupervisorNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.omega_threshold = rospy.get_param("~omega_threshold", 0.5)
        self.velocity_threshold = rospy.get_param("~velocity_threshold", 0.01)

        self.current_car_cmd = None
        self.current_obstacle = None
        self.current_pattern = None
        self.hazard_timer = None
        self.hazard_suppressed = False

        self.set_pattern_proxy = rospy.ServiceProxy("~set_pattern", ChangePattern)

        self.sub_car_cmd = rospy.Subscriber("~car_cmd", Twist2DStamped, self.cbCarCmd, queue_size=1)
        self.sub_obstacle = rospy.Subscriber("~duckie_obstacle", DuckieObstacle, self.cbObstacle, queue_size=1)

        self.log("Waiting for initial commands...")
        self.apply_pattern("CAR_DRIVING")
        self.log("Initialized.")

    def cbCarCmd(self, msg):
        self.current_car_cmd = msg
        self.decide_pattern()

    def cbObstacle(self, msg):
        self.current_obstacle = msg
        self.decide_pattern()

    def decide_pattern(self):
        if self.current_car_cmd is None:
            return

        v = self.current_car_cmd.v
        omega = self.current_car_cmd.omega
        obstacle_detected = bool(self.current_obstacle and self.current_obstacle.detected)

        if omega > self.omega_threshold:
            target_pattern = "CAR_SIGNAL_LEFT"
            self.cancel_hazard_timer()
            self.hazard_suppressed = False
        elif omega < -self.omega_threshold:
            target_pattern = "CAR_SIGNAL_RIGHT"
            self.cancel_hazard_timer()
            self.hazard_suppressed = False
        else:
            if abs(v) < self.velocity_threshold and obstacle_detected and not self.hazard_suppressed:
                target_pattern = "CAR_HAZARD"
                self.start_hazard_timer()
            else:
                target_pattern = "CAR_DRIVING"
                self.cancel_hazard_timer()
                self.hazard_suppressed = False

        self.apply_pattern(target_pattern)

    def apply_pattern(self, pattern_name):
        if self.current_pattern == pattern_name:
            return

        try:
            msg = String()
            msg.data = pattern_name
            self.set_pattern_proxy(msg)
            self.current_pattern = pattern_name
            self.log(f"Set LED pattern to {pattern_name}")
        except (rospy.ServiceException, rospy.ROSException) as e:
            self.log(f"Failed to set LED pattern '{pattern_name}': {e}", type="err")

    def start_hazard_timer(self):
        if self.hazard_timer is not None:
            return

        self.hazard_timer = rospy.Timer(rospy.Duration(2.0), self.on_hazard_timeout, oneshot=True)

    def cancel_hazard_timer(self):
        if self.hazard_timer is not None:
            self.hazard_timer.shutdown()
            self.hazard_timer = None

    def on_hazard_timeout(self, event):
        self.hazard_timer = None
        self.hazard_suppressed = True
        if self.current_pattern == "CAR_HAZARD":
            self.apply_pattern("CAR_DRIVING")


if __name__ == "__main__":
    node = LEDSupervisorNode(node_name="led_supervisor_node")
    rospy.spin()
