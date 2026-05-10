cmake_minimum_required(VERSION 3.0.2)
project(platooning_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  duckiebot_msgs
  lane_control
)

catkin_package()

catkin_install_python(PROGRAMS
  src/platooning_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
