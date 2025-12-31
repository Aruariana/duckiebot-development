cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs
  duckiebot_msgs
  cv_bridge
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)