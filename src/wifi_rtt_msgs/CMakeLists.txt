cmake_minimum_required(VERSION 2.8.3)
project(wifi_rtt_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

add_message_files(
  FILES
  RTT_Raw.msg
  RTT_Raw_Array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime
)
