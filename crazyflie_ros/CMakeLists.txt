cmake_minimum_required(VERSION 3.0.2)
project(crazyflie_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  trajectory_msgs
  nav_msgs
  rospy
  std_msgs
  dynamic_reconfigure
)

add_service_files(
  FILES
  PlanTrajectory.srv
)




# This makes sure setup.py is considered
catkin_python_setup()




generate_messages(
  DEPENDENCIES
  geometry_msgs
)




catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES crazyflie_ros
  CATKIN_DEPENDS geometry_msgs message_runtime rospy std_srvs trajectory_msgs
#  DEPENDS system_lib
)


catkin_install_python(PROGRAMS
  scripts/controller_node.py
  scripts/driver_node.py
  scripts/trajectory_node.py
  scripts/trajectory_node2.py
  scripts/trajectory_plot_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
