cmake_minimum_required(VERSION 3.0.2)
project(crazyflie_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  dynamic_reconfigure
  trajectory_msgs
)



# This makes sure setup.py is considered
catkin_python_setup()




catkin_package()



catkin_install_python(PROGRAMS
  scripts/controller_node.py
  scripts/driver_node.py
  scripts/trajectory_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
