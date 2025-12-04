#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Accel, Twist
from nav_msgs.msg import Odometry 
from trajectory_msgs.msg import MultiDOFJointTrajectory
from crazyflie_ros.controllers import DroneController
import numpy as np
from tf.transformations import euler_from_quaternion
from crazyflie_ros.cfg import ControllerConfig
import time

class Controller:
    def __init__(self, drone_id='cf'):


        rospy.init_node('controller_node', anonymous=True)
        

        # Subscribers
        rospy.Subscriber('/' + drone_id + '/odom', Odometry, self.pose_callback)
        rospy.Subscriber('/' + drone_id + '/setpoint', MultiDOFJointTrajectory, self.trajectory_callback)
        # Publisher for acceleration commands
        self.cmd_pub = rospy.Publisher('/' + drone_id + '/cmd_acc', Accel, queue_size=10)

        # Initialize variables
        self.current_pose = None
        self.current_velocity = None
        self.reference_trajectory = []

        #Initialize controller
        self.controller = DroneController()



        
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def trajectory_callback(self, msg):
        
        self.reference_trajectory = msg

        
    def run(self):
        rospy.loginfo("[Controller] Starting control loop...")
        rate = rospy.Rate(200)  # Hz

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(5, "[Controller] Control loop running...")

            if self.current_pose and self.reference_trajectory:
                cmd_msg = Accel()
                

                orientation_q = self.current_pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (_, _, current_yaw) = euler_from_quaternion(orientation_list)
                
                current_state = np.array([
                    self.current_pose.position.x, 
                    self.current_pose.position.y, 
                    self.current_pose.position.z,
                    self.current_velocity.linear.x, 
                    self.current_velocity.linear.y, 
                    self.current_velocity.linear.z,
                    current_yaw
                ])

                reference_states = []

                point = self.reference_trajectory

                pos = point.transforms[0].translation
                vel = point.velocities[0].linear
                
                # Convert quaternion to yaw for this reference point
                ref_q = point.transforms[0].rotation
                (_, _, ref_yaw) = euler_from_quaternion([ref_q.x, ref_q.y, ref_q.z, ref_q.w])

                # Append the state vector for this point
                reference_states.append(np.array([
                    pos.x, pos.y, pos.z,
                    vel.x, vel.y, vel.z,
                    ref_yaw
                ]))

                if reference_states:
                    ax,ay,az, yaw_rate_cmd = self.controller.solve(current_state, reference_states) 
                    # Populate the Accel message
                    cmd_msg.linear.x = np.clip(ax, -7.0, 7.0)
                    cmd_msg.linear.y = np.clip(ay, -7.0, 7.0)
                    cmd_msg.linear.z = np.clip(az, -7.0, 7.0)
                    cmd_msg.angular.z = np.clip(yaw_rate_cmd,-200, 200) 


                    self.cmd_pub.publish(cmd_msg)
                else:
                    rospy.logwarn_throttle(5, "[Controller] Reference states are empty.")

                    
            rate.sleep()

if __name__ == '__main__':
    #time.sleep(1)  # Give ROS time to initialize
    try:
        controller = Controller()
        controller.run()
    except rospy.ROSInterruptException:
        pass