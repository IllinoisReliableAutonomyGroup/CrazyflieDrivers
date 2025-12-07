#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Accel, Twist
from nav_msgs.msg import Odometry 
from crazyflie_ros.controllers import DroneController
import numpy as np
from tf.transformations import euler_from_quaternion
import time
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint # Add MultiDOFJointTrajectoryPoint


class TrajectoryPointWrapper:
            def __init__(self, point):
                self.points = [point]


class Controller:
    def __init__(self, drone_id='cf'):


        rospy.init_node('controller_node', anonymous=True)
        

        # Subscribers
        rospy.Subscriber('/' + drone_id + '/odom', Odometry, self.pose_callback)
        rospy.Subscriber('/' + drone_id + '/setpoint', MultiDOFJointTrajectoryPoint, self.trajectory_callback)
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
        
        #self.reference_trajectory = msg




        self.reference_trajectory = TrajectoryPointWrapper(msg)

        
    def run(self):
        rospy.loginfo("[Controller] Starting control loop...")
        rate = rospy.Rate(200)  # Hz

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(5, "[Controller] Control loop running...")

            # --- ADDED DEBUGGING LINES HERE ---
            pose_status = "Not None" if self.current_pose else "None"
            traj_status = "Not None" if self.reference_trajectory else "None"
            rospy.loginfo_throttle(1, f"[Controller Debug] Pose Status: {pose_status}, Traj Status: {traj_status}")
            # --- END OF ADDED DEBUGGING LINES ---

            if self.current_pose and self.reference_trajectory:
                rospy.loginfo_throttle(5, "inside controller loop base")
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

                point = self.reference_trajectory.points[0] 

                # Now access transforms and velocities from the 'point' object
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
                    rospy.loginfo_throttle(5, "inside controller loop")
                    ax,ay,az, yaw_rate_cmd = self.controller.solve(current_state, reference_states) 
                    # Populate the Accel message
                    cmd_msg.linear.x = np.clip(ax, -7.0, 7.0)
                    cmd_msg.linear.y = np.clip(ay, -7.0, 7.0)
                    cmd_msg.linear.z = np.clip(az, -7.0, 7.0)
                    cmd_msg.angular.z = np.clip(yaw_rate_cmd,-200, 200) 


                    self.cmd_pub.publish(cmd_msg)
                else:
                    rospy.logwarn_throttle(5, "[Controller] Reference states are empty.")

            else:

                rospy.loginfo_throttle(5, "failed")       
            rate.sleep()

if __name__ == '__main__':
    #time.sleep(1)  # Give ROS time to initialize
    try:
        controller = Controller()
        controller.run()
    except rospy.ROSInterruptException:
        pass