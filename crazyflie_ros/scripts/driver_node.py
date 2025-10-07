#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Accel 
import cflib.crtp
from crazyflie_ros.drone import CrazyflieController
from tf.transformations import euler_from_quaternion
import time
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

class Driver:
    def __init__(self, uri = 'radio://0/80/2M',drone_id = 'cf', mass = 0.027, hover_thrust = 42000, k_thrust = 0.000081):
        rospy.init_node('cf', anonymous=True)
        


        self.mass = rospy.get_param('~mass',mass)
        self.hover_thrust = rospy.get_param('~HOVER_THRUST_CF',hover_thrust)
        self.k_thrust = rospy.get_param('~K_THRUST',k_thrust)

        self.cf = CrazyflieController(uri)
        self.roll = None
        self.pitch = None
        self.yaw = None
            
        # Subscriber to get control commands
        rospy.Subscriber('/'+ drone_id +'/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/' + drone_id + '/cmd_acc', Accel, self.cmd_callback)

    def pose_callback(self, msg):
        self.current_pose = msg.pose

        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)
        


    def cmd_callback(self, msg):
        # --- MODIFIED SECTION ---
        if self.roll is None or self.pitch is None or self.yaw is None:
            rospy.logwarn("[Driver] No attitude data received yet. Skipping command.")
            return

        self.ax = msg.linear.x
        self.ay = msg.linear.y
        self.az = msg.linear.z
        self.yaw_rate = msg.angular.z 
        rospy.loginfo(f"[Driver] Acceleration Command Received: ax={self.ax}, ay={self.ay}, az={self.az}, yaw_rate={self.yaw_rate}")
        self.cf.send_acceleration_command(-self.ax,-self.ay,self.az, self.roll, self.pitch, self.yaw,-self.yaw_rate)

        
    def run(self):
        rospy.loginfo("[CF] Ready to receive commands.")
        rospy.spin()

if __name__ == '__main__':
    URI = rospy.get_param("~uri", 'radio://0/80/2M')
    DRONE_ID = rospy.get_param("~drone_id", 'karteek_drone')

    cflib.crtp.init_drivers()
    driver = Driver(uri=URI, drone_id=DRONE_ID)

    try:
        with SyncCrazyflie(URI, cf=driver.cf.cf) as scf:
            driver.cf.scf = scf
            driver.cf.setup_after_connection()
            driver.cf.warmup()
            driver.run()
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        rospy.loginfo("Shutting down driver node.")