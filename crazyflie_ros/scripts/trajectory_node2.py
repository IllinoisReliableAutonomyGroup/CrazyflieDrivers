#!/usr/bin/env python

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from crazyflie_ros.time_planner import TimeOptimalPlanner

# ROS Messages and Services
from geometry_msgs.msg import PoseArray, Transform, Twist, Vector3
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from std_srvs.srv import Trigger, TriggerResponse
from crazyflie_ros.srv import PlanTrajectory, PlanTrajectoryResponse

class TrajectoryNode:

    def __init__(self):
        rospy.init_node('trajectory_node')

        # Get parameters from the parameter server
        max_vel = rospy.get_param('~max_velocity', 1.0)
        max_acc = rospy.get_param('~max_acceleration', 15)
        self.publish_rate = rospy.get_param('~publish_rate', 100)

        # Initialize the planner
        self.planner = TimeOptimalPlanner(max_velocity=0.8, max_acceleration=8)
        
        # Publisher for trajectory setpoints using a standard message type
        self.setpoint_publisher = rospy.Publisher('/cf/setpoint', MultiDOFJointTrajectoryPoint, queue_size=10)

        # Services
        self.plan_service = rospy.Service('plan_trajectory', PlanTrajectory, self.plan_trajectory_callback)
        self.start_pub_service = rospy.Service('start_publishing', Trigger, self.start_publishing_callback)

        # Node state
        self.trajectory_planned = False
        self.publishing = False
        self.timer = None
        self.start_time = None
        
        rospy.loginfo("Trajectory Node is ready. Max Vel: {} m/s, Max Acc: {} m/s^2.".format(max_vel, max_acc))
        rospy.spin()

    def plan_trajectory_callback(self, req):
        """Service callback to plan a trajectory from waypoints."""
        if self.publishing:
            self.publishing = False
            if self.timer is not None:
                self.timer.shutdown()
            rospy.loginfo("Stopped publishing to plan a new trajectory.")

        num_waypoints = len(req.waypoints.poses)
        rospy.loginfo("Received planning request with {} waypoints.".format(num_waypoints))

        if num_waypoints < 2:
            return PlanTrajectoryResponse(success=False, message="Not enough waypoints (minimum 2).")

        # Convert PoseArray to numpy array
        waypoints_np = np.zeros((num_waypoints, 7))
        for i, pose in enumerate(req.waypoints.poses):
            waypoints_np[i, 0] = pose.position.x
            waypoints_np[i, 1] = pose.position.y
            waypoints_np[i, 2] = pose.position.z
            waypoints_np[i, 3] = pose.orientation.x
            waypoints_np[i, 4] = pose.orientation.y
            waypoints_np[i, 5] = pose.orientation.z
            waypoints_np[i, 6] = pose.orientation.w
        
        rospy.loginfo("Planning time-optimal trajectory...")
        success = self.planner.plan(waypoints=waypoints_np)

        if success:
            total_time = self.planner.cumulative_times[-1]
            msg = "Trajectory planned successfully! Total time: {:.2f}s".format(total_time)
            self.trajectory_planned = True
            rospy.loginfo(msg)
            return PlanTrajectoryResponse(success=True, message=msg)
        else:
            msg = "Failed to find a feasible trajectory."
            self.trajectory_planned = False
            rospy.logerr(msg)
            return PlanTrajectoryResponse(success=False, message=msg)

    def start_publishing_callback(self, req):
        """Service callback to start publishing the trajectory."""
        if not self.trajectory_planned:
            msg = "No trajectory planned. Call /plan_trajectory first."
            rospy.logwarn(msg)
            return TriggerResponse(success=False, message=msg)
        
        if self.publishing:
            msg = "Trajectory is already being published."
            rospy.logwarn(msg)
            return TriggerResponse(success=False, message=msg)

        self.publishing = True
        self.start_time = rospy.Time.now()
        
        self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_timer_callback)
        
        msg = "Started publishing trajectory setpoints."
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    def publish_timer_callback(self, event):
        """Timer callback to publish a setpoint."""
        if not self.publishing:
            return

        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        total_time = self.planner.cumulative_times[-1]

        if elapsed_time > total_time:
            self.publishing = False
            self.timer.shutdown()
            self.timer = None
            rospy.loginfo("Trajectory finished. Stopped publishing.")
            self.publish_setpoint(total_time) # Publish final point
            return

        self.publish_setpoint(elapsed_time)

    def publish_setpoint(self, eval_time):
        """Evaluates trajectory and publishes a MultiDOFJointTrajectoryPoint."""
        state = self.planner.evaluate(eval_time)
        if state is None:
            return

        msg = MultiDOFJointTrajectoryPoint()

        # Create a transform for position and orientation
        transform = Transform()
        transform.translation.x = state['pos'][0]
        transform.translation.y = state['pos'][1]
        transform.translation.z = state['pos'][2]
        
        # Calculate quaternion from yaw
        quat = R.from_euler('z', state['yaw']).as_quat()
        transform.rotation.x = quat[0]
        transform.rotation.y = quat[1]
        transform.rotation.z = quat[2]
        transform.rotation.w = quat[3]

        # Create a twist for linear and angular velocities
        velocity = Twist()
        velocity.linear.x = state['vel'][0]
        velocity.linear.y = state['vel'][1]
        velocity.linear.z = state['vel'][2]
        velocity.angular.z = state['yaw_rate'] # Yaw rate

        # Create a twist for linear and angular accelerations
        acceleration = Twist()
        acceleration.linear.x = state['acc'][0]
        acceleration.linear.y = state['acc'][1]
        acceleration.linear.z = state['acc'][2]
        
        msg.transforms.append(transform)
        msg.velocities.append(velocity)
        msg.accelerations.append(acceleration)
        
        self.setpoint_publisher.publish(msg)

if __name__ == '__main__':
    try:
        TrajectoryNode()
    except rospy.ROSInterruptException:
        pass
