#!/usr/bin/env python

# ==================================================================================
#
#       File: trajectory_node.py
#       Author: Your Name/Lab Name
#       Date: October 7, 2025
#       Description: This ROS node is responsible for planning and publishing
#                    a trajectory for the drone. It subscribes to gate poses,
#                    plans a path using the Planner class, and then waits for a
#                    service call to begin publishing the trajectory to the
#                    controller node.
#
# ==================================================================================

import rospy
from geometry_msgs.msg import PoseStamped, Transform, Vector3, Quaternion
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_srvs.srv import Trigger, TriggerResponse
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from crazyflie_ros.trajectory import Planner

class TrajectoryNode:
    def __init__(self, drone_id='cf'):
        rospy.init_node('trajectory_node')

        self.state = "INITIALIZING"
        self.initial_pose_msg = None
        self.gate_poses = {}
        self.num_gates = 4 # Number of gates to wait for

        self.trajectory_planned = False
        self.start_time = None

        # --- Parameters ---
        self.pub_rate = rospy.get_param('~publish_rate', 200)
        self.drone_id = drone_id

        # Instantiate your planner
        self.planner = Planner(max_velocity=max_vel, max_acceleration=max_accel)

        # --- Publisher ---
        self.trajectory_pub = rospy.Publisher(f'/{self.drone_id}/setpoint', MultiDOFJointTrajectory, queue_size=10)

        # --- Subscribers ---
        # Waits for the first message to consider the drone's position initialized
        self.drone_sub = rospy.Subscriber(f'/{self.drone_id}/pose', PoseStamped, self.drone_pose_callback, queue_size=1)

        self.gate_subs = []
        for i in range(self.num_gates):
            callback = lambda msg, gate_id=i: self.gate_pose_callback(msg, gate_id)
            sub = rospy.Subscriber(f'/gate_{i+1}/pose', PoseStamped, callback, queue_size=1)
            self.gate_subs.append(sub)
        
        rospy.loginfo("Trajectory node initialized. Waiting for drone and gate poses...")

    def drone_pose_callback(self, msg):
        # This callback runs only once to get the initial position
        if self.initial_pose_msg is None:
            self.initial_pose_msg = msg
            rospy.loginfo(f"Received initial drone pose at [{msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}]")
            self.drone_sub.unregister() # Unsubscribe after receiving the first message

    def gate_pose_callback(self, msg, gate_id):
        if gate_id not in self.gate_poses:
            self.gate_poses[gate_id] = msg.pose
            rospy.loginfo(f"Received pose for Gate {gate_id + 1}. Total gates found: {len(self.gate_poses)}/{self.num_gates}")

    def plan_trajectory(self):
        """
        Gathers waypoints and calls the planner. This is run in a separate thread
        to avoid blocking the ROS node while the optimization is running.
        """
        self.state = "PLANNING"
        rospy.loginfo("State -> PLANNING. All poses received, planning trajectory...")

        # Assemble waypoints in order: initial pose -> gates -> landing pose
        p = self.initial_pose_msg.pose
        waypoints = [[p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]]

        for i in sorted(self.gate_poses.keys()):
            p = self.gate_poses[i]
            waypoints.append([p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        
        # Add a final landing waypoint behind the last gate at a safe height
        last_gate_pose = waypoints[-1]
        landing_pose = [last_gate_pose[0] - 0.5, last_gate_pose[1], 0.1] + last_gate_pose[3:] # Land 0.5m behind last gate
        waypoints.append(landing_pose)

        # --- Call the Planner ---
        success = self.planner.plan(waypoints=waypoints)

        if success:
            self.trajectory_planned = True
            self.state = "READY"
            # Advertise the service that will trigger the trajectory execution
            self.ready_service = rospy.Service('~start_trajectory', Trigger, self.handle_start_trajectory)
            rospy.loginfo(f"Trajectory planning successful! Total time: {self.planner.cumulative_times[-1]:.2f}s")
            rospy.loginfo("State -> READY. Call the '~start_trajectory' service to begin.")
        else:
            rospy.logerr("Trajectory planning FAILED! The node will shut down.")
            self.state = "ERROR"
            rospy.signal_shutdown("Trajectory Planning Failed")

    def handle_start_trajectory(self, req):
        """
        Service callback to start the trajectory execution.
        """
        if self.state == "READY":
            rospy.loginfo("State -> EXECUTING. Start service called.")
            self.start_time = rospy.Time.now()
            self.state = "EXECUTING"
            return TriggerResponse(success=True, message="Trajectory started.")
        else:
            return TriggerResponse(success=False, message=f"Cannot start trajectory, current state is {self.state}")

    def run(self):
        """
        Main execution loop. Handles state transitions and trajectory publishing.
        """
        rate = rospy.Rate(self.pub_rate)

        while not rospy.is_shutdown():
            # State 1: INITIALIZING - Wait for all poses
            if self.state == "INITIALIZING":
                if self.initial_pose_msg is not None and len(self.gate_poses) == self.num_gates:
                    # Start planning in a non-blocking thread
                    planning_thread = threading.Thread(target=self.plan_trajectory)
                    planning_thread.start()

            # State 2: READY - Trajectory is planned, waiting for start signal
            elif self.state == "READY":
                # Publish a hover setpoint at the starting position
                hover_state = self.planner.evaluate(0)[0] # Get the very first point
                traj_msg = self.create_trajectory_msg_from_states([hover_state])
                self.trajectory_pub.publish(traj_msg)

            # State 3: EXECUTING - Publishing the moving trajectory
            elif self.state == "EXECUTING":
                elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
                
                if elapsed_time > self.planner.cumulative_times[-1]:
                    self.state = "FINISHED"
                    rospy.loginfo("State -> FINISHED. Trajectory complete.")
                    continue

                # Get the reference states for the controller's horizon
                reference_states = self.planner.evaluate(elapsed_time)
                
                if reference_states:
                    traj_msg = self.create_trajectory_msg_from_states(reference_states)
                    self.trajectory_pub.publish(traj_msg)

            # State 4: FINISHED - Trajectory is over, hover at the end
            elif self.state == "FINISHED":
                # Publish a hover setpoint at the final position
                final_state = self.planner.evaluate(self.planner.cumulative_times[-1])[0]
                traj_msg = self.create_trajectory_msg_from_states([final_state])
                self.trajectory_pub.publish(traj_msg)

            rate.sleep()

    def create_trajectory_msg_from_states(self, states):
        """
        Helper function to convert a list of numerical states into a
        MultiDOFJointTrajectory ROS message.
        """
        traj_msg = MultiDOFJointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()

        for state in states:
            # state is [x, y, z, vx, vy, vz, yaw]
            point = MultiDOFJointTrajectoryPoint()
            
            # Convert yaw to quaternion
            quat = R.from_euler('z', state[6]).as_quat()

            # Populate message fields
            transform = Transform(
                translation=Vector3(x=state[0], y=state[1], z=state[2]),
                rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            )
            velocities = Vector3(x=state[3], y=state[4], z=state[5])
            accelerations = Vector3(x=0, y=0, z=0) # Accelerations can be added if your planner provides them

            point.transforms.append(transform)
            point.velocities.append(velocities)
            point.accelerations.append(accelerations)
            
            traj_msg.points.append(point)

        return traj_msg


if __name__ == '__main__':
    try:
        node = TrajectoryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
