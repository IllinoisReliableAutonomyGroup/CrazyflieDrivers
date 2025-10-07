# ==================================================================================
#
#       File: trajectory.py
#       Description: This file contains the template for the trajectory planner.
#                    Students will implement the logic to generate a dynamically
#                    feasible, time-optimal trajectory through a series of
#                    waypoints (gates). The planner should respect the drone's
#                    physical constraints on velocity and acceleration.
#
# ==================================================================================

import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Planner:
    """
    This class is a template for a trajectory planner.
    Its primary job is to solve an optimization problem to find the best path
    through a set of waypoints, respecting the drone's physical limits. 
    You can choose minsnap,cubic spline, time optimal etc.
    """

    def __init__(self, mass=0.032, max_velocity=1, max_acceleration=7):
        """
        Initializes the trajectory planner with the drone's physical properties
        and constraints.

        Args:
            mass (float): The mass of the drone in kilograms.
            max_velocity (float): The maximum allowable velocity in m/s.
            max_acceleration (float): The maximum allowable acceleration in m/s^2.
        """


        self.mass = mass
        self.g = 9.81
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.horizon = 1 # Use 1 for PID, increase for MPC if needed


        # These attributes will be populated by the `plan` method
        self.poses = None
        self.velocities = None
        self.accelerations = None
        self.yaws = None
        self.cumulative_times = None


    def plan(self, waypoints):
         """
        Plans a time-optimal, minimum-snap or other dynamically feasible trajectory
        through the given waypoints.

        This is the core function students will implement. It involves setting up
        and solving a complex optimization problem, likely using CasADi.

        Args:
            waypoints (list of lists): A list where each inner list represents a
                waypoint with 7 elements: [x, y, z, qx, qy, qz, qw].
                - x, y, z: position of the waypoint (gate)
                - qx, qy, qz, qw: orientation of the waypoint as a quaternion.

        Returns:
            bool: True if the planning was successful and the trajectory attributes
                  (e.g., self.poses) have been populated. False otherwise.
        """

        waypoints = np.array(waypoints, dtype=float)
        self.cumulative_times = None #time evaluated to complete the trajectory

        #TODO: Implement the trajectory planning logic here.
        # --- STUDENT TO-DO: IMPLEMENT TRAJECTORY OPTIMIZATION LOGIC ---
        # 1. Define your optimization variables (e.g., polynomial coefficients, time segments).
        # 2. Set up constraints:
        #    - Waypoint constraints (trajectory must pass through gates).
        #    - Continuity constraints (position, velocity, acceleration must be smooth).
        #    - Dynamic feasibility constraints (respect self.max_velocity, self.max_acceleration).
        # 3. Define the cost function (e.g., minimize total time, minimize snap).
        # 4. Populate the class attributes with the results:
        #    - self.poses: A list or array of planned (x, y, z) positions over time.
        #    - self.velocities: A list or array of planned (vx, vy, vz) velocities.
        #    - self.accelerations: A list or array of planned (ax, ay, az) accelerations.
        #    - self.yaws: A list or array of planned yaw angles.
        #    - self.cumulative_times: A list or array of the time at each point in the trajectory.
        
        # Example check after solving:
        if cumulative_times is None or self.poses is None or self.velocities is None or self.accelerations is None or self.yaws is None :
            return False # Planning failed
        else:
            return True # Planning successful


    def evaluate(self, t):
        """
        Evaluates the planned trajectory at a specific time `t` to get the
        reference states for the controller.

        This function samples the trajectory to generate the horizon needed by the MPC.

        Args:
            t (float): The current time at which to evaluate the trajectory.

        Returns:
            list of np.ndarray: A list of future reference states. Each element is
                a NumPy array in the format [x, y, z, vx, vy, vz, yaw]. The list
                represents the prediction horizon for the controller.
        """
        # --- STUDENT TO-DO: IMPLEMENT TRAJECTORY SAMPLING LOGIC ---
        # 1. Find the correct segment of your planned trajectory corresponding to time `t`.
        # 2. Sample `self.horizon` points along the trajectory starting from `t`.
        # 3. For each sample, retrieve the position, velocity, and yaw.
        # 4. Format each sample into the required NumPy array state format.
        
        reference_states = []
        
        if self.cumulative_times is not None:
            # This is a simplified example. A real implementation would interpolate
            # between points based on the time `t`.
            start_index = np.searchsorted(self.cumulative_times, t, side='right')
            
            for i in range(self.horizon):
                index = start_index + i
                if index < len(self.poses):
                    reference_states.append(np.array([
                        self.poses[index][0], self.poses[index][1], self.poses[index][2],
                        self.velocities[index][0], self.velocities[index][1], self.velocities[index][2],
                        self.yaws[index]
                    ]))
                
        return reference_states
  