# ==================================================================================
#
#       File: controllers.py
#       Description: This file contains the template for the drone's flight
#                    controller. You will implement the control logic
#                    within the `solve` method of the DroneController class.
#                    The controller is responsible for calculating the required
#                    accelerations and yaw rate to follow a given trajectory.
#
# ==================================================================================



import casadi as ca
import numpy as np
import time
import math


class DroneController:
    def __init__(self, dt=0.01, vmax=1, amax=7.0):
        """
        Initializes the controller. You can add any parameters
        or initializations that are needed here (e.g., PID gains, MPC parameters).

        NOTE : Build Your controller respecting the velocity and acceleration constraints of crazyflie.

        Args:
            dt (float): The time step (sampling time) of the controller, in seconds.
            vmax (float): Maximum crazyflies velocity in m/s.
            amax (float): Maximum crazyflies acceleration in m/s^2. 

        """

        self.dt = dt  #100hz
        self.vmax = vmax
        self.amax = amax

    def solve(self, current_state, reference_states):
        """
        Calculates the required control commands (accelerations and yaw rate)
        to follow the reference trajectory.

        This is the primary function that students need to implement.

        Args:
            current_state (np.ndarray): A 1D NumPy array containing the drone's
                current state, with the following format:
                [x, y, z, vx, vy, vz, yaw]
                - x, y, z: current position in the world frame (meters)
                - vx, vy, vz: current velocity in the world frame (m/s)
                - yaw: current yaw angle in radians

            reference_states (list of np.ndarray): A list where each element is a
                1D NumPy array representing a future reference state. The format
                of each array is the same as `current_state`:
                [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref, yaw_ref]
                - For a simple PID controller, you might only use the first
                  element of this list (reference_states[0]).
                - For an MPC controller, you can use the entire list as your
                  prediction horizon. If you want to increase the horizon, update your trajectory function to publish more points.

        Returns:
            tuple: A tuple containing the calculated control commands:
                (ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd)
                - ax_cmd, ay_cmd, az_cmd: desired linear accelerations in the
                  world frame (m/s^2).
                - yaw_rate_cmd: desired yaw rate (rad/s).
        """

        #TODO: Implement your control logic here.
        # You can use libraries like CasADi for optimization if needed.
        # Make sure to respect the velocity and acceleration constraints of crazyflie.


        ax,ay,az, yaw_rate = 0,0,0
        return ax,ay,az, yaw_rate



