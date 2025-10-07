import time
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
import numpy as np
import cflib.crtp
from scipy.spatial.transform import Rotation as R
from cflib.crazyflie.extpos import Extpos

# Init drivers globally
cflib.crtp.init_drivers()



def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (in radians) to quaternion [qx, qy, qz, qw].
    """
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()  # [qx, qy, qz, qw]


class CrazyflieController():

    def __init__(self, uri, logger=None, CF_MASS = 0.027 ,HOVER_THRUST=42000,K_THRUST=0.000081):
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = None
        self.log_config = None
        self.log_variables = [
            ('stabilizer.roll', 'float'),
            ('stabilizer.pitch', 'float'),
            ('stabilizer.yaw', 'float'),
            ('pm.vbat', 'float')
        ]
        self.latest_data = {}

        self.HOVER_THRUST_CF = HOVER_THRUST
        self.K_THRUST = K_THRUST
        self.CF_MASS = CF_MASS
        self.DRONE_MASS_KG = CF_MASS  # kg
        self.GRAVITY = 9.81  # m/s²
        self.MAX_ANGLE_DEG = 30  # degrees
        self.MAX_YAW_RATE_DEG = 200  # degrees/sec
        self.logger = logger
    
    # Add this new method to drone.py inside the CrazyflieController class
    def setup_after_connection(self):
        """
        Sets up components that require an active connection (scf).
        """
        if self.scf:
            # Now that scf exists, we can create the Extpos object
            self.extpos = Extpos(self.scf.cf)
            self.setup_logging()
            print("[INFO] Logging and Extpos have been set up.")
        else:
            rospy.logerr("[Controller] Cannot run setup: SyncCrazyflie object not available.")


    def connect(self, pose = None):
        print(f"[INFO] Connecting to {self.uri}")
        #self.scf = SyncCrazyflie(self.uri, cf=self.cf)
        
        self.scf.open_link()
        
        self.setup_logging()
        print("[INFO] Logging started")
        self.warmup()



        """
        for group in self.cf.param.toc.toc:
            for name in self.cf.param.toc.toc[group]:
                print(f"{group}.{name}")
        """


    def set_initial_yaw_and_reset(self, yaw_rad):
        """
        Set the initial yaw angle (in radians) and reset the Kalman filter to apply it.
        
        Parameters:
            yaw_rad (float): Initial yaw in radians (from Vicon or external source)
        """
        yaw_deg = np.degrees(yaw_rad)
        print(f"[INFO] Setting initial yaw to {yaw_deg:.2f} degrees")
        
        self.cf.param.set_value("kalman.initialYaw", str(yaw_deg))
        time.sleep(0.05)
        
        self.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(0.1)
        
        print("[INFO] Kalman filter reset with new yaw")



    def warmup(self, duration=1.0):
        print("[INFO] Warming up (sending zero setpoints)...")
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.05)
        print("[INFO] Warmup complete")

    def setup_logging(self):
        from cflib.crazyflie.log import LogConfig
        self.log_config = LogConfig(name='StateLog', period_in_ms=100)
        for var, var_type in self.log_variables:
            self.log_config.add_variable(var, var_type)

        self.cf.log.add_config(self.log_config)
        self.log_config.data_received_cb.add_callback(self._log_data_callback)
        self.log_config.start()

    def send_extpose(self, x, y, z,roll, pitch, yaw):
        """
        Send external pose (position + orientation) to Crazyflie Kalman filter.

        Parameters:
            x, y, z       : position in meters
            qx, qy, qz, qw: orientation quaternion (ENU frame)
        """

        qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)


        if self.extpos:
            self.extpos.send_extpose(x, y, z, 0,0,0,1)
            #print("sent extpos")
          



    def _log_data_callback(self, timestamp, data, logconf):
        self.latest_data = data

    def send_setpoint(self, roll, pitch, yaw_rate, thrust):
        self.cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)

    def get_state(self):
        return self.latest_data

    def stop(self):
        print("[INFO] Sending stop setpoint")
        
        # Send multiple zero setpoints to ensure motors shut down
        for _ in range(10):
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.05)

        if self.log_config:
            self.log_config.stop()

        # Give some delay to let firmware settle
        time.sleep(0.5)

        self.scf.close_link()
        print("[INFO] Disconnected")




    def send_acceleration_command(self, ax, ay, az, roll_world, pitch_world, yaw_world, yaw_rate_world):
        """
        Converts desired world-frame acceleration and yaw rate to Crazyflie commands.

        This function implements the correct physical transformations.

        Parameters:
            ax, ay, az (float): Desired acceleration in world frame (m/s²).
            roll_world, pitch_world, yaw_world (float): Current drone orientation from Vicon (radians).
            yaw_rate_world (float): Desired yaw rate in world frame (radians/sec).
        """

        # --- 1. Yaw Rate Transformation ---
        # Convert world yaw rate to body yaw rate using current attitude
        # This is necessary because a spin around the world Z-axis is not the same
        # as a spin around the drone's tilted Z-axis.
        body_yaw_rate = yaw_rate_world * (np.cos(roll_world) * np.cos(pitch_world))
        
        # --- 2. Calculate Total Required Force in World Frame ---
        # The drone must counteract gravity in addition to the desired acceleration
        force_world = np.array([
            self.DRONE_MASS_KG * ax,
            self.DRONE_MASS_KG * ay,
            self.DRONE_MASS_KG * (az + self.GRAVITY)
        ])

        # --- 3. Calculate Total Thrust Command ---
        # The total thrust is the magnitude of the required force vector
        total_force = np.linalg.norm(force_world)
        
        # Map the physical force (Newtons) to the Crazyflie's thrust unit (0-65535)
        # The scaling factor (total_force / self.GRAVITY) is a good approximation
        thrust_cf = int(self.HOVER_THRUST_CF * (total_force / (self.DRONE_MASS_KG * self.GRAVITY)))

        # --- 4. Calculate Target Roll and Pitch Commands ---
        # To find the required roll and pitch, we project the world force vector
        # onto the drone's current yaw-aligned horizontal plane.
        cy, sy = np.cos(yaw_world), np.sin(yaw_world)
        
        # F_horiz_yaw_aligned is the component of force in the drone's forward direction
        F_horiz_yaw_aligned = force_world[0] * cy + force_world[1] * sy
        
        # F_side_yaw_aligned is the component of force in the drone's side direction
        F_side_yaw_aligned = -force_world[0] * sy + force_world[1] * cy
        
        # The vertical component of force remains the same
        F_vertical = force_world[2]

        # atan2 is used to find the angle of the force vector, which corresponds to the
        # required tilt of the drone.
        # Note the negative sign for pitch, as per your observation.
        pitch = -np.arctan2(F_horiz_yaw_aligned, F_vertical)
        roll = np.arctan2(F_side_yaw_aligned, F_vertical)

        # --- 5. Clip and Send Commands ---
        max_angle_rad = np.radians(self.MAX_ANGLE_DEG)
        roll = np.clip(roll, -max_angle_rad, max_angle_rad)
        pitch = np.clip(pitch, -max_angle_rad, max_angle_rad)
        
        max_yaw_rate_rad = np.radians(self.MAX_YAW_RATE_DEG)
        body_yaw_rate = np.clip(body_yaw_rate, -max_yaw_rate_rad, max_yaw_rate_rad)
        
        thrust_cf = int(np.clip(thrust_cf, 10000, 60000))
        #print(roll,pitch,thrust_cf)
        self.send_setpoint(
            roll=np.degrees(roll),
            pitch=np.degrees(pitch),
            yaw_rate=np.degrees(body_yaw_rate),
            thrust=thrust_cf
        )


    def send_acceleration_command_rpy(self, ax, ay, az, roll_world, pitch_world, yaw_world, yaw_rate):
        """
        Convert world-frame desired acceleration and orientation to Crazyflie commands.

        Parameters:
            ax, ay, az       : desired acceleration in world frame (m/s²)
            roll_world       : drone roll from Vicon (radians)
            pitch_world      : drone pitch from Vicon (radians)
            yaw_world        : drone yaw from Vicon (radians)
            yaw_rate         : desired yaw rate (radians/sec)
        """

        
        # --- Step 2: Full 3D rotation matrix R_wb = Rz(yaw) @ Ry(pitch) @ Rx(roll) ---
        #HOVER_THRUST_CF=42000
        max_angle_deg=30
        max_yaw_rate_deg=200
        a_world = np.array([ax, ay, az])
        cy, sy = np.cos(yaw_world), np.sin(yaw_world)
        cp, sp = np.cos(pitch_world), np.sin(pitch_world)
        cr, sr = np.cos(roll_world), np.sin(roll_world)

        R_wb = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,    cp*sr,            cp*cr]
        ])
        R_bw = R_wb.T

        a_body = R_bw @ a_world
        ax_b, ay_b, az_b = a_body
        ax_filt, ay_filt, az_filt = ax_b, ay_b, az_b

        g = 9.81
        az_total = az_filt + g
        pitch = np.arctan2(ax_filt, az_total)
        roll  = -np.arctan2(ay_filt, az_total)

        max_angle_rad = np.radians(max_angle_deg)
        pitch = np.clip(pitch, -max_angle_rad, max_angle_rad)
        roll  = np.clip(roll,  -max_angle_rad, max_angle_rad)

        norm_acc = np.sqrt(ax_filt**2 + ay_filt**2 + az_total**2)
        thrust_cf = int(self.HOVER_THRUST_CF * (norm_acc / g))
        thrust_cf = int(np.clip(thrust_cf, 10000, 60000))

        max_yaw_rate_rad = np.radians(max_yaw_rate_deg)
        yaw_rate = np.clip(yaw_rate, -max_yaw_rate_rad, max_yaw_rate_rad)


        self.send_setpoint(
            roll=np.degrees(roll),
            pitch=np.degrees(pitch),
            yaw_rate=np.degrees(yaw_rate),
            thrust=thrust_cf
        )
        print(ax_filt, ay_filt, az_total) # np.degrees(roll), np.degrees(pitch), thrust_cf)
        return np.array([ax_filt, ay_filt, az_total]), np.degrees(roll), np.degrees(pitch)


    def activete_high_level_command(self):
        self.cf.param.set_value('commander.enHighLevel', '1') # enable high-level commander
        self.cf.param.set_value('stabilizer.controller', '1') 
        print("[INFO] Enabled high-level commander mode")
        


    def send_velocity_command(self, vx, vy, vz, yaw_rate_deg):
        """
        Send velocity commands to the Crazyflie using the high-level commander.
        Velocity in m/s, yaw_rate in deg/s.
        """
        # Send velocity setpoint
        self.cf.high_level_commander.set_velocity(
            vx,  # velocity x [m/s]
            vy,  # velocity y [m/s]
            vz,  # velocity z [m/s]
            yaw_rate_deg  # yaw rate [deg/s]
        )

    def get_attitude(self):
        """Return current roll, pitch, yaw from internal Crazyflie estimator (in radians)"""
        data = self.get_state()
        roll_deg = data.get('stabilizer.roll')
        pitch_deg = data.get('stabilizer.pitch')
        yaw_deg = data.get('stabilizer.yaw')
        return (
            np.radians(roll_deg),
            np.radians(pitch_deg),
            np.radians(yaw_deg)
        )


