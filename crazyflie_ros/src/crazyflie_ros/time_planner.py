import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TimeOptimalPlanner:
    """
    Plans a time-optimal, minimum-snap trajectory through a series of 7-DOF gates (position + quaternion) using CasADi.
    """

    def __init__(self, mass=0.032, max_velocity=0.8, max_acceleration=7):
        self.mass = mass
        self.g = 9.81
        self.polynom_coeffs_x = None
        self.polynom_coeffs_y = None
        self.polynom_coeffs_z = None
        self.polynom_coeffs_yaw = None
        self.segment_times = None
        self.num_segments = 0
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.cumulative_times = None

    def _create_poly_matrices(self, order, t):
        p = np.array([t**i for i in range(order + 1)])
        v = np.array([i * t**(i - 1) if i > 0 else 0 for i in range(order + 1)])
        a = np.array([i * (i - 1) * t**(i - 2) if i > 1 else 0 for i in range(order + 1)])
        j = np.array([i * (i - 1) * (i - 2) * t**(i - 3) if i > 2 else 0 for i in range(order + 1)])
        s = np.array([i * (i - 1) * (i - 2) * (i - 3) * t**(i - 4) if i > 3 else 0 for i in range(order + 1)])
        return p, v, a, j, s

    def _get_poly_expressions(self, order, t_sym, coeffs_sym):
        """Creates CasADi symbolic expressions for position, velocity, etc."""
        p_expr = 0; v_expr = 0; a_expr = 0; j_expr = 0
        for i in range(order + 1):
            p_expr += coeffs_sym[i] * t_sym**i
            if i >= 1: v_expr += i * coeffs_sym[i] * t_sym**(i - 1)
            if i >= 2: a_expr += i * (i - 1) * coeffs_sym[i] * t_sym**(i - 2)
            if i >= 3: j_expr += i * (i - 1) * (i - 2) * coeffs_sym[i] * t_sym**(i - 3)
        return p_expr, v_expr, a_expr, j_expr

    def plan(self, waypoints, initial_guess_avg_velocity=1):
        waypoints = np.array(waypoints, dtype=float)
        self.num_segments = len(waypoints) - 1
        if self.num_segments < 1:
            return False

        distances = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
        initial_times = np.maximum(distances / initial_guess_avg_velocity, 0.1)

        order = 7
        num_coeffs = order + 1
        
        opti = ca.Opti()

        # --- Optimization Variables ---
        coeffs_x = opti.variable(self.num_segments, num_coeffs)
        coeffs_y = opti.variable(self.num_segments, num_coeffs)
        coeffs_z = opti.variable(self.num_segments, num_coeffs)
        T = opti.variable(self.num_segments) # Segment times

        # --- Cost Function: Minimize Total Time ---
        cost = ca.sum1(T)
        time_weight = 0.01  # Lower this weight
        snap_weight = 2.0 
        # Add a small regularization term for snap to ensure smoothness
        snap_cost_proxy = 0
        for i in range(self.num_segments):
            snap_cost_proxy += ca.sumsqr(coeffs_x[i, 4:])
            snap_cost_proxy += ca.sumsqr(coeffs_y[i, 4:])
            snap_cost_proxy += ca.sumsqr(coeffs_z[i, 4:])
        cost += snap_weight * snap_cost_proxy
                
        opti.minimize(cost)

        # --- Constraints ---
        for i in range(self.num_segments):
            c_x = coeffs_x[i, :]; c_y = coeffs_y[i, :]; c_z = coeffs_z[i, :]
            Ti = T[i]
            
            opti.subject_to(Ti >= 0.01) # Minimum segment time

            # Waypoint position constraints
            p_x_start, _, _, _ = self._get_poly_expressions(order, 0, c_x)
            p_x_end, _, _, _ = self._get_poly_expressions(order, Ti, c_x)
            p_y_start, _, _, _ = self._get_poly_expressions(order, 0, c_y)
            p_y_end, _, _, _ = self._get_poly_expressions(order, Ti, c_y)
            p_z_start, _, _, _ = self._get_poly_expressions(order, 0, c_z)
            p_z_end, _, _, _ = self._get_poly_expressions(order, Ti, c_z)
            opti.subject_to(p_x_start == waypoints[i, 0])
            opti.subject_to(p_x_end == waypoints[i+1, 0])
            opti.subject_to(p_y_start == waypoints[i, 1])
            opti.subject_to(p_y_end == waypoints[i+1, 1])
            opti.subject_to(p_z_start == waypoints[i, 2])
            opti.subject_to(p_z_end == waypoints[i+1, 2])

            # Start and End constraints (zero velocity, acceleration, jerk)
            if i == 0:
                _, v_x, a_x, j_x = self._get_poly_expressions(order, 0, c_x)
                _, v_y, a_y, j_y = self._get_poly_expressions(order, 0, c_y)
                _, v_z, a_z, j_z = self._get_poly_expressions(order, 0, c_z)
                opti.subject_to(ca.vertcat(v_x, v_y, v_z) == 0)
                opti.subject_to(ca.vertcat(a_x, a_y, a_z) == 0)
                opti.subject_to(ca.vertcat(j_x, j_y, j_z) == 0)

            if i == self.num_segments - 1:
                _, v_x, a_x, j_x = self._get_poly_expressions(order, Ti, c_x)
                _, v_y, a_y, j_y = self._get_poly_expressions(order, Ti, c_y)
                _, v_z, a_z, j_z = self._get_poly_expressions(order, Ti, c_z)
                opti.subject_to(ca.vertcat(v_x, v_y, v_z) == 0)
                opti.subject_to(ca.vertcat(a_x, a_y, a_z) == 0)
                opti.subject_to(ca.vertcat(j_x, j_y, j_z) == 0)
                
            # Continuity constraints between segments
            if i < self.num_segments - 1:
                c_x_next = coeffs_x[i+1, :]; c_y_next = coeffs_y[i+1, :]; c_z_next = coeffs_z[i+1, :]
                _, v_x_curr, a_x_curr, j_x_curr = self._get_poly_expressions(order, Ti, c_x)
                _, v_y_curr, a_y_curr, j_y_curr = self._get_poly_expressions(order, Ti, c_y)
                _, v_z_curr, a_z_curr, j_z_curr = self._get_poly_expressions(order, Ti, c_z)
                _, v_x_next, a_x_next, j_x_next = self._get_poly_expressions(order, 0, c_x_next)
                _, v_y_next, a_y_next, j_y_next = self._get_poly_expressions(order, 0, c_y_next)
                _, v_z_next, a_z_next, j_z_next = self._get_poly_expressions(order, 0, c_z_next)
                opti.subject_to(v_x_curr == v_x_next); opti.subject_to(a_x_curr == a_x_next); opti.subject_to(j_x_curr == j_x_next)
                opti.subject_to(v_y_curr == v_y_next); opti.subject_to(a_y_curr == a_y_next); opti.subject_to(j_y_curr == j_y_next)
                opti.subject_to(v_z_curr == v_z_next); opti.subject_to(a_z_curr == a_z_next); opti.subject_to(j_z_curr == j_z_next)

            # Dynamic constraints (Velocity and Acceleration)
            num_samples = 5
            for k in range(num_samples + 1):
                t_sample = k/num_samples * Ti
                _, vx, ax, _ = self._get_poly_expressions(order, t_sample, c_x)
                _, vy, ay, _ = self._get_poly_expressions(order, t_sample, c_y)
                _, vz, az, _ = self._get_poly_expressions(order, t_sample, c_z)
                opti.subject_to(vx**2 + vy**2 + vz**2 <= self.max_velocity**2)
                opti.subject_to(ax**2 + ay**2 + az**2 <= self.max_acceleration**2)

        # --- Solve the optimization problem ---
        opti.set_initial(T, initial_times) # Provide initial guess for segment times
        solver_opts = {'ipopt.print_level': 5, 'print_time': True}
        opti.solver('ipopt', solver_opts)
        
        try:
            sol = opti.solve()
            self.polynom_coeffs_x = sol.value(coeffs_x)
            self.polynom_coeffs_y = sol.value(coeffs_y)
            self.polynom_coeffs_z = sol.value(coeffs_z)
            self.segment_times = sol.value(T)
            self.cumulative_times = np.insert(np.cumsum(self.segment_times), 0, 0)
            
            self._plan_yaw_trajectory_from_quaternions(waypoints, self.segment_times)

        except RuntimeError as e:
            print(f"[Planner] Error: Optimization failed. {e}")
            print("Solver might have failed to find a feasible solution.")
            return False

        self.check_constraints()
        print(f"[Planner] Time-optimal trajectory planning successful. Total time: {self.cumulative_times[-1]:.2f}s")
        return True

    def _plan_yaw_trajectory_from_quaternions(self, waypoints, segment_times):
        """Plans a smooth yaw trajectory by extracting yaw from quaternions."""
        self.polynom_coeffs_yaw = []
        num_coeffs = 5 + 1
        
        yaw_points = []
        for i in range(len(waypoints)):
            quat = waypoints[i, 3:7] # qx, qy, qz, qw
            r = R.from_quat(quat)
            yaw, _, _ = r.as_euler('zyx', degrees=False)
            yaw_points.append(yaw)

        # Ensure continuous yaw by unwrapping
        yaw_points = np.unwrap(yaw_points)

        for i in range(self.num_segments):
            A = np.zeros((6, 6)); b = np.zeros(6)
            T = segment_times[i]
            A[0,:]=[0,0,0,0,0,1]; A[1,:]=[T**5,T**4,T**3,T**2,T,1]
            A[2,:]=[0,0,0,0,1,0]; A[3,:]=[5*T**4,4*T**3,3*T**2,2*T,1,0]
            A[4,:]=[0,0,0,2,0,0]; A[5,:]=[20*T**3,12*T**2,6*T,2,0,0]
            
            b[0] = yaw_points[i]
            b[1] = yaw_points[i+1]
            
            coeffs = np.linalg.solve(A, b)
            self.polynom_coeffs_yaw.append(np.flip(coeffs))

    def evaluate(self, t):
        if self.polynom_coeffs_x is None or t < 0 or t > self.cumulative_times[-1] + 1e-6:
            return None
        segment_idx = np.searchsorted(self.cumulative_times, t, side='right') - 1
        segment_idx = np.clip(segment_idx, 0, self.num_segments - 1)
        t_local = t - self.cumulative_times[segment_idx]
        
        coeffs_x=self.polynom_coeffs_x[segment_idx]; coeffs_y=self.polynom_coeffs_y[segment_idx]
        coeffs_z=self.polynom_coeffs_z[segment_idx]; coeffs_yaw=self.polynom_coeffs_yaw[segment_idx]
        
        p,v,a,_,_=self._create_poly_matrices(7,t_local)
        p_y,v_y,_,_,_=self._create_poly_matrices(5,t_local)
        
        pos=np.array([coeffs_x@p,coeffs_y@p,coeffs_z@p])
        vel=np.array([coeffs_x@v,coeffs_y@v,coeffs_z@v])
        acc=np.array([coeffs_x@a,coeffs_y@a,coeffs_z@a])
        yaw=coeffs_yaw@p_y
        yaw_rate=coeffs_yaw@v_y
        
        return {"pos":pos,"vel":vel,"acc":acc,"yaw":yaw,"yaw_rate":yaw_rate}
        
    def check_constraints(self):
        # ... (This function remains unchanged)
        if self.cumulative_times is None: return
        num_samples = int(self.cumulative_times[-1] * 100) + 2
        t_samples = np.linspace(0, self.cumulative_times[-1], num_samples)
        max_v_found=0; max_a_found=0
        for t in t_samples:
            state=self.evaluate(t)
            if state:
                v_mag=np.linalg.norm(state['vel']); a_mag=np.linalg.norm(state['acc'])
                if v_mag>max_v_found: max_v_found=v_mag
                if a_mag>max_a_found: max_a_found=a_mag
        print(f"[Planner] Constraint Check: Max Velocity = {max_v_found:.2f} m/s (Limit: {self.max_velocity:.2f} m/s)")
        print(f"[Planner] Constraint Check: Max Acceleration = {max_a_found:.2f} m/s^2 (Limit: {self.max_acceleration:.2f} m/s^2)")
        if max_v_found > self.max_velocity + 0.1 or max_a_found > self.max_acceleration + 0.1:
            print("\n[Planner] WARNING: Trajectory may violate dynamic constraints!")

def visualize_trajectory_quaternion(planner, waypoints, gate_radius=0.5):
    # ... (This function remains unchanged)
    if planner.polynom_coeffs_x is None:
        print("[Visualizer] Planner has not generated a trajectory yet.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    t_samples = np.linspace(0, planner.cumulative_times[-1], 500)
    positions = np.array([planner.evaluate(t)['pos'] for t in t_samples])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Planned Trajectory', color='b')

    waypoints = np.array(waypoints)
    ax.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2], color='green', s=100, label='Start')
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], color='red', s=100, label='End')

    for i in range(len(waypoints)):
        pos = waypoints[i, 0:3]
        quat = waypoints[i, 3:7] # qx, qy, qz, qw
        
        theta = np.linspace(0, 2 * np.pi, 25)
        circle_body = np.zeros((len(theta), 3))
        circle_body[:, 1] = gate_radius * np.cos(theta)
        circle_body[:, 2] = gate_radius * np.sin(theta)

        rot = R.from_quat(quat)
        circle_world = rot.apply(circle_body) + pos
        
        ax.plot(circle_world[:, 0], circle_world[:, 1], circle_world[:, 2], color='purple', alpha=0.7)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('Time-Optimal Trajectory Visualization (CasADi)')
    
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                           positions[:, 1].max()-positions[:, 1].min(), 
                           positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    plt.show()


if __name__ == '__main__':
    gates_quaternion = [
        [-0.385, 0.541, 0.94] + R.from_euler('zyx', [3.005, -0.012, -3.083]).as_quat().tolist(),
        [0.869, 0.544, 0.889] + R.from_euler('zyx', [0, 0, 0]).as_quat().tolist(),
        [1.962, -1.226, 0.819] + R.from_euler('zyx', [-1.630, 0.022, -0.021]).as_quat().tolist(),
        [0.272, -2.319, 0.840] + R.from_euler('zyx', [3.099, 0.001, 0.0007]).as_quat().tolist(),
        [-0.927, -1.011, 0.908] + R.from_euler('zyx', [-1.681, 0.008, -0.005]).as_quat().tolist(),
        [-0.385, 0.541, 0.142] + R.from_euler('zyx', [3.005, -0.012, -3.083]).as_quat().tolist(),
    ]
    
    # Initialize the planner with dynamic constraints
    planner = TimeOptimalPlanner(max_velocity=3.0, max_acceleration=15.0)
    
    success = planner.plan(waypoints=gates_quaternion, initial_guess_avg_velocity=1.5)
    
    if success:
        t_eval = planner.cumulative_times[-1] / 2.0
        state = planner.evaluate(t_eval)
        if state:
            print(f"\nState at t={t_eval:.2f}s:")
            print(f"  Position: {np.round(state['pos'], 2)}")
            print(f"  Velocity: {np.round(state['vel'], 2)}")
            print(f"  Acceleration: {np.round(state['acc'], 2)}")
            print(f"  Yaw: {np.rad2deg(state['yaw']):.2f} deg")

        visualize_trajectory_quaternion(planner, gates_quaternion)