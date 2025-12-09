#!/usr/bin/env python

# NOTE: This solution requires the 'plotly' and 'dash' libraries.
# See "Setup Instructions" below for installation.

import rospy
import threading
import time
import numpy as np

# --- Plotly & Dash Imports ---
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# ROS Message Imports
from nav_msgs.msg import Odometry 
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

# ====================================================================
# --- 1. GLOBAL DATA STORAGE (Accessed by ROS thread and Dash thread) ---
# ====================================================================

# We use global lists to store trajectory data. These are protected by a lock.
global_odom_x, global_odom_y, global_odom_z = [], [], []
global_setpoint_x, global_setpoint_y, global_setpoint_z = [], [], []
data_lock = threading.Lock()
max_data_points = 1000 # Limit data points for performance

# ====================================================================
# --- 2. ROS NODE CLASS (Runs in the main thread) ---
# ====================================================================

class TrajectoryDataCollector:
    """ROS Node that subscribes to topics and updates the global data lists."""
    def __init__(self, drone_id='cf'):
        rospy.init_node('live_3d_plotter', anonymous=True)
        
        # --- ROS Subscribers ---
        rospy.Subscriber('/' + drone_id + '/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/' + drone_id + '/setpoint', MultiDOFJointTrajectoryPoint, self.setpoint_callback, queue_size=1)
        
        rospy.loginfo("TrajectoryDataCollector Initialized. Waiting for data...")

    def odom_callback(self, msg):
        """Callback for /drone/odom (Actual Position)."""
        global global_odom_x, global_odom_y, global_odom_z
        pos = msg.pose.pose.position
        
        with data_lock:
            global_odom_x.append(pos.x)
            global_odom_y.append(pos.y)
            global_odom_z.append(pos.z)
            
            # Keep only the latest N points
            global_odom_x = global_odom_x[-max_data_points:]
            global_odom_y = global_odom_y[-max_data_points:]
            global_odom_z = global_odom_z[-max_data_points:]

    def setpoint_callback(self, msg):
        """Callback for /drone/setpoint (Commanded Position)."""
        global global_setpoint_x, global_setpoint_y, global_setpoint_z
        try:
            pos = msg.transforms[0].translation
            with data_lock:
                global_setpoint_x.append(pos.x)
                global_setpoint_y.append(pos.y)
                global_setpoint_z.append(pos.z)
                
                # Keep only the latest N points
                global_setpoint_x = global_setpoint_x[-max_data_points:]
                global_setpoint_y = global_setpoint_y[-max_data_points:]
                global_setpoint_z = global_setpoint_z[-max_data_points:]

        except IndexError:
            rospy.logwarn_once("Setpoint message is missing 'transforms' data.")

# ====================================================================
# --- 3. DASH APPLICATION (Runs in a separate thread) ---
# ====================================================================

def run_dash_app(host='0.0.0.0', port=8050):
    """Initializes and runs the Dash web server in its own thread."""
    
    # Dash app setup
    app = Dash(__name__)
    
    # Define the layout
    app.layout = html.Div(
        style={'backgroundColor': '#111111', 'height': '100vh'},
        children=[
            html.H1(
                children='Crazyflie Live Trajectory (Plotly/Dash)',
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF'
                }
            ),

            # The 3D Plotly Graph
            dcc.Graph(id='live-trajectory-graph'),

            # Interval component to trigger graph updates every 100ms
            dcc.Interval(
                id='interval-component',
                interval=100,  # in milliseconds
                n_intervals=0
            ),
        ]
    )

    # Define the callback to update the graph
    @app.callback(
        Output('live-trajectory-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph_live(n):
        """Function called by Dash to regenerate the Plotly figure."""
        
        # Access the global data lists safely
        with data_lock:
            # Copy data for plotting
            odom_x, odom_y, odom_z = list(global_odom_x), list(global_odom_y), list(global_odom_z)
            setpoint_x, setpoint_y, setpoint_z = list(global_setpoint_x), list(global_setpoint_y), list(global_setpoint_z)

        # --- Create Plotly Traces (Lines and Points) ---
        traces = []

        # 1. Actual Trajectory Line
        traces.append(go.Scatter3d(
            x=odom_x, y=odom_y, z=odom_z,
            mode='lines',
            name='Actual Trajectory',
            line=dict(color='blue', width=4)
        ))

        # 2. Actual Current Position Marker
        if odom_x:
            traces.append(go.Scatter3d(
                x=[odom_x[-1]], y=[odom_y[-1]], z=[odom_z[-1]],
                mode='markers',
                name='Actual Position',
                marker=dict(color='blue', size=8, symbol='circle')
            ))

        # 3. Setpoint Trajectory Line
        traces.append(go.Scatter3d(
            x=setpoint_x, y=setpoint_y, z=setpoint_z,
            mode='lines',
            name='Setpoint Trajectory',
            line=dict(color='red', width=2, dash='dash')
        ))

        # 4. Setpoint Current Position Marker
        if setpoint_x:
            traces.append(go.Scatter3d(
                x=[setpoint_x[-1]], y=[setpoint_y[-1]], z=[setpoint_z[-1]],
                mode='markers',
                name='Setpoint',
                marker=dict(color='red', size=10, symbol='x')
            ))

        # --- Define Layout and Camera ---
        # Calculate limits dynamically
        all_x = odom_x + setpoint_x
        all_y = odom_y + setpoint_y
        all_z = odom_z + setpoint_z

        if all_x:
            # Calculate mean position and set a fixed range around it (e.g., 3m cube)
            center_x, center_y, center_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        else:
            # Default center if no data yet
            center_x, center_y, center_z = 0.0, 0.0, 0.5
            
        plot_range = 3.0
        r = plot_range / 2.0
        
        # Configure the 3D scene (ensures 1:1:1 aspect ratio)
        scene = dict(
            xaxis=dict(range=[center_x - r, center_x + r], title='X Position (m)', backgroundcolor="#333333", gridcolor="#666666", zerolinecolor="#999999"),
            yaxis=dict(range=[center_y - r, center_y + r], title='Y Position (m)', backgroundcolor="#333333", gridcolor="#666666", zerolinecolor="#999999"),
            zaxis=dict(range=[center_z - r, center_z + r], title='Z Position (m)', backgroundcolor="#333333", gridcolor="#666666", zerolinecolor="#999999"),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )

        # Create the Figure object
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title='Live Drone Trajectory Plot (Actual vs. Setpoint)',
                showlegend=True,
                margin=dict(l=0, r=0, b=0, t=30),
                scene=scene,
                # Set a uirevision value to prevent the camera view from resetting 
                # every time the data updates.
                uirevision='keep_view',
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font=dict(color='#FFFFFF')
            )
        )
        
        return fig

    # Run the Dash server in non-blocking mode
    # use_reloader=False and debug=False are important for production/threading
    app.run(debug=False, use_reloader=False, host=host, port=port)

# ====================================================================
# --- 4. MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    try:
        # Start the Dash web server in a separate thread
        # This keeps the main thread free to run the ROS node
        dash_thread = threading.Thread(target=run_dash_app, daemon=True)
        dash_thread.start()
        rospy.loginfo("Dash web server started at: http://localhost:8050")

        # Start the ROS node and its subscribers
        collector = TrajectoryDataCollector()
        rospy.spin() # This loop runs until the ROS node is shut down

    except rospy.ROSInterruptException:
        pass