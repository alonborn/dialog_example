import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
from std_srvs.srv import Trigger  # Standard service type for triggering actions
import tf2_ros
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
import transforms3d
from geometry_msgs.msg import Pose
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
import easy_handeye2 as hec
from easy_handeye2_msgs.srv import TakeSample, SaveCalibration,ComputeCalibration
from tkinter import messagebox
import queue
import time
import numpy as np
import re
import json
import debugpy
import os
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import tf_transformations
from std_srvs.srv import SetBool
import yaml
import pathlib
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, Pose
import time
from rclpy.time import Time
import datetime
import serial
from tkinter import ttk
from functools import partial
from my_robot_interfaces.srv import NudgeJoint
import threading
from my_robot_interfaces.srv import MoveServoToAngle


class TkinterROS(Node):
    counter = 0

    def __init__(self):
        super().__init__('tkinter_ros_node')


        # --- Serial connection to Arduino ---
        # Change '/dev/ttyUSB0' and baudrate to match your setup
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            self.log_with_time('info', "Serial port opened successfully")
        except Exception as e:
            self.serial_port = None
            self.log_with_time('error', f"Failed to open serial port: {e}")

            
        self.gui_queue = queue.Queue()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.zero_velocity_positions = None
        self.num_valid_samples = 0
        self.pose_in_camera = None
        self.arm_is_moving = False
        self.init_cal_poses()
        self.initial_marker_pose_base = None

        self.top_dx_mm = None
        self.top_dy_mm = None
        self.top_angle_deg = None
        self.top_est_height_mm = None

        self.last_joint_update_time = self.get_clock().now()
        self.marker_index = 0
        self.last_pos = ""
        self.filtered_position = None
        self.ema_alpha = 0.2  # default filter constant
        self.calib_dir = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/calibrations'))
        self.calib_path = self.calib_dir / 'ar4_calibration.calib'
        # self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        self.create_subscription(PoseArray, '/aruco_poses', self.aruco_pose_callback, 10)

        # Subscriber to aruco_poses topic
        # self.create_subscription(
        #     PoseArray,
        #     '/aruco_poses',
        #     self.aruco_pose_callback,
        #     qos_profile_sensor_data,
        # )

        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.brick_info_sub = self.create_subscription(
            Float32MultiArray,
            '/brick_info_array',
            self.brick_info_callback,
            10
        )

        self.create_subscription(
            Float32MultiArray,           # Message type
            'brick_top_info',            # Topic name
            self.brick_top_info_callback,# Callback function
            10                           # QoS
        )
        self.latest_brick_center = None  # store camera-frame center
        self.latest_brick_yaw = None


        self.log_with_time('info', 'ArucoPoseFollower initialized, listening to /aruco_poses')


        self.open_gripper_client  = self.create_client(Trigger, '/ar4_hardware_interface_node/open_gripper')
        self.close_gripper_client = self.create_client(Trigger, '/ar4_hardware_interface_node/close_gripper')
        self.move_servo_client = self.create_client(MoveServoToAngle, '/ar4_hardware_interface_node/move_servo_to_angle')



        # Optional: warn if not up yet (won’t block forever)
        for cli, nm in [(self.open_gripper_client,  'open_gripper'),
                        (self.close_gripper_client, 'close_gripper')]:
            if not cli.wait_for_service(timeout_sec=0.5):
                self.log_with_time('warn', f"Service '{nm}' not available yet")


        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')
        self.move_arm_client = self.create_client(MoveToPose, '/ar_move_to_pose')
        self.refresh_transform_client = self.create_client(Trigger, 'refresh_handeye_transform')
        self.last_joint_info = ""
        self._last_aruco_update_time = 0
        # GUI init
        self.init_dialog()

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.log_with_time('info', 'registering ar_move_to_pose service')
        self.root.after(20, self.process_gui_queue) 
        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)
        self.log_with_time('info', 'take_sample service registered')

        # self.init_moveit()
        self.init_right_frame()
        # self.periodic_status_check()
        self.aruco_follower_enabled_client = self.create_client(SetBool, '/set_aruco_follower_enabled')

    def move_servo_to_angle(self, angle_deg: float):
        if not self.move_servo_client.wait_for_service(timeout_sec=1.0):
            self.log_with_time('error', "move_servo_to_angle service not available")
            return False

        req = MoveServoToAngle.Request()
        req.angle_deg = float(angle_deg)

        # Use your existing blocking helper:
        resp = self.call_service_blocking(self.move_servo_client, req, timeout_sec=5.0)
        if resp is None:
            self.log_with_time('error', "move_servo_to_angle timed out / failed")
            return False

        if getattr(resp, "success", False):
            self.log_with_time('info', f"Servo moved to {angle_deg:.1f}°")
            self.status_label.config(text=f"Servo → {angle_deg:.1f}°")
            return True
        else:
            msg = getattr(resp, "message", "(no message)")
            self.log_with_time('warn', f"move_servo_to_angle failed: {msg}")
            self.status_label.config(text=f"Servo move failed: {msg}")
            return False


    def log_with_time(self, level, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        if level == 'info':
            self.get_logger().info(full_message)
        elif level == 'error':
            self.get_logger().error(full_message)
        elif level == 'warn':
            self.get_logger().warn(full_message)
        else:
            self.get_logger().debug(full_message)        


    def brick_info_callback(self, msg):
        if len(msg.data) == 4:
            self.latest_brick_center = tuple(msg.data[0:3])
            self.latest_brick_yaw = msg.data[3]
            # self.log_with_time('info', f"Received brick center: {self.latest_brick_center}, yaw: {self.latest_brick_yaw:.2f}")
        else:
            self.log_with_time('warn' ,"Received unexpected brick info format.")

    def brick_top_info_callback(self, msg: Float32MultiArray):
        """
        Callback for receiving brick top info [dx_mm, dy_mm, angle_deg, est_height_mm].
        """
        if len(msg.data) != 4:
            self.log_with_time('warn' ,f"Received brick_top_info with unexpected length: {len(msg.data)}")
            return

        self.top_dx_mm, self.top_dy_mm, self.top_angle_deg, self.top_est_height_mm = msg.data
        # self.log_with_time('info', 
        #     f"Brick offset: dx={dx_mm:.1f} mm, dy={dy_mm:.1f} mm, "
        #     f"angle={angle_deg:.1f}°, height={est_height_mm:.1f} mm"
        # )

        # Example: trigger an action when the brick is centered
        # if abs(dx_mm) < 5 and abs(dy_mm) < 5:
        #     self.log_with_time('info', "✅ Brick is centered within 5 mm!")



    def process_gui_queue(self):
        while not self.gui_queue.empty():
            task = self.gui_queue.get()
            task()  # Execute the GUI update
        self.root.after(50, self.process_gui_queue)  # keep polling every 50ms

    def init_dialog(self):
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")
        self.root.geometry("1200x800")

        self.mode_var = tk.StringVar(value="Calibration")
        self.cartesian_var = tk.BooleanVar(value=True)

        # Main container
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)

        # LEFT side with tabs
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.tab_main = tk.Frame(self.notebook)
        self.tab_tools = tk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Main")
        self.notebook.add(self.tab_tools, text="Tools")
        
        self.init_calibration_tab()

        # ======== MAIN TAB ========
        self.status_label = tk.Label(self.tab_main, text="Waiting", font=("Arial", 16))
        self.status_label.pack(pady=10)

        # Homing / Calibrate / Go Home
        self.button_row = tk.Frame(self.tab_main)
        self.button_row.pack(pady=10)

        tk.Button(self.button_row, text="Homing", command=self.on_homing_button_click,
                font=("Arial", 14), width=10, height=1).grid(row=0, column=0, padx=5)
        tk.Button(self.button_row, text="Calibrate", command=self.on_calibrate_button_click,
                font=("Arial", 14), width=10, height=1).grid(row=0, column=1, padx=5)
        tk.Button(self.button_row, text="Go Home", command=self.on_go_home_button_click,
                font=("Arial", 14), width=10, height=1).grid(row=0, column=2, padx=5)

        # Pose text + copy button
        self.pos_container = tk.Frame(self.tab_main)
        self.pos_container.pack(pady=10)

        self.pos_text = tk.Text(self.pos_container, height=2, font=("Arial", 10), wrap="word", width=60)
        self.pos_text.pack(side=tk.LEFT)

        self.copy_button = tk.Button(self.pos_container, text="Copy to entries", command=self.copy_pos_to_entries)
        self.copy_button.pack(side=tk.LEFT, padx=5)

        # Pose entries + move
        self.pose_frame = tk.Frame(self.tab_main)
        self.pose_frame.pack(pady=10)

        self.pos_num_label = tk.Label(self.pose_frame, text="#0", font=("Arial", 12))
        self.pos_num_label.grid(row=0, column=0, padx=5)

        self.translation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=21)
        self.translation_entry.grid(row=0, column=2, padx=5)
        self.translation_entry.insert(0, str(self.cal_poses[0][0]))

        self.rotation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=27)
        self.rotation_entry.grid(row=0, column=4, padx=5)
        self.rotation_entry.insert(0, str(self.cal_poses[0][1]))

        tk.Button(self.pose_frame, text="Move!", command=self.on_send_pos_button_click,
                font=("Arial", 14), width=8, height=1).grid(row=0, column=5, padx=5)

        self.cartesian_checkbox = tk.Checkbutton(self.pose_frame, text="Cartesian",
                                                variable=self.cartesian_var, font=("Arial", 12))
        self.cartesian_checkbox.grid(row=0, column=6, padx=5)

        tk.Button(self.pose_frame, text="Save Pos", command=self.save_current_pose).grid(row=0, column=7, padx=5)

        # Sample / calibration buttons
        self.sample_frame = tk.Frame(self.tab_main)
        self.sample_frame.pack(pady=(20, 5))

        tk.Button(self.sample_frame, text="Take Sample", command=self.on_take_sample_button_click,
                font=("Arial", 10), width=8).grid(row=0, column=0, padx=5)
        tk.Button(self.sample_frame, text="Save Samples", command=self.on_save_samples_button_click,
                font=("Arial", 10), width=8).grid(row=0, column=1, padx=5)
        tk.Button(self.sample_frame, text="Move To Marker", command=self.move_to_marker,
                font=("Arial", 10), width=12).grid(row=0, column=2, padx=5)
        tk.Button(self.sample_frame, text="Move to Brick", command=self.move_to_brick_process,
                font=("Arial", 10), width=12).grid(row=0, column=3, padx=5)
        tk.Button(self.sample_frame, text="Auto Calib", command=self.on_auto_calibrate_button_click,
                font=("Arial", 10), width=12).grid(row=0, column=4, padx=5)

        # Jog / adjustment controls
        self.add_pose_adjustment_controls()  # ← contains your further/closer/left/right/up/down
        self.add_adjustment_frame()

        # ======== TOOLS TAB ========
        tk.Label(self.tab_tools, text="Utilities", font=("Arial", 12, "bold")).pack(pady=(8, 4))

        # Gripper controls
        grip = tk.LabelFrame(self.tab_tools, text="Gripper")
        grip.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(grip, text="Open Gripper", command=self.open_gripper_srv, width=12)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(grip, text="Close Gripper", command=self.close_gripper_srv, width=12)\
            .pack(side=tk.LEFT, padx=4, pady=5)

        # Z quick moves
        zgrp = tk.LabelFrame(self.tab_tools, text="Z Height (absolute)")
        zgrp.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(zgrp, text="40 cm", command=lambda: self.move_to_height(0.40), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="30 cm", command=lambda: self.move_to_height(0.30), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="25 cm", command=lambda: self.move_to_height(0.25), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="21 cm", command=lambda: self.move_to_height(0.21), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="20 cm", command=lambda: self.move_to_height(0.20), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="17 cm", command=lambda: self.move_to_height(0.17), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="14 cm", command=lambda: self.move_to_height(0.14), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(zgrp, text="13 cm", command=lambda: self.move_to_height(0.13), width=8)\
            .pack(side=tk.BOTTOM, padx=4, pady=5)
        tk.Button(zgrp, text="12 cm", command=lambda: self.move_to_height(0.12), width=8)\
            .pack(side=tk.LEFT, padx=4, pady=5)

        # Start periodic updates
        self.update_position_label()


    def open_gripper_srv(self):
        req = Trigger.Request()
        res = self.call_service_blocking(self.open_gripper_client, req, timeout_sec=3.0)
        if res is None:
            self.log_with_time('error', "open_gripper: no response / timeout")
            self.gui_queue.put(lambda: self.status_label.config(text="Open gripper failed"))
            return False
        self.log_with_time('info', f"open_gripper: {res.message}")
        self.gui_queue.put(lambda: self.status_label.config(
            text="Gripper opened" if res.success else f"Open failed: {res.message}"))
        return bool(res.success)

    def close_gripper_srv(self):

        self.move_servo_to_angle(50.0)

        # req = Trigger.Request()
        # res = self.call_service_blocking(self.close_gripper_client, req, timeout_sec=3.0)
        # if res is None:
        #     self.log_with_time('error', "close_gripper: no response / timeout")
        #     self.gui_queue.put(lambda: self.status_label.config(text="Close gripper failed"))
        #     return False
        # self.log_with_time('info', f"close_gripper: {res.message}")
        # self.gui_queue.put(lambda: self.status_label.config(
        #     text="Gripper closed" if res.success else f"Close failed: {res.message}"))
        # return bool(res.success)


    def init_calibration_tab(self):
        """Initialize and populate the Calibration tab with ± buttons for each joint."""
        # Create Calibration tab
        self.tab_calibration = tk.Frame(self.notebook)
        self.notebook.add(self.tab_calibration, text="Calibration")

        # Column config
        for c in range(4):
            self.tab_calibration.grid_columnconfigure(c, weight=1, uniform="cal")

        # Header and step size control
        tk.Label(
            self.tab_calibration, text="Joint Nudges (HM)", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=4, pady=(8, 6), sticky="w")

        tk.Label(self.tab_calibration, text="Step size (steps):").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.hm_step_entry = tk.Entry(self.tab_calibration, width=6)
        self.hm_step_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        self.hm_step_entry.insert(0, "5")

        # Service status label
        self.nudge_status_var = tk.StringVar(value="Service: unknown")
        tk.Label(self.tab_calibration, textvariable=self.nudge_status_var).grid(row=1, column=2, columnspan=2, sticky="w")

        # --- Joint 1 ---
        tk.Label(self.tab_calibration, text="1. Base Rotate").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="–", width=6,
                command=partial(self.hm_nudge, 0, -1)).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="+", width=6,
                command=partial(self.hm_nudge, 0, +1)).grid(row=2, column=2, padx=4, pady=4, sticky="ew")

        # --- Joint 2 ---
        tk.Label(self.tab_calibration, text="2. Shoulder").grid(row=3, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="BACK", width=6,
                command=partial(self.hm_nudge, 1, -1)).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="FWD", width=6,
                command=partial(self.hm_nudge, 1, +1)).grid(row=3, column=2, padx=4, pady=4, sticky="ew")

        # --- Joint 3 ---
        tk.Label(self.tab_calibration, text="3. Elbow").grid(row=4, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="UP", width=6,
                command=partial(self.hm_nudge, 2, -1)).grid(row=4, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="DOWN", width=6,
                command=partial(self.hm_nudge, 2, +1)).grid(row=4, column=2, padx=4, pady=4, sticky="ew")

        # --- Joint 4 ---
        tk.Label(self.tab_calibration, text="4. Wrist Pitch").grid(row=5, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="CW", width=6,
                command=partial(self.hm_nudge, 3, -1)).grid(row=5, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="CCW", width=6,
                command=partial(self.hm_nudge, 3, +1)).grid(row=5, column=2, padx=4, pady=4, sticky="ew")

        # --- Joint 5 ---
        tk.Label(self.tab_calibration, text="5. Wrist Rotate").grid(row=6, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="UP", width=6,
                command=partial(self.hm_nudge, 4, -1)).grid(row=6, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="DOWN", width=6,
                command=partial(self.hm_nudge, 4, +1)).grid(row=6, column=2, padx=4, pady=4, sticky="ew")

        # --- Joint 6 ---
        tk.Label(self.tab_calibration, text="6. End Effector").grid(row=7, column=0, sticky="e", padx=6, pady=4)
        tk.Button(self.tab_calibration, text="CW", width=6,
                command=partial(self.hm_nudge, 5, -1)).grid(row=7, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(self.tab_calibration, text="CCW", width=6,
                command=partial(self.hm_nudge, 5, +1)).grid(row=7, column=2, padx=4, pady=4, sticky="ew")

        # --- Create the NudgeJoint service client ---
        self.nudge_service_name = "/ar4_hardware_interface_node/nudge_joint"
        self.nudge_joint_srv_type = NudgeJoint
        self.nudge_joint_client = self.create_client(self.nudge_joint_srv_type, self.nudge_service_name)

        try:
            self.nudge_cli = self.create_client(NudgeJoint, self.nudge_service_name)
        except Exception as e:
            self.log_with_time('error', f"Failed to create NudgeJoint client: {e}")
            # self.nudge_status_var.set("Service: client create failed")



    def copy_pos_to_entries(self):
        """
        Parse the text from `self.pos_text`, extract position and orientation,
        and update `self.translation_entry` and `self.rotation_entry`.
        """
        try:
            # Get the text
            text = self.pos_text.get("1.0", "end").strip()

            # Use regex to extract position and orientation
            import re
            match = re.search(
                r"pos\s*=\s*\((-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)\s*ori\s*=\s*\((-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)",
                text
            )

            if not match:
                self.log_with_time('error' ,"Failed to parse pose from pos_text")
                return

            # Extract groups
            translation = tuple(match.groups()[0:3])
            rotation = tuple(match.groups()[3:7])

            # Format for entries
            translation_str = f"({', '.join(translation)})"
            rotation_str = f"({', '.join(rotation)})"

            # Update entry fields
            self.translation_entry.delete(0, "end")
            self.translation_entry.insert(0, translation_str)

            self.rotation_entry.delete(0, "end")
            self.rotation_entry.insert(0, rotation_str)

        except Exception as e:
            self.log_with_time('error' ,f"Error in copy_pos_to_entries: {e}")


    def on_auto_calibrate_button_click(self):
        if self.pose_in_camera is None:
            messagebox.showwarning("No Marker", "No ArUco marker pose available.")
            return

        # Convert pose from camera to base_link
        try:
            marker_pose_base = self._transform_pose(
                self.pose_in_camera,
                source_frame="camera_color_optical_frame",
                target_frame="base_link"
            )
        except Exception as e:
            self.log_with_time('error' ,f"Failed to transform marker pose: {e}")
            messagebox.showerror("Transform Error", "Could not transform marker pose to base_link.")
            return

        # If first press: save and wait for second press
        if self.initial_marker_pose_base is None:
            self.initial_marker_pose_base = marker_pose_base
            self.status_label.config(text="Marker pose saved. Now move the marker under the arm and press again.")
            return

        # Second press: compute delta and correct calibration
        dx = marker_pose_base.position.x - self.initial_marker_pose_base.position.x
        dy = marker_pose_base.position.y - self.initial_marker_pose_base.position.y

        self.log_with_time('info', f"[Auto Calib] Marker moved: dx={dx:.4f}, dy={dy:.4f}")

        # Load current calibration
        translation, rotation = self.read_calibration_file(self.calib_path)

        # Apply correction (camera moved dx, so base->camera should be corrected -dx)
        translation['x'] -= dx
        translation['y'] -= dy

        self.update_calibration_file(self.calib_path, translation, rotation)
        self.trigger_update_calib_file()

        self.status_label.config(text=f"Calibration adjusted by Δx={-dx:+.3f}, Δy={-dy:+.3f}")
        self.initial_marker_pose_base = None  # reset for next use

    def move_to_brick_process(self):
        self.open_gripper_srv()
        self.refine_pose_with_ee_camera(0.20)                                        
        # time.sleep(0.5)
        self.refine_pose_with_ee_camera(0.13)
        self.close_gripper_srv()
        self.refine_pose_with_ee_camera(0.30)

        # self.refine_pose_with_ee_camera()

        # self.move_to_height (0.17)
        # self.refine_pose_with_ee_camera()
        # self.move_to_height (0.12)


    def _get_hm_step_magnitude(self) -> int:
        raw = self.hm_step_entry.get().strip()
        try:
            mag = int(raw)
        except Exception:
            self.log_with_time('warn', f"Invalid HM step '{raw}', defaulting to 1")
            mag = 1
        if mag == 0:
            self.log_with_time('warn', "HM step of 0 ignored; using 1")
            mag = 1
        return abs(mag)  # magnitude only; sign comes from button


    def hm_nudge(self, joint_index: int, direction: int):
        # direction is ±1 from the button
        magnitude = self._get_hm_step_magnitude()
        steps = int(direction) * magnitude

        client = self.nudge_joint_client
        if client is None:
            self.log_with_time('error', "Nudge client not initialized")
            return

        try:
            if not client.wait_for_service(timeout_sec=0.5):
                # self.nudge_status_var.set("Service: not available")
                self.log_with_time('warn', f"{self.nudge_service_name} not available yet")
                return
        except Exception as e:
            self.log_with_time('error', f"Waiting for {self.nudge_service_name} failed: {e}")
            return

        req = self.nudge_joint_srv_type.Request()
        req.joint_index = int(joint_index)
        req.steps = steps

        self.log_with_time('info', f"Sending NudgeJoint: joint_index={req.joint_index}, steps={req.steps}")
        future = client.call_async(req)

        def _on_done(fut):
            try:
                resp = fut.result()
                if getattr(resp, "success", False):
                    self.log_with_time('info', f"Nudged joint {req.joint_index} by {req.steps} steps")
                    # self.nudge_status_var.set("Service: OK")
                else:
                    msg = getattr(resp, "message", "(no message)")
                    self.log_with_time('warn', f"Nudge failed: {msg}")
                    # self.nudge_status_var.set(f"Service: failed — {msg}")
            except Exception as e:
                self.log_with_time('error', f"Nudge call exception: {e}")
                # self.nudge_status_var.set("Service: error")

        future.add_done_callback(_on_done)

    def refine_pose_with_ee_camera(self,height = 0.15):
        if None in (self.top_dx_mm, self.top_dy_mm, self.top_angle_deg, self.top_est_height_mm):
            self.log_with_time('warn' ,"No brick info from end-effector camera.")
            return

        dx_cam = self.top_dx_mm / 1000.0
        dy_cam = self.top_dy_mm / 1000.0
        est_height_m = self.top_est_height_mm / 1000.0
        rel_rotation_deg = -self.top_angle_deg  # Directly from vision: rotation to apply to EE

        ee_pose = self.get_current_ee_pose()
        if ee_pose is None:
            self.log_with_time('error' ,"Cannot read current end-effector pose.")
            return

        quat = [
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        ]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        roll = -np.pi   #face down
        pitch = 0   #face down
        

        current_yaw_deg = np.degrees(yaw) % 360

        # New yaw = current yaw + relative rotation from vision
        new_yaw_deg = (current_yaw_deg + rel_rotation_deg) % 360

        # self.log_with_time('info', 
        #     f"[EE Rotation] Current EE yaw: {current_yaw_deg:.2f}°, "
        #     f"Relative rotation needed: {rel_rotation_deg:.2f}°, "
        #     f"Target EE yaw: {new_yaw_deg:.2f}°"
        # )

        print('info', 
            f"[EE Rotation] Current EE yaw: {current_yaw_deg:.2f}°, "
            f"Relative rotation needed: {rel_rotation_deg:.2f}°, "
            f"Target EE yaw: {new_yaw_deg:.2f}°"
        )

        # Transform dx/dy to base frame
        R = tf_transformations.quaternion_matrix(quat)[0:3, 0:3]
        offset_base = R @ np.array([dx_cam, dy_cam, 0.0])

        corrected_pose = Pose()
        corrected_pose.position.x += ee_pose.position.x + dx_cam
        corrected_pose.position.y += ee_pose.position.y + dy_cam
        corrected_pose.position.z = height #ee_pose.position.z

        # Keep roll/pitch, update yaw
        yaw_rad = np.radians(new_yaw_deg)
        final_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw_rad)
        corrected_pose.orientation.x = final_quat[0]
        corrected_pose.orientation.y = final_quat[1]
        corrected_pose.orientation.z = final_quat[2]
        corrected_pose.orientation.w = final_quat[3]

        self.send_move_request(corrected_pose, is_cartesian=False)

        final_pose = self.get_current_ee_pose()
        if final_pose:
            _, _, final_yaw = tf_transformations.euler_from_quaternion([
                final_pose.orientation.x,
                final_pose.orientation.y,
                final_pose.orientation.z,
                final_pose.orientation.w,
            ])
            final_yaw_deg = np.degrees(final_yaw) % 360
        
            print('info', 
                f"[EE Rotation] Final EE yaw after motion: {final_yaw_deg:.2f}°"
            )

            
        if final_pose:
            # Format and update the GUI entries with the final pose
            pos = final_pose.position
            ori = final_pose.orientation

            # Format as tuples
            pos_str = f"({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
            ori_str = f"({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})"

            # Update GUI entries (on the main thread)
            def update_entries():
                self.translation_entry.delete(0, tk.END)
                self.translation_entry.insert(0, pos_str)
                self.rotation_entry.delete(0, tk.END)
                self.rotation_entry.insert(0, ori_str)

            self.gui_queue.put(update_entries)


    def add_adjustment_frame(self):
        # Parent is the Main tab
        self.adjustment_frame = tk.Frame(self.tab_main)
        self.adjustment_frame.pack(pady=10, fill="x", expand=True)

        # 7 columns (0..6) because we also place the delta entry in column 6
        for i in range(7):
            self.adjustment_frame.grid_columnconfigure(i, weight=1, uniform="adj")

        tk.Label(self.adjustment_frame, text="Adjust Translation/Rotation",
                font=("Arial", 12)).grid(row=0, column=0, columnspan=7, pady=(0, 4), sticky="w")

        # --- Translation row ---
        btn = tk.Button(self.adjustment_frame, text="Further",
                        command=lambda: self.adjust_calib_translation('x', 1.0))
        btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Closer",
                        command=lambda: self.adjust_calib_translation('x', -1.0))
        btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Left",
                        command=lambda: self.adjust_calib_translation('y', 1.0))
        btn.grid(row=1, column=2, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Right",
                        command=lambda: self.adjust_calib_translation('y', -1.0))
        btn.grid(row=1, column=3, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Down",
                        command=lambda: self.adjust_calib_translation('z', 1.0))
        btn.grid(row=1, column=4, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Up",
                        command=lambda: self.adjust_calib_translation('z', -1.0))
        btn.grid(row=1, column=5, padx=2, pady=2, sticky="ew")

        self.translation_delta_entry = tk.Entry(self.adjustment_frame, font=("Arial", 12), width=6)
        self.translation_delta_entry.grid(row=1, column=6, padx=6, pady=2, sticky="w")
        self.translation_delta_entry.insert(0, "0.01")

        # spacer
        tk.Label(self.adjustment_frame, text="").grid(row=2, column=0)

        # --- Orientation row (single row) ---
        btn = tk.Button(self.adjustment_frame, text="Roll +",
                        command=lambda: self.adjust_calib_orientation('x', 1.0))
        btn.grid(row=3, column=0, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Roll -",
                        command=lambda: self.adjust_calib_orientation('x', -1.0))
        btn.grid(row=3, column=1, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Pitch +",
                        command=lambda: self.adjust_calib_orientation('y', 1.0))
        btn.grid(row=3, column=2, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Pitch -",
                        command=lambda: self.adjust_calib_orientation('y', -1.0))
        btn.grid(row=3, column=3, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Yaw +",
                        command=lambda: self.adjust_calib_orientation('z', 1.0))
        btn.grid(row=3, column=4, padx=2, pady=2, sticky="ew")

        btn = tk.Button(self.adjustment_frame, text="Yaw -",
                        command=lambda: self.adjust_calib_orientation('z', -1.0))
        btn.grid(row=3, column=5, padx=2, pady=2, sticky="ew")

        self.orientation_delta_entry = tk.Entry(self.adjustment_frame, font=("Arial", 12), width=6)
        self.orientation_delta_entry.grid(row=3, column=6, padx=6, pady=2, sticky="w")



    def adjust_calib_translation(self, axis, direction):
        delta = direction * float(self.translation_delta_entry.get())
        # Read current calibration
        translation, rotation = self.read_calibration_file(self.calib_path)
        translation[axis] += delta

        self.update_calibration_file(self.calib_path, translation, rotation)
        self.trigger_update_calib_file()

        self.status_label.config(text=f"{axis.upper()} adjusted by {delta:+.3f}")
        self.move_to_marker()

    def adjust_calib_orientation(self, axis, direction):
        delta = direction * float(self.orientation_delta_entry.get())
        # Read current calibration
        translation, rotation = self.read_calibration_file(self.calib_path)
        rotation[axis] += delta

        self.update_calibration_file(self.calib_path, translation, rotation)
        self.trigger_update_calib_file()

        self.status_label.config(text=f"{axis.upper()} adjusted by {delta:+.3f}")

    def trigger_update_calib_file(self):
        self.log_with_time('info', "Updating calibration file...")
        request = Trigger.Request()
        future = self.refresh_transform_client.call_async(request)

    def get_move_request(self, pose, is_cartesian=True):
        # Create a request
        request = MoveToPose.Request()
        request.pose = pose
        request.cartesian = is_cartesian
        return request
    
    def on_save_samples_button_click(self):
        self.compute_calibration()
        self.save_calibration()
        print("calibration saved")
        

    def on_take_sample_button_click(self):
        self.num_valid_samples += 1
        #time.sleep(3)
        self.take_sample()
        #time.sleep(3)
        if self.num_valid_samples > 2:
            self.compute_calibration()
        print(f"sample no. {self.num_valid_samples} taken - pose " ) 
    
    def init_right_frame(self): 
        self.right_frame = tk.Frame(self.main_frame, bg='lightgray')
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        self.aruco_pose_entry = tk.Text(self.right_frame, font=("Courier", 10), width=21, height=2)
        self.aruco_pose_entry.pack(pady=10)

        self.copy_aruco_button = tk.Button(
            self.right_frame, 
            text="Copy Aruco Pose", 
            font=("Arial", 10), 
            command=self.copy_aruco_pose_to_clipboard
        )
        self.copy_aruco_button.pack(pady=5)
        self.joint_states_var = tk.StringVar()
        self.joint_states_entry = tk.Text(self.right_frame, font=("Courier", 10), width=30, height=10)
        self.joint_states_entry.pack(pady=10)

        tk.Radiobutton(self.right_frame, text="Calibration", variable=self.mode_var, value="Calibration",
                        command=self.on_mode_change).pack(anchor=tk.W)
        tk.Radiobutton(self.right_frame, text="Validation", variable=self.mode_var, value="Validation",
                    command=self.on_mode_change).pack(anchor=tk.W)

    def on_mode_change(self):
        mode = self.mode_var.get()
        enable_follower = (mode == "Validation")  # enable only in validation
        self.log_with_time('info', f"Switching mode to: {mode} (follower {'enabled' if enable_follower else 'disabled'})")

        request = SetBool.Request()
        request.data = enable_follower

        future = self.aruco_follower_enabled_client.call_async(request)

        # Optionally wait (non-blocking)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            self.log_with_time('info', f"Service response: {future.result().message}")
        else:
            self.log_with_time('error' ,"Service call failed")

    def update_status_label(self):
        self.gui_queue.put(lambda: self._update_status_label())
        # self.root.after(0, self.process_gui_queue)

    def _update_status_label(self):
        if self.arm_is_moving == True:
            self.status_label.config(text="Arm Is Moving", fg="red")
        else:
            self.status_label.config(text="Waiting", fg="black")
        self.update_gui_once()

    def periodic_status_check(self):
        self.update_status_label()
        self.root.after(200, self.periodic_status_check)

    # --- GUI Pose Adjustment Controls (to be called inside TkinterROS.__init__()) ---
    def add_pose_adjustment_controls(self):
        """Place pose controls at the bottom of the Main tab with consistent geometry manager.
        NOTE: Avoid mixing pack() and grid() on the same parent. Here we grid() all children
        of pose_ctrl_frame, and only pack() the frame itself into tab_main.
        """
        self.pose_index = 0

        # The frame itself is packed into the tab; inside we will ONLY use grid()
        pose_ctrl_frame = tk.Frame(self.tab_main)
        pose_ctrl_frame.pack(pady=10, fill=tk.X)

        def move_to_current_pose():
            def format_float(f):
                return f"{f:.4f}".rstrip("0").rstrip(".") if "." in f"{f:.4f}" else f"{f:.4f}"
            pos, ori = self.cal_poses[self.pose_index]
            _ = self.create_pose(pos, ori)  # not used directly; just ensures pose validity
            # Update entry widgets
            self.translation_entry.delete(0, tk.END)
            self.translation_entry.insert(0, f"({', '.join(format_float(x) for x in pos)})")
            self.rotation_entry.delete(0, tk.END)
            self.rotation_entry.insert(0, f"({', '.join(format_float(x) for x in ori)})")
            self.send_pose_from_entries()

        def adjust_orientation(axis, delta):
            rotation_str = self.rotation_entry.get()
            try:
                ori = tuple(float(x.strip()) for x in rotation_str.strip("()").split(","))
                euler = tf_transformations.euler_from_quaternion(ori)
                euler = list(euler)
                euler[axis] += delta
                new_ori = tf_transformations.quaternion_from_euler(*euler)
                self.rotation_entry.delete(0, tk.END)
                self.rotation_entry.insert(0, f"({new_ori[0]:.4f}, {new_ori[1]:.4f}, {new_ori[2]:.4f}, {new_ori[3]:.4f})")
                self.send_pose_from_entries()
            except Exception as e:
                self.log_with_time('error', f"Failed to adjust orientation: {e}")

        def prev_pose():
            self.pose_index = max(0, self.pose_index - 1)
            self.pos_num_label.configure(text=f"#{self.pose_index}")
            move_to_current_pose()

        def next_pose():
            if self.pose_index >= len(self.cal_poses) - 1:
                self.pose_index = 0
            else:
                self.pose_index += 1
            self.pos_num_label.configure(text=f"#{self.pose_index}")
            move_to_current_pose()

        # Row 0: header
        tk.Label(pose_ctrl_frame, text="Pose Controls", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=6, pady=(0,6))

        # Row 1: Prev/Next and Move-to
        tk.Button(pose_ctrl_frame, text="Prev Pose", command=prev_pose, width=10).grid(row=1, column=0, padx=4, pady=3)
        tk.Button(pose_ctrl_frame, text="Next Pose", command=next_pose, width=10).grid(row=1, column=1, padx=4, pady=3)
        tk.Button(pose_ctrl_frame, text="Move to Current Pose", command=move_to_current_pose, width=20).grid(row=1, column=2, columnspan=2, padx=4, pady=3)

        # Row 2-3: Orientation jogs
        delta = 0.1
        tk.Button(pose_ctrl_frame, text="Yaw +", command=lambda: adjust_orientation(2,  delta), width=8).grid(row=2, column=0, padx=3, pady=3)
        tk.Button(pose_ctrl_frame, text="Yaw -", command=lambda: adjust_orientation(2, -delta), width=8).grid(row=2, column=1, padx=3, pady=3)
        tk.Button(pose_ctrl_frame, text="Roll +", command=lambda: adjust_orientation(0,  delta), width=8).grid(row=2, column=2, padx=3, pady=3)
        tk.Button(pose_ctrl_frame, text="Roll -", command=lambda: adjust_orientation(0, -delta), width=8).grid(row=2, column=3, padx=3, pady=3)
        tk.Button(pose_ctrl_frame, text="Pitch +", command=lambda: adjust_orientation(1,  delta), width=8).grid(row=3, column=2, padx=3, pady=3)
        tk.Button(pose_ctrl_frame, text="Pitch -", command=lambda: adjust_orientation(1, -delta), width=8).grid(row=3, column=3, padx=3, pady=3)


    # def update_pose_display(self):
    #     pos, ori = self.cal_poses[self.pose_index]
    #     self.current_pose_display.configure(state='normal')
    #     self.current_pose_display.delete("1.0", tk.END)
    #     self.current_pose_display.insert(tk.END, f"{pos}\n{ori}")
    #     self.current_pose_display.configure(state='disabled')

    def save_current_pose(self):
        try:
            translation_str = self.translation_entry.get()
            rotation_str = self.rotation_entry.get()
            new_pos = tuple(float(x.strip()) for x in translation_str.strip('()').split(','))
            new_ori = tuple(float(x.strip()) for x in rotation_str.strip('()').split(','))
            if len(new_pos) == 3 and len(new_ori) == 4:
                self.cal_poses[self.pose_index] = (new_pos, new_ori)
                # update_pose_display()
            else:
                print("Invalid pose length.")
            path = os.path.join(os.path.expanduser("~/ros_ws/src/dialog_example/dialog_example"), "cal_poses.jsonc")

            self.save_calibration_poses(self.cal_poses,path)
        except Exception as e:
            self.log_with_time('error' ,f"Failed to save pose from entries: {e}")

    def save_calibration_poses(self, poses, json_path="cal_poses.jsonc"):
        """
        Save calibration poses to a JSONC file.
        
        :param poses: List of (position, orientation) tuples.
        :param json_path: Path to the output file.
        """
        data = [
            {"position": list(pos), "orientation": list(orient)}
            for pos, orient in poses
        ]

        comment = (
            "// Calibration poses for the robot\n"
            "// Each item has 'position': [x, y, z] and 'orientation': [x, y, z, w]\n"
        )

        with open(json_path, "w") as f:
            f.write(comment)
            json.dump(data, f, indent=2)
            
    def load_calibration_poses(self, json_path="cal_poses.jsonc"):
        with open(json_path, "r") as f:
            content = f.read()
        content = re.sub(r"//.*?$|/\*.*?\*/", "", content, flags=re.MULTILINE | re.DOTALL)
        data = json.loads(content)
        return [(tuple(p["position"]), tuple(p["orientation"])) for p in data]

    def copy_aruco_pose_to_clipboard(self):
        try:
            pose_text = self.aruco_pose_entry.get("1.0", tk.END).strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(pose_text)
            self.root.update()  # now it stays on the clipboard after the window is closed
            joint_text = self.joint_states_entry.get("1.0", tk.END).strip()
            print("Aruco pose copied to clipboard:", pose_text)
            pose_id = self.pose_index if hasattr(self, "pose_index") else "N/A"
            # Also retrieve and save the robot position display
            pos_text = self.pos_text.get("1.0", tk.END).strip()

            # Save both to a file
            with open("copied_pose_log.txt", "a") as f:
                f.write(f"=== Pose Index: {pose_id} ===\n")
                f.write("=== Aruco Pose ===\n")
                f.write(pose_text + "\n")
                f.write("=== Robot Position ===\n")
                f.write(pos_text + "\n\n")
                f.write("===Joint States: ===\n")
                f.write(joint_text + "\n\n")

        except Exception as e:
            self.log_with_time('error' ,f"Error copying Aruco pose to clipboard: {e}")


    def joint_states_callback(self, msg):

            if self.arm_is_moving == True:
                self.zero_velocity_positions = None
                return
            
            now = time.time()
            if now - getattr(self, "_last_joint_update_gui", 0) < 0.5:
                return
            self._last_joint_update_gui = now

            try:
                # Check if all joint velocities are near zero
                all_zero_velocities = all(abs(v) < 1e-5 for v in msg.velocity)

                if all_zero_velocities:
                    if self.zero_velocity_positions is None:
                        # Save current positions for comparison later
                        self.zero_velocity_positions = list(msg.position)
                    # else:
                        # Compare current positions with previously saved positions
                        # for i, (new_pos, ref_pos) in enumerate(zip(msg.position, self.zero_velocity_positions)):
                        #     if abs(new_pos - ref_pos) > 0.1:  # Tolerance in radians
                        #         self.log_with_time('error' ,
                        #             f"Unintended motion detected on joint {i}: "
                        #             f"saved={ref_pos:.4f}, current={new_pos:.4f}, delta={abs(new_pos - ref_pos):.5f}"
                        #         )
                        #         messagebox.showinfo("Info", "Encoder error: unintended movement with zero velocity reported.")
                else:
                    # If robot is moving, clear saved reference
                    self.zero_velocity_positions = None

                # UI update
                joint_info = "\n".join([f"{pos:.4f}," for pos in msg.position])
                self.gui_queue.put(lambda: self.update_joint_states_gui(joint_info))
                

            except Exception as e:
                self.log_with_time('error' ,f"Error in joint_states_callback: {e}")


    def update_joint_states_gui(self, text):
        self.joint_states_entry.configure(state='normal')
        self.joint_states_entry.delete("1.0", tk.END)
        self.joint_states_entry.insert(tk.END, text)
        self.joint_states_entry.configure(state='disabled')



    def update_joint_states_gui(self, joint_info):
        if self.last_joint_info == joint_info:
            return
        self.last_joint_info = joint_info
        self.joint_states_entry.configure(state='normal')
        self.joint_states_entry.delete("1.0", tk.END)
        self.joint_states_entry.insert(tk.END, joint_info)
        self.joint_states_entry.configure(state='disabled')


    def init_cal_poses(self):
        # self.cal_poses =    [[(0.03, -0.38, 0.39), (0.31, -0.56, 0.64, -0.43)],
        #                     [(0.03, -0.39, 0.39), (0.26, -0.48, 0.70, -0.46)],
        #                     [(0.03, -0.38, 0.39), (0.34, -0.43, 0.77, -0.34)],
        #                     [(0.03, -0.38, 0.39), (0.36, -0.48, 0.73, -0.31)],
        #                     [(0.03, -0.38, 0.39), (0.39, -0.47, 0.72, -0.33)],
        #                     [(0.03, -0.38, 0.39), (0.31, -0.52, 0.66, -0.44)]]
    
        from ament_index_python.packages import get_package_share_directory
        path = "/home/alon/ros_ws/src/dialog_example/dialog_example/cal_poses.jsonc"

        self.cal_poses = self.load_calibration_poses(path)


    
    def call_service_blocking(self, client, request, timeout_sec=10.0):
        
        if not client.service_is_ready():
            self.log_with_time('error' ,"Service not available")
            return None
        print(f"Calling service {client.srv_name} with request: {request}")
        future = client.call_async(request)
        done_event = threading.Event()
        result_container = {'result': None}

        def _on_response(fut):
            print(f"Service {client.srv_name} response received")
            self.arm_is_moving = False
            self.update_status_label()
            try:
                result_container['result'] = fut.result()
                # self.log_with_time('info', f"Service call ended, result: {result_container['result']}")
                print(f"Service call ended, result: {result_container['result']}")
            except Exception as e:
                self.log_with_time('error' ,f"Service call failed: {e}")
            finally:
                done_event.set()

        self._update_status_label()
        self.update_gui_once()
        future.add_done_callback(_on_response)

        if not done_event.wait(timeout=timeout_sec):
            self.log_with_time('error' ,"Service call timed out")
            return None

        self._update_status_label()
        self.update_gui_once()
        
        return result_container['result']


        
                        
    def update_position_label(self):
        pose = self.get_current_ee_pose()
        if pose:
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z

            ox = pose.orientation.x
            oy = pose.orientation.y
            oz = pose.orientation.z
            ow = pose.orientation.w
            new_text = "Cur Pose:"
            new_text += (f"pos = ({x:.3f}, {y:.3f}, {z:.3f})"
                        f" ori = ({ox:.3f}, {oy:.3f}, {oz:.3f}, {ow:.3f})")
            if self.last_pos != new_text:
                self.last_pos = new_text
                current_text = self.pos_text.get("1.0", tk.END).strip()

                if new_text != current_text:
                    self.pos_text.configure(state='normal')
                    self.pos_text.delete(1.0, tk.END)
                    self.pos_text.insert(tk.END, new_text)
                    self.pos_text.configure(state='disabled')

        self.root.after(1000, self.update_position_label)


    def spin_once(self):
    
        # Process ROS messages
        rclpy.spin_once(self, timeout_sec=0.1)

        # Process Tkinter GUI events
        self.root.update_idletasks()
        self.root.update()

    def get_current_ee_pose(self) -> Pose:
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                target_frame='base_link',
                source_frame='ee_link',
                time=now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            pose = Pose()
            pose.position = trans.transform.translation
            pose.orientation = trans.transform.rotation
            return pose

        except Exception as e:
            self.log_with_time('warn' ,f"TF lookup failed: {e}")
            return None


    def aruco_pose_callback(self, msg: PoseArray):
        # return
        if not msg.poses:
            self.log_with_time('warn' ,'Received empty PoseArray, skipping.')
            return
        
        current_time = time.time()
        if current_time - self._last_aruco_update_time < 1.0:
            return  # Skip this update if less than 1 second since last

        self._last_aruco_update_time = current_time 

        # ----------------------------
        # 1) Work with pose in camera frame
        # ----------------------------
        self.pose_in_camera = msg.poses[self.marker_index]

        # Display the camera-frame pose in the GUI

        try:
            transformed_pose = self._transform_pose(
                self.pose_in_camera,
                source_frame="camera_color_optical_frame",
                target_frame="base_link"
            )
            p = transformed_pose.position
            suffix = "(B)"
        except Exception as e:
            p = self.pose_in_camera.position
            suffix = "(C)"
        
        pose_text = f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) {suffix}"
        self.gui_queue.put(lambda: self.update_aruco_pose_gui(pose_text))


    def update_aruco_pose_gui(self, pose_text):
        self.aruco_pose_entry.configure(state='normal')
        self.aruco_pose_entry.delete("1.0", tk.END)
        self.aruco_pose_entry.insert(tk.END, pose_text)
        self.aruco_pose_entry.configure(state='disabled')


    def move_to_marker(self):
        if self.pose_in_camera is None:
            self.log_with_time('warn' ,"No transformed ArUco pose available to move to.")
            return

        self.move_to_marker_pose(self.pose_in_camera)

    

    def _transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        """
        Transforms a pose from source_frame to target_frame using TF2,
        and returns the transformed Pose in target_frame.
        """

        try:
            # Wrap the input Pose in a PoseStamped
            stamped_pose_in = PoseStamped()
            stamped_pose_in.header.stamp = self.get_clock().now().to_msg()
            stamped_pose_in.header.frame_id = source_frame
            stamped_pose_in.pose = pose

            # Lookup the transform from source_frame -> target_frame
            now = rclpy.time.Time()
            can_transform = self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                # now,
                self.get_clock().now(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            if not can_transform:
                raise RuntimeError(f"Cannot transform from {source_frame} to {target_frame}!")

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            # self.log_with_time('info', f"------------------------(4)Transform: {transform}")
            # Transform the pose into target_frame
            transformed_pose = do_transform_pose(stamped_pose_in.pose, transform)
            # self.log_with_time('info', f"------------------------(5)Transformed pose: {transformed_pose}")  
            # Optional: publish for visualization if you defined publishers
            # if hasattr(self, 'pose_pub'):
            #     self.pose_pub.publish(transformed_stamped)

            return transformed_pose

        except Exception as e:
            self.log_with_time('error' ,f"[TF ERROR] Failed to transform pose: {e}")
            raise

    def move_to_marker_pose(self, pose_in_camera: Pose):
        """Process a single Pose (in camera frame), transform it, filter it, and move the robot."""
        try:
            self.log_with_time('info', f"--------------------(1)Received pose in camera frame: {pose_in_camera}")
            transformed_pose = self._transform_pose(pose_in_camera,
                                                        "camera_color_optical_frame",
                                                        "base_link")

            self.log_with_time('info', f"---------------------------(2)Transformed pose (after _transform_pose): {transformed_pose}")
        except Exception as e:
            self.log_with_time('error' ,f"Unexpected exception during pose processing: {e}")
            return  # 🟢 Critical: exit early so the rest of the method isn't run on invalid data

        # Continue only if no exception
        # Flip the pose upside down by applying a 180-degree rotation around X
        quat = [
            transformed_pose.orientation.w,
            transformed_pose.orientation.x,
            transformed_pose.orientation.y,
            transformed_pose.orientation.z,
        ]
        x_180_deg_quat = [0, 1, 0, 0]
        flipped_quat = transforms3d.quaternions.qmult(quat, x_180_deg_quat)
        transformed_pose.orientation.w = flipped_quat[0]
        transformed_pose.orientation.x = flipped_quat[1]
        transformed_pose.orientation.y = flipped_quat[2]
        transformed_pose.orientation.z = flipped_quat[3]

        # Offset in Z
        transformed_pose.position.z += 0.10

        self.log_with_time('info', f"------------------(3)Following pose: {transformed_pose}")
        self.move_to(transformed_pose)


    def move_to(self, msg: Pose):

        self.send_move_request(pose=msg,is_cartesian=False)


    def send_move_request(self, pose, is_cartesian=True):
        self.arm_is_moving = True
        self.update_status_label()
        # pose_goal = PoseStamped() 
        # pose_goal.header.frame_id = "base_link"
        # pose_goal.pose = Pose(position = pose.position, orientation = pose.orientation)
        pose = Pose(position = pose.position, orientation = pose.orientation)
        print ("starting to move")
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        motion_type = "Cartesian" if is_cartesian else "Joint-space"
        if not (-0.5 <= pose.position.x <= 0.5 and -0.54 <= pose.position.y <= 0.6 and 0.03 <= pose.position.z <= 0.55):
            print(f"[{now}] 🚫 Refused Motion: {motion_type} target out of bounds! Target: pos=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) ori={pose.orientation}")
            self.arm_is_moving = False
            return False

        request = self.get_move_request(pose, is_cartesian=is_cartesian)
        # Send the request


        response = self.call_service_blocking(self.move_arm_client, request,timeout_sec=4448.0)
        print("Got response:", response)
        return response is not None
 
    def create_pose(self, position, rotation):
        """
        Creates a Pose message from position and rotation values.

        Args:
            position (tuple): (x, y, z) coordinates.
            rotation (tuple): (x, y, z, w) quaternion values.

        Returns:
            Pose: A populated Pose message.
        """
        pose_msg = Pose()

        # Set position
        pose_msg.position.x = position[0]
        pose_msg.position.y = position[1]
        pose_msg.position.z = position[2]

        # Set rotation
        pose_msg.orientation.x = rotation[0]
        pose_msg.orientation.y = rotation[1]
        pose_msg.orientation.z = rotation[2]
        pose_msg.orientation.w = rotation[3]

        return pose_msg

    # def publish_message(self):
    #     msg = String()
    #     msg.data = self.joints_entry.get()  # Send the value from the text field
    #     self.publisher.publish(msg)
        #self.log_with_time('warn' ,'Publishing message: ' + msg.data)

    def on_homing_button_click(self):
        self.trigger_homing_service("Homing in progress...", "Homing successful!", "Homing failed")

    def take_sample(self):
        self.call_service_blocking(self.take_sample_client, TakeSample.Request(), timeout_sec=115.0)
        #self.log_with_time('info', "Sample taken - in dialog_node")
  
    def save_calibration(self):
        self.call_service_blocking(self.save_sample_calibration_client, SaveCalibration.Request(), timeout_sec=115.0)
        self.log_with_time('info', "Calibration saved - in dialog_node")
 
    def compute_calibration(self):
        self.call_service_blocking(self.compute_calibration_client, ComputeCalibration.Request(), timeout_sec=115.0)
        self.log_with_time('info', "Calibration computed - in dialog_node")
    
    def send_pose_from_entries(self):
            try:
                translation_str = self.translation_entry.get()
                rotation_str = self.rotation_entry.get()
                
                translation_str = translation_str.replace(" ", "")
                rotation_str = rotation_str.replace(" ", "")

                translation = tuple(float(x.strip()) for x in translation_str.strip('()').split(','))
                rotation = tuple(float(x.strip()) for x in rotation_str.strip('()').split(','))

                if len(translation) != 3 or len(rotation) != 4:
                    raise ValueError("Invalid pose format")

                pose_msg = self.create_pose(translation, rotation)
                is_cartesian = self.cartesian_var.get()
                self.send_move_request(pose_msg,is_cartesian = is_cartesian)
            except Exception as e:
                print(f"Error sending pose: {e}")
 

    def move_to_height(self, height: float, is_cartesian: bool = False):
        """
        Move the robot to a specified height while preserving X, Y, and orientation.

        Parameters:
            height (float): Desired Z value in meters.
            is_cartesian (bool): Whether to use Cartesian motion.
        """
        try:
            # Get current pose
            current_pose = self.get_current_ee_pose()

            # Extract translation and rotation
            x = current_pose.position.x
            y = current_pose.position.y
            translation = (x, y, height)

            # Preserve current orientation
            orientation = current_pose.orientation
            rotation = (orientation.x, orientation.y, orientation.z, orientation.w)

            # Create updated pose message
            pose_msg = self.create_pose(translation, rotation)

            # Send motion request
            self.send_move_request(pose_msg, is_cartesian=is_cartesian)

        except Exception as e:
            self.log_with_time('error' ,f"Failed to move to new height: {e}")

    def on_calibrate_button_click(self):

        # position = Pose()
        
        # pose = self.create_pose((0.04, -0.31, 0.4), (0.044, -0.702, 0.71, -0.03))
        # print(pose)
        # print ("sending move request to:" + str(pose)) 
        # self.send_move_request(pose)
        # time.sleep(2)
        self.num_valid_samples = 0   
        for pose in self.cal_poses:
            #print("sending move request")
            pose_msg = self.create_pose(pose[0], pose[1])
            #print(pose_msg)
            if self.send_move_request(pose_msg):
                self.num_valid_samples += 1
                #time.sleep(3)
                self.take_sample()
                #time.sleep(3)
                if self.num_valid_samples > 2:
                    self.compute_calibration()
                print(f"sample no. {self.num_valid_samples} taken - pose " + str(pose[0]) + " - " + str(pose[1])) 
            else:
                print(f"move request failed for pose " + str(pose[0]) + " - " + str(pose[1])) 
                
           

        #self.compute_calibration()
        print("calibration computed")
        self.save_calibration()
        print("calibration saved")
        print("---------------done calibrating poses-------------")

    def on_go_home_button_click(self):
        pose = [(0.01, -0.33, 0.49), (0.01, -0.69, 0.72, 0.01)]
        pose_msg = self.create_pose(pose[0], pose[1])
        self.send_move_request(pose_msg,False)
    def on_send_pos_button_click(self):
        self.send_pose_from_entries()

    def trigger_homing_service(self, start_msg, success_msg, failure_msg):
        self.status_label.config(text=start_msg)
        request = Trigger.Request()
        future = self.homing_client.call_async(request)
        future.add_done_callback(lambda f: self.handle_service_response(f, success_msg, failure_msg))

    def update_gui(self, message):
        self.status_label.config(text=message)

    def tk_mainloop(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # This means the window was closed
            self.log_with_time('info', "Tkinter window closed. Shutting down.")
            return

        # Schedule this method again
        self.root.after(20, self.tk_mainloop)



    def on_shutdown(self):
        self.root.quit()

     
    def read_calibration_file(self,filepath):
        """
        Reads the calibration file and returns the translation and rotation.

        Parameters:
            filepath (str): Path to the calibration YAML file.

        Returns:
            tuple: (translation_dict, rotation_dict)
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        translation = data['transform']['translation']
        rotation = data['transform']['rotation']
        
        return translation, rotation

    def update_calibration_file(self, filepath, new_translation, new_rotation):
        """
        Update the translation and rotation in a calibration file.

        Parameters:
            filepath (str): Path to the calibration file (YAML format).
            new_translation (dict): Dictionary with keys 'x', 'y', 'z'.
            new_rotation (dict): Dictionary with keys 'x', 'y', 'z', 'w'.
        """
        import numpy as np

        def to_float(x):
            return float(x) if isinstance(x, (np.integer, np.floating)) else x

        # Load existing calibration file
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Update translation with safe casting
        data['transform']['translation'] = {
            'x': to_float(new_translation['x']),
            'y': to_float(new_translation['y']),
            'z': to_float(new_translation['z']),
        }

        # Update rotation with safe casting
        data['transform']['rotation'] = {
            'x': to_float(new_rotation['x']),
            'y': to_float(new_rotation['y']),
            'z': to_float(new_rotation['z']),
            'w': to_float(new_rotation['w']),
        }

        # Save updated calibration back to file with clean output
        with open(filepath, 'w') as f:
            yaml.dump(data, f, sort_keys=False)

    def update_gui_once(self):
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.01)

#def ros_spin(tkinter_ros):
def ros_spin_executor(executor): # New
    while rclpy.ok():
        # process any ready ROS callbacks, then give the GIL back
        executor.spin_once(timeout_sec=0.01)
        time.sleep(0.005)



def main(): 
    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    # print("Debugger connected.")
   
    rclpy.init()

    # Create GUI + ROS2 app
    tkinter_ros = TkinterROS()
    
    # Start ROS executor in background
    executor = MultiThreadedExecutor(3)
    executor.add_node(tkinter_ros)
    ros_thread = threading.Thread(target=ros_spin_executor, args=(executor,))
    ros_thread.start()
    
    while True:
        try:
            tkinter_ros.update_gui_once()
        except tk.TclError:
            # This means the window was closed
            tkinter_ros.get_logger().info("Tkinter window closed. Shutting down.")
    
    

    # Run Tkinter's custom main loop in the foreground
    try:
        while True:
            time.sleep(0.001)  # just keep the process alive
    except KeyboardInterrupt:
        pass
    finally:
        tkinter_ros.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
