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
import tf_transformations
from std_srvs.srv import SetBool
import yaml
import pathlib
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, Pose
import time
from rclpy.time import Time

class TkinterROS(Node):
    counter = 0
    

    def __init__(self):
        super().__init__('tkinter_ros_node')
        self.gui_queue = queue.Queue()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.zero_velocity_positions = None
        self.num_valid_samples = 0
        self.pose_in_camera = None
        self.arm_is_moving = False
        self.init_cal_poses()

        self.last_joint_update_time = self.get_clock().now()
        self.marker_index = 0
        self.last_pos = ""
        self.filtered_position = None
        self.ema_alpha = 0.2  # default filter constant
        self.calib_dir = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/calibrations'))
        self.calib_path = self.calib_dir / 'ar4_calibration.calib'
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        # self.create_subscription(Point, '/aruco_pose', self.aruco_pose_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        # Subscriber to aruco_poses topic
        self.create_subscription(
            PoseArray,
            '/aruco_poses',
            self.aruco_pose_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('ArucoPoseFollower initialized, listening to /aruco_poses')


        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')
        self.move_arm_client = self.create_client(MoveToPose, '/ar_move_to_pose')
        self.refresh_transform_client = self.create_client(Trigger, 'refresh_handeye_transform')
        self.last_joint_info = ""
        self._last_aruco_update_time = 0
        # GUI init
        self.init_dialog()

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.get_logger().info('registering ar_move_to_pose service')
        self.root.after(20, self.process_gui_queue) 
        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)
        self.get_logger().info('take_sample service registered')

        # self.init_moveit()
        self.init_right_frame()
        self.periodic_status_check()
        self.aruco_follower_enabled_client = self.create_client(SetBool, '/set_aruco_follower_enabled')
        

    def process_gui_queue(self):
        while not self.gui_queue.empty():
            task = self.gui_queue.get()
            task()  # Execute the GUI update
        self.root.after(50, self.process_gui_queue)  # keep polling every 50ms

    def init_dialog(self):
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")
        self.root.geometry("1000x600")
        self.mode_var = tk.StringVar(value="Calibration")
        self.cartesian_var = tk.BooleanVar(value=True)
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        self.status_label = tk.Label(self.left_frame, text="Waiting", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.button_row = tk.Frame(self.left_frame)
        self.button_row.pack(pady=10)

        self.homing_button = tk.Button(self.button_row, text="Homing", command=self.on_homing_button_click, font=("Arial", 14), width=10, height=1)
        self.homing_button.grid(row=0, column=0, padx=5)

        self.calibrate_button = tk.Button(self.button_row, text="Calibrate", command=self.on_calibrate_button_click, font=("Arial", 14), width=10, height=1)
        self.calibrate_button.grid(row=0, column=1, padx=5)

        self.validate_button = tk.Button(self.button_row, text="Go Home", command=self.on_go_home_button_click, font=("Arial", 14), width=10, height=1)
        self.validate_button.grid(row=0, column=2, padx=5)

        self.joints_entry = tk.Entry(self.left_frame, font=("Arial", 14), width=10)
        self.joints_entry.pack(pady=10)
        self.joints_entry.insert(0, "000001")

        self.pos_text = tk.Text(self.left_frame, height=2, font=("Arial", 10), wrap="word", width=60)
        self.pos_text.pack(pady=10)
        self.pos_text.configure(state='disabled')

        self.pose_frame = tk.Frame(self.left_frame)
        self.pose_frame.pack(pady=(10))

        self.pos_num_label = tk.Label(self.pose_frame, text="#0", font=("Arial", 12))
        self.pos_num_label.grid(row=0, column=0, padx=5)

        self.translation_label = tk.Label(self.pose_frame, text="Translation:", font=("Arial", 10))
        self.translation_label.grid(row=0, column=1, padx=5)

        self.translation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=21)
        self.translation_entry.grid(row=0, column=2, padx=5)
        self.translation_entry.insert(0, str(self.cal_poses[0][0]))

        self.rotation_label = tk.Label(self.pose_frame, text="Rotation:", font=("Arial", 10))
        self.rotation_label.grid(row=0, column=3, padx=5)

        self.rotation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=27)
        self.rotation_entry.grid(row=0, column=4, padx=5)
        self.rotation_entry.insert(0, str(self.cal_poses[0][1]))

        self.send_pos_button = tk.Button(self.pose_frame, text="Move!", command=self.on_send_pos_button_click, font=("Arial", 14), width=8, height=1)
        self.send_pos_button.grid(row=0, column=5, padx=5)

        self.cartesian_checkbox = tk.Checkbutton(self.pose_frame,text="Cartesian",variable=self.cartesian_var,font=("Arial", 12))
        self.cartesian_checkbox.grid(row=0, column=6, padx=5)

        tk.Button(self.pose_frame, text="Save Pos", command=self.save_current_pose).grid(row=0, column=7, columnspan=1)

        self.sample_frame = tk.Frame(self.left_frame)
        self.sample_frame.pack(pady=(20, 5))

        self.take_sample_button = tk.Button(self.sample_frame, text="Take Sample", command=self.on_take_sample_button_click, font=("Arial", 10), width=8, height=1)
        self.take_sample_button.grid(row=0, column=0, padx=5)

        self.save_sample_button = tk.Button(self.sample_frame, text="Save Samples", command=self.on_save_samples_button_click, font=("Arial", 10), width=8, height=1)
        self.save_sample_button.grid(row=0, column=1, padx=5)

        self.auto_adjust_button = tk.Button(self.sample_frame,text="Move To Marker",command=self.move_to_marker, font=("Arial", 10), width=12, height=1)
        self.auto_adjust_button.grid(row=0, column=2, padx=5)

        self.add_pose_adjustment_controls()
        # Translation adjustment controls
        self.add_adjustment_frame()

        self.update_position_label()
        self.root.after(100, self.tk_mainloop)



    def add_adjustment_frame(self):
        self.adjustment_frame = tk.Frame(self.left_frame)
        self.adjustment_frame.pack(pady=10)

        tk.Label(self.adjustment_frame, text="Adjust Translation/Rotation", font=("Arial", 12)).grid(row=0, column=0, columnspan=6)

        # X Axis
        tk.Button(self.adjustment_frame, text="Further", command=lambda: self.adjust_calib_translation('x', 1.0), width=6).grid(row=1, column=0)
        tk.Button(self.adjustment_frame, text="Closer", command=lambda: self.adjust_calib_translation('x', -1.0), width=6).grid(row=1, column=1)

        # Y Axis
        tk.Button(self.adjustment_frame, text="Left", command=lambda: self.adjust_calib_translation('y', 1.0), width=6).grid(row=1, column=2)
        tk.Button(self.adjustment_frame, text="Right", command=lambda: self.adjust_calib_translation('y', -1.0), width=6).grid(row=1, column=3)

        # Z Axis
        tk.Button(self.adjustment_frame, text="Down", command=lambda: self.adjust_calib_translation('z', 1.0), width=6).grid(row=1, column=4)
        tk.Button(self.adjustment_frame, text="Up", command=lambda: self.adjust_calib_translation('z', -1.0), width=6).grid(row=1, column=5)

        # Delta Entry
        self.translation_delta_entry = tk.Entry(self.adjustment_frame, font=("Arial", 12), width=6)
        self.translation_delta_entry.grid(row=1, column=6, padx=5)
        self.translation_delta_entry.insert(0, "0.01")



        # Roll
        tk.Button(self.adjustment_frame, text="Roll +", command=lambda: self.adjust_calib_orientation('x',1.0), width=6).grid(row=3, column=0)
        tk.Button(self.adjustment_frame, text="Roll -", command=lambda: self.adjust_calib_orientation('x',-1.0), width=6).grid(row=3, column=1)

        # Pitch
        tk.Button(self.adjustment_frame, text="Pitch +", command=lambda: self.adjust_calib_orientation('y',1.0), width=6).grid(row=3, column=2)
        tk.Button(self.adjustment_frame, text="Pitch -", command=lambda: self.adjust_calib_orientation('y',-1.0), width=6).grid(row=3, column=3)

        # Yaw
        tk.Button(self.adjustment_frame, text="Yaw +", command=lambda: self.adjust_calib_orientation('z',1.0), width=6).grid(row=3, column=4)
        tk.Button(self.adjustment_frame, text="Yaw -", command=lambda: self.adjust_calib_orientation('z',-1.0), width=6).grid(row=3, column=5)

        # Delta Entry
        self.orientation_delta_entry = tk.Entry(self.adjustment_frame, font=("Arial", 12), width=6)
        self.orientation_delta_entry.grid(row=3, column=6, padx=5)
        self.orientation_delta_entry.insert(0, "0.01")

    def adjust_calib_translation(self, axis, direction):
        delta = direction * float(self.translation_delta_entry.get())
        # Read current calibration
        translation, rotation = self.read_calibration_file(self.calib_path)
        translation[axis] += delta

        self.update_calibration_file(self.calib_path, translation, rotation)
        self.trigger_update_calib_file()

        self.status_label.config(text=f"{axis.upper()} adjusted by {delta:+.3f}")

    def adjust_calib_orientation(self, axis, direction):
        delta = direction * float(self.orientation_delta_entry.get())
        # Read current calibration
        translation, rotation = self.read_calibration_file(self.calib_path)
        rotation[axis] += delta

        self.update_calibration_file(self.calib_path, translation, rotation)
        self.trigger_update_calib_file()

        self.status_label.config(text=f"{axis.upper()} adjusted by {delta:+.3f}")

    def trigger_update_calib_file(self):
        self.get_logger().info("Updating calibration file...")
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
        self.joint_states_entry = tk.Text(self.right_frame, font=("Courier", 10), width=21, height=10)
        self.joint_states_entry.pack(pady=10)

        tk.Radiobutton(self.right_frame, text="Calibration", variable=self.mode_var, value="Calibration",
                        command=self.on_mode_change).pack(anchor=tk.W)
        tk.Radiobutton(self.right_frame, text="Validation", variable=self.mode_var, value="Validation",
                    command=self.on_mode_change).pack(anchor=tk.W)

    def on_mode_change(self):
        mode = self.mode_var.get()
        enable_follower = (mode == "Validation")  # enable only in validation
        self.get_logger().info(f"Switching mode to: {mode} (follower {'enabled' if enable_follower else 'disabled'})")

        request = SetBool.Request()
        request.data = enable_follower

        future = self.aruco_follower_enabled_client.call_async(request)

        # Optionally wait (non-blocking)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            self.get_logger().info(f"Service response: {future.result().message}")
        else:
            self.get_logger().error("Service call failed")

    def update_status_label(self):
        if self.arm_is_moving == True:
            self.status_label.config(text="Arm Is Moving", fg="red")
        else:
            self.status_label.config(text="Waiting", fg="black")

    def periodic_status_check(self):
        self.update_status_label()
        self.root.after(200, self.periodic_status_check)

    # --- GUI Pose Adjustment Controls (to be called inside TkinterROS.__init__()) ---
    def add_pose_adjustment_controls(self):
        self.pose_index = 0
        pose_ctrl_frame = tk.Frame(self.left_frame)
        pose_ctrl_frame.pack(pady=10)

        # self.current_pose_display = tk.Text(pose_ctrl_frame, height=3, font=("Courier", 10), width=40)
        # self.current_pose_display.grid(row=0, column=0, columnspan=6, pady=5)

        def move_to_current_pose():

            def format_float(f):
                return f"{f:.4f}".rstrip("0").rstrip(".") if "." in f"{f:.4f}" else f"{f:.4f}"

            pos, ori = self.cal_poses[self.pose_index]
            pose_msg = self.create_pose(pos, ori)
            

            # Update pose display
            # self.current_pose_display.configure(state='normal')
            # self.current_pose_display.delete("1.0", tk.END)
            # self.current_pose_display.insert(tk.END, f"{pos}\n{ori}")
            # self.current_pose_display.configure(state='disabled')

            # Also update entry widgets
            self.translation_entry.delete(0, tk.END)
            #self.translation_entry.insert(0, f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
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
                self.get_logger().error(f"Failed to adjust orientation: {e}")


        def prev_pose():
            self.pose_index = max(0, self.pose_index - 1)
            self.pos_num_label.configure(text=f"#{self.pose_index}")
            #self.copy_aruco_pose_to_clipboard()
            move_to_current_pose()

        def next_pose():
            if self.pose_index >= len(self.cal_poses) - 1:
                self.pose_index = 0  # Loop back to the first pose if at the end
                return
            self.pose_index = min(len(self.cal_poses) - 1, self.pose_index + 1)
            self.pos_num_label.configure(text=f"#{self.pose_index}")
            #self.copy_aruco_pose_to_clipboard()
            move_to_current_pose()

        tk.Button(pose_ctrl_frame, text="Prev Pose", command=prev_pose).grid(row=1, column=0)
        tk.Button(pose_ctrl_frame, text="Next Pose", command=next_pose).grid(row=1, column=1)       
        delta = 0.1
        tk.Button(pose_ctrl_frame, text="Roll +", command=lambda: adjust_orientation(0, delta)).grid(row=1, column=2)
        tk.Button(pose_ctrl_frame, text="Roll -", command=lambda: adjust_orientation(0, -delta)).grid(row=1, column=3)
        tk.Button(pose_ctrl_frame, text="Pitch +", command=lambda: adjust_orientation(1, delta)).grid(row=2, column=2)
        tk.Button(pose_ctrl_frame, text="Pitch -", command=lambda: adjust_orientation(1, -delta)).grid(row=2, column=3)
        tk.Button(pose_ctrl_frame, text="Yaw +", command=lambda: adjust_orientation(2, delta)).grid(row=2, column=0)
        tk.Button(pose_ctrl_frame, text="Yaw -", command=lambda: adjust_orientation(2, -delta)).grid(row=2, column=1)
      

        #self.update_pose_display()

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
            self.get_logger().error(f"Failed to save pose from entries: {e}")

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
            self.get_logger().error(f"Error copying Aruco pose to clipboard: {e}")


    def pose_to_matrix(self,pose):
        """Convert a ROS geometry_msgs/Pose to a 4x4 transformation matrix."""
        translation = np.array([pose.position.x, pose.position.y, pose.position.z])
        quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        rot = transforms3d.quaternions.quat2mat(quat)
        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3] = translation
        return T

        
    def auto_adjust_calibration_from_poses(self, marker_pose, ee_pose):
        """
        Given the marker's and EE's poses in the same frame (base_link),
        compute the XY offset between them and correct the base->marker calibration.
        """
        T_base_marker = self.pose_to_matrix(marker_pose)
        T_base_ee = self.pose_to_matrix(ee_pose)

        # Compute T_marker_ee = inv(T_base_marker) @ T_base_ee
        T_marker_base = np.linalg.inv(T_base_marker)
        T_marker_ee = T_marker_base @ T_base_ee

        dx, dy, dz = T_marker_ee[0:3, 3]
        self.get_logger().info(f"[AUTO CALIB] Offset detected: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

        # Read current calibration translation & rotation
        translation, rotation = self.read_calibration_file(self.calib_path)

        # Apply corrections: shift calibration by -dx, -dy
        translation[0] -= dx
        translation[1] -= dy

        # self.update_calibration_file(self.calib_path, translation, rotation)
        # self.trigger_update_calib_file()

        self.status_label.config(text=f"Calibration adjusted by dx={-dx:+.3f}, dy={-dy:+.3f}")

    def joint_states_callback(self, msg):

        if self.arm_is_moving == True:
            self.zero_velocity_positions = None
            return
        
        now = self.get_clock().now()
        if (now - self.last_joint_update_time).nanoseconds < 1e9:
            return  # Skip if less than 1 second since last update
        self.last_joint_update_time = now

        try:
            # Check if all joint velocities are near zero
            all_zero_velocities = all(abs(v) < 1e-5 for v in msg.velocity)

            if all_zero_velocities:
                if self.zero_velocity_positions is None:
                    # Save current positions for comparison later
                    self.zero_velocity_positions = list(msg.position)
                else:
                    # Compare current positions with previously saved positions
                    for i, (new_pos, ref_pos) in enumerate(zip(msg.position, self.zero_velocity_positions)):
                        if abs(new_pos - ref_pos) > 0.1:  # Tolerance in radians
                            self.get_logger().error(
                                f"Unintended motion detected on joint {i}: "
                                f"saved={ref_pos:.4f}, current={new_pos:.4f}, delta={abs(new_pos - ref_pos):.5f}"
                            )
                            messagebox.showinfo("Info", "Encoder error: unintended movement with zero velocity reported.")
            else:
                # If robot is moving, clear saved reference
                self.zero_velocity_positions = None

            # UI update
            joint_info = "\n".join([f"{pos:.4f}," for pos in msg.position])
            self.gui_queue.put(lambda: self.update_joint_states_gui(joint_info))
            

        except Exception as e:
            self.get_logger().error(f"Error in joint_states_callback: {e}")



    def update_joint_states_gui(self, joint_info):
        if self.last_joint_info == joint_info:
            return
        self.last_joint_info = joint_info
        self.joint_states_entry.configure(state='normal')
        self.joint_states_entry.delete("1.0", tk.END)
        self.joint_states_entry.insert(tk.END, joint_info)
        self.joint_states_entry.configure(state='disabled')


    def test_timer_callback(self):
        self.get_logger().info('ROS loop is alive!')

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
            self.get_logger().error("Service not available")
            return None

        future = client.call_async(request)
        done_event = threading.Event()
        result_container = {'result': None}

        def _on_response(fut):
            self.arm_is_moving = False
            try:
                result_container['result'] = fut.result()
                self.get_logger().info(f"Service call ended, result: {result_container['result']}")
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")
            finally:
                done_event.set()

        future.add_done_callback(_on_response)

        if not done_event.wait(timeout=timeout_sec):
            self.get_logger().error("Service call timed out")
            return None

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

        self.root.after(500, self.update_position_label)


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
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None


    def aruco_pose_callback(self, msg: PoseArray):
        if not msg.poses:
            self.get_logger().warn('Received empty PoseArray, skipping.')
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
        pose_text = f"({self.pose_in_camera.position.x:.2f}, {self.pose_in_camera.position.y:.2f}, {self.pose_in_camera.position.z:.2f})"
        self.gui_queue.put(lambda: self.update_aruco_pose_gui(pose_text))


    def update_aruco_pose_gui(self, pose_text):
        self.aruco_pose_entry.configure(state='normal')
        self.aruco_pose_entry.delete("1.0", tk.END)
        self.aruco_pose_entry.insert(tk.END, pose_text)
        self.aruco_pose_entry.configure(state='disabled')


    def move_to_marker(self):
        if self.pose_in_camera is None:
            self.get_logger().warn("No transformed ArUco pose available to move to.")
            return

        self.move_to_marker_pose(self.pose_in_camera)

    def _transform_pose2(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        """Transforms a pose from the source_frame to the target_frame,
        publishes both the direct transformed pose and one offset in z by +5cm,
        and returns the direct transformed pose."""
        
        # Look up the transform from source_frame -> target_frame
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        
        # Transform the original pose
        transformed_pose = do_transform_pose(pose, transform)
        # Publish the direct transformed pose
        stamped_pose = PoseStamped()
        stamped_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        
        # Create a copy of the original pose with z offset +5cm before transforming
        pose_with_offset = Pose()
        pose_with_offset.position = Point(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z + 0.05  # add 5 cm in z
        )
        pose_with_offset.orientation = pose.orientation
        
        transformed_pose_offset = do_transform_pose(pose_with_offset, transform)
        
        # Publish the offset pose (useful for visualizations)
        stamped_offset_pose = PoseStamped()
        stamped_offset_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_offset_pose.header.frame_id = target_frame
        stamped_offset_pose.pose = transformed_pose_offset
        self.target_pose_pub.publish(stamped_offset_pose)
        
        return transformed_pose

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
                now,
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
            self.get_logger().info(f"------------------------(4)Transform: {transform}")
            # Transform the pose into target_frame
            transformed_pose = do_transform_pose(stamped_pose_in.pose, transform)
            self.get_logger().info(f"------------------------(5)Transformed pose: {transformed_pose}")  
            # Optional: publish for visualization if you defined publishers
            # if hasattr(self, 'pose_pub'):
            #     self.pose_pub.publish(transformed_stamped)

            return transformed_pose

        except Exception as e:
            self.get_logger().error(f"[TF ERROR] Failed to transform pose: {e}")
            raise

    def move_to_marker_pose(self, pose_in_camera: Pose):
        """Process a single Pose (in camera frame), transform it, filter it, and move the robot."""
        try:
            self.get_logger().info(f"--------------------(1)Received pose in camera frame: {pose_in_camera}")
            transformed_pose = self._transform_pose(pose_in_camera,
                                                        "camera_color_optical_frame",
                                                        "base_link")

            self.get_logger().info(f"---------------------------(2)Transformed pose (after _transform_pose): {transformed_pose}")
        except Exception as e:
            self.get_logger().error(f"Unexpected exception during pose processing: {e}")
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

        self.get_logger().info(f"------------------(3)Following pose: {transformed_pose}")
        self.move_to(transformed_pose)


    def move_to(self, msg: Pose):

        self.send_move_request(pose=msg,is_cartesian=False)


    def send_move_request(self, pose, is_cartesian=True):
        self.arm_is_moving = True
        # pose_goal = PoseStamped() 
        # pose_goal.header.frame_id = "base_link"
        # pose_goal.pose = Pose(position = pose.position, orientation = pose.orientation)
        pose = Pose(position = pose.position, orientation = pose.orientation)
        print ("starting to move")

        request = self.get_move_request(pose, is_cartesian=is_cartesian)
        # Send the request

        def _done_moving(fut):
            print("[DEBUG] service done callback triggered")
            self.arm_is_moving = False


        response = self.call_service_blocking(self.move_arm_client, request,timeout_sec=8.0)
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
        #self.get_logger().warn('Publishing message: ' + msg.data)

    def on_homing_button_click(self):
        self.trigger_homing_service("Homing in progress...", "Homing successful!", "Homing failed")

    def take_sample(self):
        self.call_service_blocking(self.take_sample_client, TakeSample.Request(), timeout_sec=115.0)
        #self.get_logger().info("Sample taken - in dialog_node")
  
    def save_calibration(self):
        self.call_service_blocking(self.save_sample_calibration_client, SaveCalibration.Request(), timeout_sec=115.0)
        self.get_logger().info("Calibration saved - in dialog_node")
 
    def compute_calibration(self):
        self.call_service_blocking(self.compute_calibration_client, ComputeCalibration.Request(), timeout_sec=115.0)
        self.get_logger().info("Calibration computed - in dialog_node")
    
    def send_pose_from_entries(self):
            try:
                translation_str = self.translation_entry.get()
                rotation_str = self.rotation_entry.get()

                translation = tuple(float(x.strip()) for x in translation_str.strip('()').split(','))
                rotation = tuple(float(x.strip()) for x in rotation_str.strip('()').split(','))

                if len(translation) != 3 or len(rotation) != 4:
                    raise ValueError("Invalid pose format")

                pose_msg = self.create_pose(translation, rotation)
                is_cartesian = self.cartesian_var.get()
                self.send_move_request(pose_msg,is_cartesian = is_cartesian)
            except Exception as e:
                print(f"Error sending pose: {e}")
 
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

    def handle_service_response(self, future, success_msg, failure_msg):
        try:
            response = future.result()
            if response.success:
                message = success_msg
            else:
                message = failure_msg + f": {response.message}"
        except Exception as e:
            message = f"Service call failed: {str(e)}"
        
        # 🟢 Schedule GUI update safely in main thread:
        self.gui_queue.put(lambda: self.update_gui(message))

    def update_gui(self, message):
        self.status_label.config(text=message)

    def tk_mainloop(self):
        self.root.update_idletasks()
        self.root.update()
        self.root.after(100, self.tk_mainloop)

    def on_shutdown(self):
        self.root.quit()

    
    def service_exists(self, service_name):
        """Check if a service exists by looking it up in the ROS graph"""
        service_names_and_types = self.get_service_names_and_types()
        return any(service_name == name for name, _ in service_names_and_types)
    
    def topic_exists(self, topic_name):
        """Check if a topic exists by looking it up in the ROS graph"""
        topic_names_and_types = self.get_topic_names_and_types()
        return any(topic_name == name for name, _ in topic_names_and_types)
    
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

    def update_calibration_file(self,filepath, new_translation, new_rotation):
        """
        Update the translation and rotation in a calibration file.

        Parameters:
            filepath (str): Path to the calibration file (YAML format).
            new_translation (dict): Dictionary with keys 'x', 'y', 'z'.
            new_rotation (dict): Dictionary with keys 'x', 'y', 'z', 'w'.
        """
        # Load existing calibration file
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Update translation
        data['transform']['translation'] = {
            'x': new_translation['x'],
            'y': new_translation['y'],
            'z': new_translation['z'],
        }

        # Update rotation
        data['transform']['rotation'] = {
            'x': new_rotation['x'],
            'y': new_rotation['y'],
            'z': new_rotation['z'],
            'w': new_rotation['w'],
        }

        # Save updated calibration back to file
        with open(filepath, 'w') as f:
            yaml.dump(data, f, sort_keys=False)


#def ros_spin(tkinter_ros):
def ros_spin_executor(executor): # New
    try:
        #rclpy.spin(tkinter_ros)
        executor.spin()
    except Exception as e:
        #tkinter_ros.get_logger().error(f"Error in ROS spin loop: {e}")
        print(f"Error in ROS executor spin loop: {e}")



def main(): 
    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Ensures the debugger connects before continuing
    # print("Debugger connected.")
   
    rclpy.init() 

    tkinter_ros = TkinterROS()
    
    executor = MultiThreadedExecutor()
    executor.add_node(tkinter_ros)

    # Start ROS spinning in a separate thread
    #ros_thread = threading.Thread(target=ros_spin, args=(tkinter_ros,))
    ros_thread = threading.Thread(target=ros_spin_executor, args=(executor,)) # New
    ros_thread.start()

    try:
        # Run Tkinter main loop in the main thread
        tkinter_ros.root.mainloop()

    except KeyboardInterrupt:
        pass

    # Gracefully shutdown
    tkinter_ros.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
