import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
from std_srvs.srv import Trigger  # Standard service type for triggering actions
from pymoveit2 import MoveIt2

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Pose, Point, Quaternion
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
from my_robot_interfaces.action import MoveToPoseAc  # Import the custom service type
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
import easy_handeye2_msgs as ehm
import easy_handeye2 as hec
from easy_handeye2_msgs.srv import TakeSample, SaveCalibration,ComputeCalibration
import queue
from rclpy.action import ActionClient
import time
from . import Mover
from geometry_msgs.msg import Point
import re
import json
import debugpy
import os
import tf_transformations
import math
import sys

from ament_index_python.packages import get_package_share_directory

class TkinterROS(Node):
    counter = 0

    arm_is_available = False


    def __init__(self):
        self.zero_velocity_positions = None
        self.num_valid_samples =0
        super().__init__('tkinter_ros_node')
        self.init_cal_poses()
        # self.mover = Mover.Mover(self)
        self.last_joint_update_time = self.get_clock().now()
        self.last_pos = ""
        # Set up ROS publisher
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        #self.timer = self.create_timer(1.0, self.publish_message)
        
        self.create_subscription(Point, '/aruco_pose', self.aruco_pose_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.test_timer_callback_group = ReentrantCallbackGroup()

        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')

        # Subscribe to joint states
        
        self.last_joint_info = ""
        # Set up Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")
        self.root.geometry("1000x500")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

 
        self.label = tk.Label(self.left_frame, text="Waiting for messages...", font=("Arial", 16))
        self.label.pack(pady=10)

       # Create a new frame to hold the row of buttons
        self.button_row = tk.Frame(self.left_frame)
        self.button_row.pack(pady=10)

        # Homing button
        self.homing_button = tk.Button(self.button_row, text="Homing", command=self.on_homing_button_click, font=("Arial", 14), width=10, height=1)
        self.homing_button.grid(row=0, column=0, padx=5)

        # Calibrate button
        self.calibrate_button = tk.Button(self.button_row, text="Calibrate", command=self.on_calibrate_button_click, font=("Arial", 14), width=10, height=1)
        self.calibrate_button.grid(row=0, column=1, padx=5)

        # Go Home button
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

        self.sample_frame = tk.Frame(self.left_frame)
        self.sample_frame.pack(pady=(20, 5))


        self.take_sample_button = tk.Button(self.sample_frame, text="Take Sample", command=self.on_take_sample_button_click, font=("Arial", 10), width=8, height=1)
        self.take_sample_button.grid(row=0, column=0, padx=5)

        self.save_sample_button = tk.Button(self.sample_frame, text="Save Samples", command=self.on_save_samples_button_click, font=("Arial", 10), width=8, height=1)
        self.save_sample_button.grid(row=0, column=1, padx=5)

        tk.Button(self.pose_frame, text="Save Pos", command=self.save_current_pose).grid(row=0, column=6, columnspan=1)

        self.update_position_label()
        self.root.after(100, self.tk_mainloop)

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        self.get_logger().info('registering ar_move_to_pose service')
        #self.move_client = self.create_client(MoveToPose, 'ar_move_to_pose')
        
        #self.get_logger().info('ar_move_to_pose service registered')

        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)

        self.get_logger().info('take_sample service registered')
        self.arm_is_available = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.add_pose_adjustment_controls()
        self.init_moveit()
        self.init_right_frame()
        
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



    def init_moveit(self):
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=ReentrantCallbackGroup()
        )

        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 1.0
        self.moveit2.planning_time = 5.0  # Timeout in seconds

        # Scale down velocity and acceleration of joints (percentage of maximum)
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5
        self.moveit2.cartesian_avoid_collisions = False
        self.moveit2.cartesian_jump_threshold = 0.0

    def MoveArm(self, position, quat_xyzw):
        time.sleep(0.5) #for some reason the arm is not available immediately after the call
        # Get parameters
        # cartesian = self.self.node.get_parameter("cartesian").get_parameter_value().bool_value
        # cartesian_max_step = self.self.node.get_parameter("cartesian_max_step").get_parameter_value().double_value
        # cartesian_fraction_threshold = self.self.node.get_parameter("cartesian_fraction_threshold").get_parameter_value().double_value
        # cartesian_jump_threshold = self.self.node.get_parameter("cartesian_jump_threshold").get_parameter_value().double_value
        # cartesian_avoid_collisions = self.self.node.get_parameter("cartesian_avoid_collisions").get_parameter_value().bool_value

        # Move to pose goal
        # self.moveit2.move_to_pose(
        #     position=position,
        #     quat_xyzw=quat_xyzw,
        #     synchronous=synchronous,
        #     cancel_after_secs=cancel_after_secs,
        #     cartesian=cartesian,
        #     cartesian_max_step=cartesian_max_step,
        #     cartesian_fraction_threshold=cartesian_fraction_threshold,
        #     cartesian_jump_threshold=cartesian_jump_threshold,
        #     cartesian_avoid_collisions=cartesian_avoid_collisions,
        # )
        pose_goal = PoseStamped() 
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = Pose(position = position, orientation = quat_xyzw)
        print ("starting to move")
        ret_val = False
        try:
            self.moveit2.move_to_pose(pose=pose_goal,
                                        #synchronous=True,
                                        #cancel_after_secs=0.0,
                                        #cartesian=False,
                                        #cartesian_max_step=0.0025,
                                        #cartesian_fraction_threshold=0.0,
                                        #cartesian_jump_threshold=0.0,
                                        #cartesian_avoid_collisions=False,
                                        #planner_id=self.moveit2.planner_id,                                                       
                                    )
            ret_val = self.moveit2.wait_until_executed()
        finally:
            print ("move completed")
        return ret_val

    def MoveArmCartesian(self, position, quat_xyzw):
        """
        Moves the robot arm to a given position and orientation using Cartesian path planning.

        Args:
            position: Tuple or geometry_msgs.msg.Point (x, y, z)
            quat_xyzw: Tuple or geometry_msgs.msg.Quaternion (x, y, z, w)

        Returns:
            True if motion succeeded, False otherwise.
        """
        time.sleep(0.5)  # Allow time for arm/controller to become ready
        print("Starting Cartesian motion...")

        ret_val = False
        try:
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=quat_xyzw,
                cartesian=True,
                cartesian_max_step=0.0025,  # Step size in meters
                cartesian_fraction_threshold=0.9  # Minimum fraction of path to accept
            )
            ret_val = self.moveit2.wait_until_executed()
        finally:
            print("Cartesian motion completed")

        return ret_val




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
            path = "/home/alon/ros_ws/src/dialog_example/dialog_example/cal_poses.jsonc"
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

            # Also retrieve and save the robot position display
            pos_text = self.pos_text.get("1.0", tk.END).strip()

            # Save both to a file
            with open("copied_pose_log.txt", "a") as f:
                f.write("=== Aruco Pose ===\n")
                f.write(pose_text + "\n")
                f.write("=== Robot Position ===\n")
                f.write(pos_text + "\n\n")
                f.write("===Joint States: ===\n")
                f.write(joint_text + "\n\n")

        except Exception as e:
            self.get_logger().error(f"Error copying Aruco pose to clipboard: {e}")




    def joint_states_callback(self, msg):
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
                        if abs(new_pos - ref_pos) > 0.01:  # Tolerance in radians
                            self.get_logger().error(
                                f"Unintended motion detected on joint {i}: "
                                f"saved={ref_pos:.4f}, current={new_pos:.4f}, delta={abs(new_pos - ref_pos):.5f}"
                            )
                            sys.exit("Encoder error: unintended movement with zero velocity reported.")
            else:
                # If robot is moving, clear saved reference
                self.zero_velocity_positions = None

            # UI update
            joint_info = "\n".join([f"{pos:.4f}," for pos in msg.position])
            if self.last_joint_info == joint_info:
                return
            self.last_joint_info = joint_info

            self.joint_states_entry.configure(state='normal')
            self.joint_states_entry.delete("1.0", tk.END)
            self.joint_states_entry.insert(tk.END, joint_info)
            self.joint_states_entry.configure(state='disabled')

        except Exception as e:
            self.get_logger().error(f"Error in joint_states_callback: {e}")

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

    def call_service_blocking(self, client, request, timeout_sec=5.0):
        """
        Sends an async service request using the given client, and blocks until the response is received.
        The waiting is done in a background thread so it does not block the ROS 2 event loop.
        """
        self.call_is_done = False
        result_container = {'result': None, 'exception': None}

        def _thread_func():
            future = client.call_async(request)

            def _on_response(fut):
                try:
                    result_container['result'] = fut.result()
                except Exception as e:
                    result_container['exception'] = e
                finally:
                    self.call_is_done = True

            future.add_done_callback(_on_response)

        # Start background thread
        threading.Thread(target=_thread_func, daemon=True).start()

        # Wait until the call is done or timeout is reached
        start_time = time.time()
        while not self.call_is_done:
            self.spin_once()
            time.sleep(0.01)
            if time.time() - start_time > timeout_sec:
                raise TimeoutError("Service call timed out")

        if result_container['exception']:
            raise RuntimeError(f"Service call failed: {result_container['exception']}")

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
            new_text += (f"pos = ({x:.2f}, {y:.2f}, {z:.2f})"
                        f" ori = ({ox:.2f}, {oy:.2f}, {oz:.2f}, {ow:.2f})")
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

    def response_callback(self, future):
        try:
            response = future.result()
            # Logging the status and message from the response
            #self.get_logger().info(f'Response Status: {response.success}')
            #self.get_logger().info(f'Response Message: {response.message}')
            self.arm_is_available = True
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def wait_for_arm (self):
        """Wait for the arm to be available before sending a move request."""
        print ("waiting for arm to be available")
        while not self.arm_is_available:
            self.spin_once()
            time.sleep(0.01)  # Sleep for a short duration to avoid busy-waiting
            #print ("spinning")

    def aruco_pose_callback(self, msg):
        try:
            pose_text = f"({msg.x:.4f}, {msg.y:.4f}, {msg.z:.4f})"
            self.aruco_pose_entry.configure(state='normal')
            self.aruco_pose_entry.delete("1.0", tk.END)
            self.aruco_pose_entry.insert(tk.END, pose_text)
            self.aruco_pose_entry.configure(state='disabled')
        except Exception as e:
            self.get_logger().error(f"Error in aruco_pose_callback: {e}")

    def send_move_request(self, pose):
       
        #ret_val = self.MoveArm(pose.position, pose.orientation)
        ret_val = self.MoveArmCartesian(pose.position, pose.orientation)
        #time.sleep(1)
        return ret_val
        # Create a request
        request = MoveToPose.Request()
        request.pose = pose
        
        # Send the request
        self.arm_is_available = False
        future = self.move_client.call_async(request)
              
        # Add a callback to be executed when the future is complete
        future.add_done_callback(self.response_callback)
        self.wait_for_arm()
 
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
                self.send_move_request(pose_msg)
            except Exception as e:
                print(f"Error sending pose: {e}")
 
    def spin_until_future(self, future, timeout_sec=0.1):
        while not future.done():
            rclpy.spin_once(self, timeout_sec=timeout_sec)
        return future.result()

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
                num_valid_samples += 1
                #time.sleep(3)
                self.take_sample()
                #time.sleep(3)
                if num_valid_samples > 2:
                    self.compute_calibration()
                print(f"sample no. {num_valid_samples} taken - pose " + str(pose[0]) + " - " + str(pose[1])) 
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
        self.send_move_request(pose_msg)
    def on_send_pos_button_click(self):
        self.send_pose_from_entries()

    def trigger_homing_service(self, start_msg, success_msg, failure_msg):
        self.label.config(text=start_msg)
        request = Trigger.Request()
        future = self.homing_client.call_async(request)
        future.add_done_callback(lambda f: self.handle_service_response(f, success_msg, failure_msg))

    def handle_service_response(self, future, success_msg, failure_msg):
        try:
            response = future.result()
            if response.success:
                self.update_gui(success_msg)
            else:
                self.update_gui(failure_msg + f": {response.message}")
        except Exception as e:
            self.update_gui(f"Service call failed: {str(e)}")

    def update_gui(self, message):
        self.label.config(text=message)

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
    

def ros_spin(tkinter_ros):
    rclpy.spin(tkinter_ros)


class DummyNode(Node):
    def __init__(self):
        super().__init__('dummy_node')
        self.timer_called = False
        self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        print("Timer callback called.")
        self.timer_called = True


def main():


    debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # Ensures the debugger connects before continuing
    print("Debugger connected.")
   
    rclpy.init() 

    tkinter_ros = TkinterROS()

    # Start ROS spinning in a separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(tkinter_ros,))
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
