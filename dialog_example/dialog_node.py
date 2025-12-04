# ===== STANDARD LIBRARIES =====
import datetime
import json
import math
import os
import pathlib
import queue
import re
import threading
import time
from functools import partial

# ===== TKINTER =====
import tkinter as tk
from tkinter import ttk, messagebox

# ===== NUMPY / CV2 =====
import numpy as np
import cv2

# ===== ROS2 CORE =====
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# ===== ROS2 MSGS =====
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

# ===== ROS2 SRVS =====
from std_srvs.srv import Trigger, SetBool
from my_robot_interfaces.srv import (
    MoveToPose,
    NudgeJoint,
    MoveServoToAngle,
    SetSpeedScale,
    GetNextMove,
    HasWon,
    SetRotatedForbiddenBox
)
from geometry_msgs.msg import Quaternion


# ===== TF / TRANSFORMS =====
import tf_transformations
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

# ===== EASY-HANDEYE =====
import easy_handeye2 as hec
from easy_handeye2_msgs.srv import TakeSample, SaveCalibration, ComputeCalibration

# ===== CV BRIDGE =====
from cv_bridge import CvBridge

# ===== OTHER =====
import transforms3d
import yaml
import debugpy

STATE_NONE       = 0
STATE_PICK_CHIP_POS = 1
STATE_COLLECTED_CHIP = 2
STATE_DROPPED_CHIP = 3


class TkinterROS(Node):
    counter = 0

    def __init__(self):
        super().__init__('tkinter_ros_node')
        self.load_robot_params()

        self.gui_queue = queue.Queue()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.zero_velocity_positions = None
        self.num_valid_samples = 0
        self.pose_in_camera = None
        self.arm_is_moving = False
        self.init_cal_poses()
        self.initial_marker_pose_base = None
        self.editing_endpoint = None   # can be None, 1, or 2
        self.num_board_columns = 7
        self.top_dx_mm = None
        self.top_dy_mm = None
        self.editing_chip = False
        self.top_angle_deg = None
        self.top_est_height_mm = None
        self.active_drop_idx = None  # 1 or 2 after a goto
        self.last_joint_update_time = self.get_clock().now()
        self.marker_index = 0
        self.last_pos = ""
        self.current_state = STATE_NONE
        self.filtered_position = None
        self.ema_alpha = 0.2  # default filter constant
        self.calib_dir = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/calibrations'))
        self.calib_path = self.calib_dir / 'ar4_calibration.calib'
        # self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        self.cb_group = ReentrantCallbackGroup()
        self.wait_before_refine_sec = 0.5
        self.cur_chip_index = 0
        # KEEP a reference on self:

        self.brick_detections = []  # list of dicts for latest frame
        self.create_subscription(Float32MultiArray,'brick_top_infos',self.brick_top_infos_callback,10)
        self.aruco_sub = self.create_subscription(PoseArray,'/aruco_poses',self.aruco_pose_callback,qos_profile_sensor_data,callback_group=self.cb_group)
        self.board_end_points_sub = self.create_subscription(Float32MultiArray,'board_end_points',self.board_end_points_callback,qos_profile_sensor_data,callback_group=self.cb_group)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.create_subscription(Float32MultiArray,'brick_top_info',self.brick_top_info_callback,10)
        self.image_sub = self.create_subscription(Image,'/ov5640_rot_frame',self.image_callback,qos_profile_sensor_data)
        self.brick_info_sub = self.create_subscription(Float32MultiArray,'/brick_info_array',self.brick_info_callback,10)
        self.board_sub = self.create_subscription(Int32MultiArray,"/board/state",self.board_state_callback,10)

        self.chip_collect_positions = []  # will store absolute XY for the 4 chips

        self.set_speed_scale_client = self.create_client(SetSpeedScale,"/ar4_hardware_interface_node/set_speed_scale")

        self.has_won_client = self.create_client(HasWon, "has_won")

        while not self.has_won_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /has_won service...")
            
        self.set_speed_scale_srv_type = SetSpeedScale

        self._state_path = os.path.expanduser("~/.ov5640_board_state.json")
    
        self.board_dx1_mm = None
        self.board_dy1_mm = None
        self.board_dx2_mm = None
        self.board_dy2_mm = None

        self.latest_brick_center = None  # store camera-frame center
        self.latest_brick_yaw = None

        self.log_with_time('info', 'ArucoPoseFollower initialized, listening to /aruco_poses')
        self.total_columns = 7
        self.current_column_idx = -1  # will become 0 on first press

        self.open_gripper_client  = self.create_client(Trigger, '/ar4_hardware_interface_node/open_gripper')
        self.close_gripper_client = self.create_client(Trigger, '/ar4_hardware_interface_node/close_gripper')
        self.move_servo_client = self.create_client(MoveServoToAngle, '/ar4_hardware_interface_node/move_servo_to_angle')
        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')
        self.move_arm_client = self.create_client(MoveToPose, '/ar_move_to_pose')
        self.refresh_transform_client = self.create_client(Trigger, 'refresh_handeye_transform')
        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)
        self.aruco_follower_enabled_client = self.create_client(SetBool, '/set_aruco_follower_enabled')
        self.ai_client = self.create_client(GetNextMove, "get_next_move")
        self.has_won_client = self.create_client(HasWon, "has_won")
        self.board_box_client = self.create_client(SetRotatedForbiddenBox,'set_rotated_forbidden_box')

        self.safe_points_pub = self.create_publisher(
                    MarkerArray,
                    "safe_points_markers",
                    10
                )

        self.latest_chip = (-1, -1)  # row, col of latest detected chip

        # Optional: warn if not up yet (won’t block forever)
        for cli, nm in [(self.open_gripper_client,  'open_gripper'),
                        (self.close_gripper_client, 'close_gripper')]:
            if not cli.wait_for_service(timeout_sec=0.5):
                self.log_with_time('warn', f"Service '{nm}' not available yet")

        self.last_joint_info = ""
        self._last_aruco_update_time = 0
        # GUI init
        self.init_dialog()

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.log_with_time('info', 'registering ar_move_to_pose service')
        self.root.after(20, self.process_gui_queue) 
        self.log_with_time('info', 'take_sample service registered')
        self.init_right_frame()

        self.bridge = CvBridge()
        self.latest_frame = None

        self._load_board_pos_state()  # populate entries/vars if a state file exists

        # Add a timer to periodically show frames
        self.create_timer(0.03, self.display_frame)  # ~30 FPS
        self.latest_board = None
        self.total_chips = 0             # number of chips on the board
        self.wait_for = -1


    def publish_safe_points(self):
        safe_points = self.compute_and_publish_all_safe_points()

    def compute_and_publish_all_safe_points(self):
        
        safe_points = []
        for col in range(6):
            sp = self.compute_safe_column_pose(col)
            safe_points.append(sp)

        self.publish_safe_point_markers(safe_points)
        return safe_points


    def publish_safe_point_markers(self, safe_points):
        """
        Publish safe waypoint markers to RViz.
        safe_points = list of dicts with:
        {x, y, z, qx, qy, qz, qw}
        """

        marker_array = MarkerArray()
        header_frame = "base_link"

        for i, sp in enumerate(safe_points):

            # -----------------------------
            # Sphere marker (position only)
            # -----------------------------
            m = Marker()
            m.header.frame_id = header_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "safe_point"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.scale.x = 0.03
            m.scale.y = 0.03
            m.scale.z = 0.03

            m.pose.position.x = float(sp["x"])
            m.pose.position.y = float(sp["y"])
            m.pose.position.z = float(sp["z"])

            # Identity orientation for sphere
            m.pose.orientation.w = 1.0

            # Blue sphere
            m.color.r = 0.0
            m.color.g = 0.3
            m.color.b = 1.0
            m.color.a = 1.0

            marker_array.markers.append(m)

            # -----------------------------
            # Arrow marker (orientation)
            # -----------------------------
            arrow = Marker()
            arrow.header.frame_id = header_frame
            arrow.header.stamp = m.header.stamp
            arrow.ns = "safe_point_dir"
            arrow.id = 100 + i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.scale.x = 0.05  # shaft length
            arrow.scale.y = 0.01  # shaft diameter
            arrow.scale.z = 0.01  # head diameter

            arrow.pose.position.x = float(sp["x"])
            arrow.pose.position.y = float(sp["y"])
            arrow.pose.position.z = float(sp["z"])

            # Use the quaternion from compute_safe_column_pose
            arrow.pose.orientation.x = float(sp["qx"])
            arrow.pose.orientation.y = float(sp["qy"])
            arrow.pose.orientation.z = float(sp["qz"])
            arrow.pose.orientation.w = float(sp["qw"])

            # Green arrow
            arrow.color.r = 0.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0

            marker_array.markers.append(arrow)

        # Publish to RViz
        self.safe_points_pub.publish(marker_array)



    def send_rotated_board_box(self, depth=0.04):
        """
        Send the rotated board collision box to move_ar via service.
        Uses drop_p1_xy and drop_p2_xy as endpoints.
        """
        self.load_robot_params()
        if not hasattr(self, "drop_p1_xy") or not hasattr(self, "drop_p2_xy"):
            self.log_with_time("warn", "Board endpoints not defined; cannot send board box.")
            return

        p1x, p1y = self.drop_p1_xy
        p2x, p2y = self.drop_p2_xy

        height = getattr(self, "board_height", None)
        depth = self.board_width
        if height is None:
            self.log_with_time("warn", "board_height not set; cannot send board box.")
            return

        if not self.board_box_client.wait_for_service(timeout_sec=1.0):
            self.log_with_time("error", "set_rotated_board_box service not available.")
            return

        req = SetRotatedForbiddenBox.Request()
        req.p1_x = float(p1x)
        req.p1_y = float(p1y)
        req.p2_x = float(p2x)
        req.p2_y = float(p2y)
        req.height = float(height)
        req.depth = float(depth)

        future = self.board_box_client.call_async(req)

        def _done_cb(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.log_with_time("info", f"Board collision box updated: {resp.message}")
                else:
                    self.log_with_time("error", f"Failed to update board box: {resp.message}")
            except Exception as e:
                self.log_with_time("error", f"Exception in board box service call: {e}")

        future.add_done_callback(_done_cb)


    def check_win_condition(self):
        if self.latest_board is None:
            return

        flat = self.latest_board.flatten().tolist()

        # Check P1 and P2
        for player in [1, 2]:
            req = HasWon.Request()
            req.board = flat
            req.player = player

            future = self.has_won_client.call_async(req)

            # Bind the player value to callback
            future.add_done_callback(
                lambda f, p=player: self.after_has_won(f, p)
            )

    def move_to_current_chip_position(self):
        # Wrap index
        pos = self.chip_positions[self.cur_chip_index]
        cx, cy = pos

        self.log_with_time("info", f"Moving to chip #{self.cur_chip_index + 1}: ({cx:.3f}, {cy:.3f})")


        ok = self.move_to_xy_align_board(cx, cy, self.prepare_to_pick_z)
        if not ok:
            self.log_with_time("error", "Failed to move to chip position.")
            return
        #refresh latest bricks
        self.latest_bricks = []
        time.sleep(self.wait_before_refine_sec)

    def on_set_box(self):
        self.send_rotated_board_box()
        
    def on_next_chip_position(self):
        """Move the robot to the next chip position in the list."""
        if not self.chip_positions or len(self.chip_positions) == 0:
            self.log_with_time("warn", "No chip positions available.")
            return


        # Advance index
        self.cur_chip_index = (self.cur_chip_index + 1) % len(self.chip_positions)

        self.move_to_current_chip_position()

        self.editing_chip = True
        self.editing_endpoint = None


    def after_has_won(self, future, player):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"has_won service failed: {e}")
            return

        if result.has_won:
            msg = f"🎉 Player {player} has WON!"
            self.get_logger().info(msg)

            # Send to GUI main thread if needed
            if hasattr(self, "gui_queue"):
                self.gui_queue.put(lambda: self.handle_win(player))
            else:
                self.handle_win(player)

    def load_robot_params(self):
        """
        Load all robot parameters safely from robot_params.json.
        Includes chip_pick, gripper, chip_start, board, identification,
        chip_positions, and NEW: start_position.
        """

        # =====================================================
        # DEFAULTS (used if JSON missing or partial)
        # =====================================================
        self.lift_z = 0.35
        self.hover_z = 0.30
        self.pick_z1 = 0.24
        self.pick_z2 = 0.17
        self.pick_z3 = 0.127
        self.prepare_to_pick_z = 0.21
        self.drop_xy = (0.15, -0.30)

        self.GRIP_FORCE_OPEN  = 35
        self.GRIP_FORCE_CLOSE = 90

        # Chip-start (your “start of chip collection” pose)
        self.chip_start_x = 0.23
        self.chip_start_y = -0.30
        self.chip_start_z = 0.35

        # Height for board placement
        self.board_column_z = 0.28
        self.board_height = 0.02
        self.board_width = 0.04
        self.board_hover_z = 0.3
        # Identification pose
        self.chip_ident_x = 0.353
        self.chip_ident_y = -0.388
        self.chip_ident_z = 0.349

        # NEW: General “start” position for whole sequence
        self.start_x = 0.116
        self.start_y = -0.344
        self.start_z = 0.401

        # =====================================================
        # Open JSON
        # =====================================================
        folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(folder, "robot_params.json")

        if not os.path.exists(json_path):
            self.get_logger().warn(f"No robot_params.json found at {json_path}, using defaults")
            return

        with open(json_path, "r") as f:
            params = json.load(f)

        # =====================================================
        # A. Chip picking parameters
        # =====================================================
        chip_pick = params.get("chip_pick", {})
        self.lift_z  = chip_pick.get("lift_z",  self.lift_z)
        self.hover_z = chip_pick.get("hover_z", self.hover_z)
        self.pick_z1 = chip_pick.get("pick_z1", self.pick_z1)
        self.pick_z2 = chip_pick.get("pick_z2", self.pick_z2)
        self.pick_z3 = chip_pick.get("pick_z3", self.pick_z3)
        self.prepare_to_pick_z = chip_pick.get("prepare_to_pick_z", self.prepare_to_pick_z)
        self.drop_xy = tuple(chip_pick.get("drop_xy", self.drop_xy))

        # =====================================================
        # B. Gripper parameters
        # =====================================================
        grip = params.get("gripper", {})
        self.GRIP_FORCE_OPEN  = grip.get("GRIP_FORCE_OPEN", self.GRIP_FORCE_OPEN)
        self.GRIP_FORCE_CLOSE = grip.get("GRIP_FORCE_CLOSE", self.GRIP_FORCE_CLOSE)

        # =====================================================
        # C. Chip-start pose
        # =====================================================
        chip_start = params.get("chip_start", {})
        self.chip_start_x = chip_start.get("x", self.chip_start_x)
        self.chip_start_y = chip_start.get("y", self.chip_start_y)
        self.chip_start_z = chip_start.get("z", self.chip_start_z)

        # =====================================================
        # D. Board parameters
        # =====================================================
        board = params.get("board", {})
        self.board_column_z = board.get("column_z", self.board_column_z)
        self.board_height = board.get("board_height", self.board_height)
        self.board_width = board.get("board_width", self.board_width)
        self.board_hover_z = board.get("board_hover_z", self.board_hover_z)

        # =====================================================
        # E. Chip-table identification pose
        # =====================================================
        chip_ident = params.get("chip_table_identification", {})
        self.chip_ident_x = chip_ident.get("x", self.chip_ident_x)
        self.chip_ident_y = chip_ident.get("y", self.chip_ident_y)
        self.chip_ident_z = chip_ident.get("z", self.chip_ident_z)

        # =====================================================
        # F. NEW: General “start” position
        # =====================================================
        start = params.get("start_position", {})
        self.start_x = start.get("x", self.start_x)
        self.start_y = start.get("y", self.start_y)
        self.start_z = start.get("z", self.start_z)

        # =====================================================
        # G. Chip positions array
        # =====================================================
        self.chip_positions = self.load_chip_positions_from_json(json_path)
        

        self.get_logger().info(f"Loaded robot parameters from {json_path}")

    def load_chip_positions_from_json(self,json_path):
        """Load chip positions from JSON, return list of (x,y)."""
        if not os.path.exists(json_path):
            return []

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            return []

        block = data.get("chip_table_identification", {})
        raw_list = block.get("chip_positions", [])

        positions = []
        for item in raw_list:
            try:
                x = float(item.get("x", 0.0))
                y = float(item.get("y", 0.0))
                positions.append((round(x, 3), round(y, 3)))
            except Exception:
                continue

        return positions


    def save_chip_positions_to_json(self):
        """Save chip_positions (flexible length) into robot_params.json."""

        folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(folder, "robot_params.json")

        data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        # Ensure category exists
        if "chip_table_identification" not in data:
            data["chip_table_identification"] = {}

        # Store array as {"positions": [{"x":..,"y":..}, ...]}
        data["chip_table_identification"]["chip_positions"] = [
            {"x": round(p[0], 3), "y": round(p[1], 3)}
            for p in self.chip_positions
        ]

        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            self.log_with_time("info", f"Saved chip positions ({len(self.chip_positions)}) to JSON.")
        except Exception as e:
            self.log_with_time("error", f"Failed to save chip positions: {e}")


    def nudge_ee_yaw(self, delta_deg: float, is_cartesian: bool = False) -> bool:
        """
        Nudges the End-Effector (EE) orientation (Joint 6 / Yaw) by a small delta amount
        while maintaining the current EE position.

        :param delta_deg: The change in yaw angle in DEGREES. Positive is CCW, negative is CW.
        :param is_cartesian: Whether to use Cartesian path planning for execution.
        :return: True if the motion request was successful, False otherwise.
        """
        try:
            # --- Parameter Conversion ---
            # Convert the input degrees to radians for internal calculation
            import numpy as np
            delta_rad = np.radians(delta_deg)

            # ------------------------------
            # 1) Current EE Pose
            # ------------------------------
            pose = self.get_current_ee_pose()
            if pose is None:
                self.log_with_time('error', "Cannot read current EE pose.")
                return False

            # ------------------------------
            # 2) Extract Current RPY
            # ------------------------------
            q_curr = [
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w
            ]
            # Extracts Roll (R), Pitch (P), and Yaw (Y) from the current quaternion
            # Assumes tf_transformations and _wrap_to_pi are available within the class scope
            roll_curr, pitch_curr, yaw_curr = tf_transformations.euler_from_quaternion(q_curr)
            yaw_curr = self._wrap_to_pi(yaw_curr) # Normalize to [-pi, pi]

            self.log_with_time(
                'info',
                f"[NUDGE] Current EE RPY = "
                f"({np.degrees(roll_curr):.2f}°, {np.degrees(pitch_curr):.2f}°, {np.degrees(yaw_curr):.2f}°)"
            )

            # ------------------------------
            # 3) Calculate Target Yaw
            # ------------------------------
            yaw_target = self._wrap_to_pi(yaw_curr + delta_rad)

            self.log_with_time(
                'info',
                f"[NUDGE] Target yaw = {np.degrees(yaw_target):.2f}° "
                f"(Delta input: {delta_deg:.2f}°)"
            )

            # ------------------------------
            # 4) Build Target Pose
            # ------------------------------
            from geometry_msgs.msg import Pose
            
            target = Pose()
            # Keep the current position (x, y, z)
            target.position = pose.position

            # Keep current Roll and Pitch, apply new Yaw
            qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
                roll_curr, pitch_curr, yaw_target
            )

            target.orientation.x = qx
            target.orientation.y = qy
            target.orientation.z = qz
            target.orientation.w = qw

            self.log_with_time(
                'info',
                f"[MOVE] Nudging EE Yaw to new orientation at current position "
                f"({target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f})"
            )

            # ------------------------------
            # 5) Execute motion
            # ------------------------------
            ok = self.send_move_request(
                target,
                is_cartesian=is_cartesian,
                via_points=[] # Nudges are typically direct moves
            )

            self.log_with_time(
                'info',
                f"[MOVE] Final motion result: {ok}"
            )

            return ok

        except Exception as e:
            self.log_with_time('error', f"Error in nudge_ee_yaw: {e}")
            return False

    def board_state_callback(self, msg):
        arr = np.array(msg.data, dtype=np.int32)

        if arr.size != 42:
            self.get_logger().warn(f"Invalid board size: %d" % arr.size)
            return

        # Save previous board before updating
        prev = self.latest_board.copy() if self.latest_board is not None else None

        # Reshape, flip vertically + horizontally
        current = arr.reshape((6, 7))[::-1, ::-1]

        self.latest_board = current

        # Count chips
        self.total_chips_old = self.total_chips
        self.total_chips = int(np.count_nonzero(current))

        # Detect new chip
        if prev is not None and self.total_chips > self.total_chips_old:
            # Find cell that changed 0 → (1 or 2)
            diff = (prev == 0) & (current != 0)
            positions = np.argwhere(diff)

            if positions.size > 0:
                r, c = positions[0]   # There should be exactly one
                self.latest_chip = (int(r), int(c))
                self.get_logger().info(f"New chip at row={r}, col={c}")

        # Log update
        if self.total_chips != self.total_chips_old:
            self.log_with_time("info", f"Board updated. Chips: {self.total_chips}")
            self.log_with_time("info", f"latest_chip: {self.latest_chip}")


    def _load_board_pos_state(self):
        """Load step and endpoints from JSON and reflect in UI + memory."""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r") as f:
                data = json.load(f)

            # step
            step = data.get("step_m")
            if isinstance(step, (int, float)) and self.board_step_entry:
                self.board_step_entry.delete(0, tk.END)
                self.board_step_entry.insert(0, f"{step:.3f}")

            # endpoints
            p1 = data.get("drop_p1_xy")
            p2 = data.get("drop_p2_xy")

            if isinstance(p1, (list, tuple)) and len(p1) == 2:
                self.drop_p1_xy = (float(p1[0]), float(p1[1]))
                self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)

            if isinstance(p2, (list, tuple)) and len(p2) == 2:
                self.drop_p2_xy = (float(p2[0]), float(p2[1]))
                self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

            # keep direction vector in sync when both exist
            self._recompute_board_dir_xy()

            self.log_with_time('info', f"Loaded board state from {self._state_path}")
        except Exception as e:
            self.log_with_time('warn', f"Could not load board state: {e}")

    def set_speed_scale(self, scale: float):
        client = self.set_speed_scale_client

        if client is None:
            self.log_with_time('error', "SpeedScale client not initialized")
            return
        
        try:
            if not client.wait_for_service(timeout_sec=0.5):
                self.log_with_time('warn', "SetSpeedScale service not available")
                return
        except Exception as e:
            self.log_with_time('error', f"Waiting for SetSpeedScale service failed: {e}")
            return

        req = self.set_speed_scale_srv_type.Request()
        req.scale = float(scale)

        self.log_with_time('info', f"Sending speed scale: {req.scale}")

        future = client.call_async(req)

        def _on_done(fut):
            try:
                resp = fut.result()
                if getattr(resp, "success", False):
                    self.log_with_time('info', f"Speed scale set to {req.scale}")
                else:
                    msg = getattr(resp, "message", "(no message)")
                    self.log_with_time('warn', f"SpeedScale failed: {msg}")
            except Exception as e:
                self.log_with_time('error', f"SpeedScale call exception: {e}")

        future.add_done_callback(_on_done)
        time.sleep(0.1)  # slight delay to avoid overwhelming the service
    def _save_board_state(self):
        """Save step and endpoints to JSON."""
        try:
            # read step directly from entry (fallback to 0.01 if bad)
            try:
                step_m = float(self.board_step_entry.get().strip())
            except Exception:
                step_m = 0.01

            data = {
                "step_m": step_m,
                "drop_p1_xy": list(self.drop_p1_xy) if getattr(self, "drop_p1_xy", None) else None,
                "drop_p2_xy": list(self.drop_p2_xy) if getattr(self, "drop_p2_xy", None) else None,
                "saved_at": time.time(),
            }
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
            # optional: log quietly to avoid spam
            # self.log_with_time('info', f"Saved board state to {self._state_path}")
        except Exception as e:
            self.log_with_time('warn', f"Could not save board state: {e}")



    def compute_safe_column_pose(self, col_index):
        """
        Compute a safe waypoint for a given Connect-4 column.
        The waypoint is offset perpendicular to the board, with the gripper facing downward.
        """
        p1_xy, p2_xy = self.drop_p1_xy, self.drop_p2_xy
        offset = 0.05
        z_height = self.board_hover_z

        if not (0 <= col_index <= 5):
            raise ValueError("col_index must be between 0 and 5")

        x1, y1 = p1_xy
        x2, y2 = p2_xy

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            raise ValueError("p1_xy and p2_xy cannot be the same point")

        # Unit direction along board axis
        ux = dx / length
        uy = dy / length

        # Perpendicular left vector
        nx = -uy
        ny = ux

        # Column center
        spacing = length / 5.0
        col_center_x = x1 + ux * spacing * col_index
        col_center_y = y1 + uy * spacing * col_index

        # Safe point offset to the side
        safe_x = col_center_x + nx * offset
        safe_y = col_center_y + ny * offset
        safe_z = z_height

        # Yaw aligning gripper parallel to board axis
        yaw = math.atan2(dy, dx)

        # Facing down: roll=-pi, pitch=0
        roll = -math.pi
        pitch = 0.0

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw)

        return {
            "x": safe_x,
            "y": safe_y,
            "z": safe_z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw
        }



    def compute_board_column_xy(self, col_idx: int):
        """
        Return (x,y) of Connect-4 column center for col_idx in [0..6], using stored endpoints.
        Convention: col 0 is at drop_p1_xy, col 6 is at drop_p2_xy, and centers are linearly spaced.
        """
        if not (0 <= col_idx <= 6):
            raise ValueError("col_idx must be in [0..6]")

        p1 = getattr(self, "drop_p1_xy", None)
        p2 = getattr(self, "drop_p2_xy", None)
        if p1 is None or p2 is None:
            self.log_with_time('warn', "Board endpoints not set; compute drop points first.")
            return None

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d  = p2 - p1
        n  = np.linalg.norm(d)
        if n < 1e-9:
            self.log_with_time('warn', "Board endpoints are identical; cannot compute columns.")
            return None

        t = col_idx / 6.0  # 0..1 across the board
        p = p1 + t * d
        return float(p[0]), float(p[1])


    def _wrap_to_pi(self, a: float) -> float:
        return (a + np.pi) % (2*np.pi) - np.pi

    def move_to_xy_align_board(self, x: float, y: float, z: float = None,
                            is_cartesian: bool = False,
                            via_points: list = None):
        """
        Move to (x, y, z=current if None) with orientation guaranteed parallel to ground:
        roll = -pi (face-down), pitch = 0, yaw = ⟂ to board.
        Optionally accepts via-points (list of Pose) which are forwarded to send_move_request().
        """

        if via_points is None:
            via_points = []

        # ------------------------------
        # 1) Compute board yaw
        # ------------------------------
        yaw_board = self._compute_board_yaw_rad()
        if yaw_board is None:
            return False

        self.log_with_time(
            'info',
            f"[YAW] Board yaw = {np.degrees(yaw_board):.2f}°"
        )

        # ------------------------------
        # 2) Compute perpendicular yaw
        # ------------------------------
        yaw_perp = self._wrap_to_pi(yaw_board + np.pi/2)
        self.log_with_time(
            'info',
            f"[YAW] Perpendicular yaw = {np.degrees(yaw_perp):.2f}°"
        )

        # ------------------------------
        # 3) Current EE yaw
        # ------------------------------
        pose = self.get_current_ee_pose()
        if pose is None:
            self.log_with_time('error', "Cannot read current EE pose.")
            return False

        q_curr = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ]
        _, _, yaw_curr = tf_transformations.euler_from_quaternion(q_curr)
        yaw_curr = self._wrap_to_pi(yaw_curr)

        self.log_with_time(
            'info',
            f"[YAW] Current EE yaw = {np.degrees(yaw_curr):.2f}°"
        )

        # ------------------------------
        # 4) Evaluate 3 yaw candidates
        # ------------------------------
        cands = [
            yaw_perp,
            self._wrap_to_pi(yaw_perp + np.pi),
            self._wrap_to_pi(yaw_perp - np.pi),
        ]
        self.log_with_time(
            'info',
            f"[YAW] Candidates = "
            f"{[f'{np.degrees(c):.2f}°' for c in cands]}"
        )

        # Choose closest candidate
        yaw_target = min(cands, key=lambda a: abs(self._wrap_to_pi(a - yaw_curr)))

        self.log_with_time(
            'info',
            f"[YAW] Selected target yaw = {np.degrees(yaw_target):.2f}°"
        )

        # ------------------------------
        # 5) Build target pose
        # ------------------------------
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = (
            float(pose.position.z) if z is None else float(z)
        )

        roll = -np.pi
        pitch = 0.0
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            roll, pitch, yaw_target
        )

        target.orientation.x = qx
        target.orientation.y = qy
        target.orientation.z = qz
        target.orientation.w = qw

        self.log_with_time(
            'info',
            f"[MOVE] Moving to ({x:.3f}, {y:.3f}, {target.position.z:.3f}) "
            f"with yaw_target={np.degrees(yaw_target):.2f}°"
        )

        # ------------------------------
        # 6) Execute motion (with via points)
        # ------------------------------
        ok = self.send_move_request(
            target,
            is_cartesian=is_cartesian,
            via_points=via_points
        )

        self.log_with_time(
            'info',
            f"[MOVE] Final motion result: {ok}"
        )

        # ------------------------------
        # 7) Post-move verification
        # ------------------------------
        if ok:
            final = self.get_current_ee_pose()
            if final:
                qf = [
                    final.orientation.x, final.orientation.y,
                    final.orientation.z, final.orientation.w
                ]
                rf, pf, yf = tf_transformations.euler_from_quaternion(qf)
                yf = self._wrap_to_pi(yf)

                self.log_with_time(
                    'info',
                    f"[POST] Final EE yaw = {np.degrees(yf):.2f}° "
                    f"(roll={np.degrees(rf):.2f}°, pitch={np.degrees(pf):.2f}°)"
                )

                # Snap correction if needed
                if abs(rf + np.pi) > np.degrees(2e-2) or abs(pf) > np.degrees(2e-2):
                    self.log_with_time(
                        'warn',
                        f"[POST] Orientation drift detected → correcting. "
                        f"rf={np.degrees(rf):.3f}°, pf={np.degrees(pf):.3f}°"
                    )

                    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
                        -np.pi, 0.0, yf
                    )
                    target2 = Pose()
                    target2.position = final.position
                    target2.orientation.x = qx
                    target2.orientation.y = qy
                    target2.orientation.z = qz
                    target2.orientation.w = qw

                    self.send_move_request(target2, is_cartesian=is_cartesian)

                    self.log_with_time(
                        'info',
                        f"[POST] Correction applied. New yaw={np.degrees(yf):.2f}°"
                    )

        return ok

    def _compute_board_yaw_rad(self):
        """
        Yaw (radians) parallel to the board, computed from drop_p1_xy -> drop_p2_xy.
        Returns None if endpoints are missing/degenerate.
        """
        p1 = getattr(self, "drop_p1_xy", None)
        p2 = getattr(self, "drop_p2_xy", None)
        if p1 is None or p2 is None:
            self.log_with_time('warn', "Board endpoints not set; cannot compute board yaw.")
            return None

        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        n  = (dx*dx + dy*dy) ** 0.5
        if n < 1e-9:
            self.log_with_time('warn', "Board endpoints identical; cannot compute board yaw.")
            return None
        return np.arctan2(dy, dx)  # radians

    def on_next_board_column(self):
        self.current_column_idx = (self.current_column_idx + 1) % self.total_columns
        self.goto_board_column(self.current_column_idx)


    def compute_board_alignment_orientation(self):
        """
        Returns a quaternion that aligns the tool flat (roll=-pi, pitch=0)
        and yaw perpendicular to the board (based on board yaw).
        """
        yaw_board = self._compute_board_yaw_rad()
        if yaw_board is None:
            self.log_with_time('error', "Cannot compute board yaw for orientation.")
            return Quaternion()

        # Desired yaw is perpendicular to board
        yaw_perp = self._wrap_to_pi(yaw_board + np.pi/2)

        # Standard face-down orientation: roll=-pi, pitch=0
        roll = -np.pi
        pitch = 0.0

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            roll, pitch, yaw_perp
        )

        q = Quaternion()
        q.x = qx
        q.y = qy
        q.z = qz
        q.w = qw
        return q


    def goto_board_column(self, col_idx: int):
        """
        Move EE to the center of a board column with yaw parallel to board.
        If current EE height is below the board height, insert a safe via-point.
        """
        self.load_robot_params()

        xy = self.compute_board_column_xy(col_idx)
        if xy is None:
            return

        x, y = xy
        z = self.board_hover_z
        board_h = self.board_height

        self.log_with_time(
            'info',
            f"Going to board column {col_idx}: ({x:.3f}, {y:.3f}) (yaw aligned)"
        )

        # Refresh the rotated board collision box
        self.send_rotated_board_box()

        # ------------------------------------------------------------
        # Read current EE Z
        # ------------------------------------------------------------
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "link_6", rclpy.time.Time()
            )
            current_z = tf.transform.translation.z
        except Exception as e:
            self.log_with_time('warn', f"TF lookup failed ({e}), skipping via point logic.")
            current_z = 999.0

        # ------------------------------------------------------------
        # Compute face-down quaternion (used at final pose if needed)
        # ------------------------------------------------------------
        orientation_face_down = self.compute_board_alignment_orientation()

        # ------------------------------------------------------------
        # Build via points
        # ------------------------------------------------------------
        via_points = []

        if current_z < board_h:
            safe = self.compute_safe_column_pose(col_idx)

            via = Pose()
            via.position.x = safe["x"]
            via.position.y = safe["y"]
            via.position.z = safe["z"]

            # Use the safe orientation (already roll=-pi, pitch=0, yaw aligned)
            via.orientation.x = safe["qx"]
            via.orientation.y = safe["qy"]
            via.orientation.z = safe["qz"]
            via.orientation.w = safe["qw"]

            via_points.append(via)

            self.log_with_time(
                'info',
                f"Current Z = {current_z:.3f} < board height {board_h:.3f}, "
                f"added safe via-point at ({via.position.x:.3f}, "
                f"{via.position.y:.3f}, {via.position.z:.3f})"
            )

        # ------------------------------------------------------------
        # Move to final column XY, with optional via points
        # ------------------------------------------------------------
        self.move_to_xy_align_board(
            x, y, z,
            is_cartesian=False,
            via_points=via_points
        )

    def board_end_points_callback(self, msg: Float32MultiArray):
        """
        Callback for receiving board endpoints [dx1_mm, dy1_mm, dx2_mm, dy2_mm].
        These represent the 2 end points of the Connect-4 board relative to the ArUco marker.
        """
        data = list(msg.data)
        if len(data) != 5:
            self.log_with_time('warn', f"board_end_points message unexpected length: {len(data)}")
            return
        type = 2
        type, self.board_dx1_mm, self.board_dy1_mm, self.board_dx2_mm, self.board_dy2_mm = data

        # self.log_with_time(
        #     'info',
        #     f"Board endpoints received: "
        #     f"P1=({self.board_dx1_mm:.1f}, {self.board_dy1_mm:.1f}) mm, "
        #     f"P2=({self.board_dx2_mm:.1f}, {self.board_dy2_mm:.1f}) mm"
        # )


    def get_joint_angle(self, joint_name: str):
        """Return the angle (in degrees) for a given joint name, or None if not found."""
        if not self.joint_states:
            return None
        try:
            idx = self.joint_states.name.index(joint_name)
            return np.degrees(self.joint_states.position[idx])
        except ValueError:
            self.get_logger().warn(f"{joint_name} not found in joint states")
            return None
    
    def image_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def display_frame(self):
        if self.latest_frame is None:
            return
        cv2.imshow("Processed Frame (from /processed_frame)", self.latest_frame)
        key = cv2.waitKey(0) & 0xFF

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


    def _fmt_xy(self, p):
        return f"{p[0]:.3f}, {p[1]:.3f}"

    def _parse_xy_entry(self, entry: tk.Entry):
        txt = entry.get().strip().replace(",", " ")
        parts = [t for t in txt.split() if t]
        if len(parts) < 2:
            raise ValueError("XY entry must be 'x, y' or 'x y'")
        return (float(parts[0]), float(parts[1]))

    def _set_xy_entry(self, entry, xy):
        def update():
            entry.delete(0, tk.END)
            entry.insert(0, f"{xy[0]:.4f}, {xy[1]:.4f}")
        self.gui_queue.put(update)


    def _get_board_dir_xy_from_marker(self):
        """Fallback: board direction from marker +Y axis (green), projected to XY."""
        pose_b = self.get_marker_pose_in_base()
        q = [pose_b.orientation.x, pose_b.orientation.y, pose_b.orientation.z, pose_b.orientation.w]
        R = tf_transformations.quaternion_matrix(q)[0:3, 0:3]
        y_axis = R[:, 1]  # marker +Y (green)
        dir_xy = np.array([y_axis[0], y_axis[1]])
        n = np.linalg.norm(dir_xy)
        if n < 1e-6:
            return np.array([1.0, 0.0])
        return dir_xy / n

    def _recompute_board_dir_xy(self):
        """
        Define board direction from the two stored endpoints (drop_p1_xy, drop_p2_xy).
        No fallback to marker; if unavailable/degenerate, keep previous direction.
        """
        p1 = getattr(self, "drop_p1_xy", None)
        p2 = getattr(self, "drop_p2_xy", None)

        if not p1 or not p2:
            self.log_with_time('warn', "Board endpoints not set yet; board_dir_xy unchanged.")
            return

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d = p2 - p1
        n = np.linalg.norm(d)

        if n < 1e-6:
            self.log_with_time('warn', "Board endpoints are identical; board_dir_xy unchanged.")
            return

        u = d / n
        # store as ndarray or tuple—pick what you use elsewhere; ndarray is handy for math
        self.board_dir_xy = u
        # (Optional) also store the perpendicular if you use it:
        # self.board_perp_xy = np.array([-u[1], u[0]], dtype=float)


    def _board_normal_xy(self):
        """Perpendicular (left/right) = rotate board_dir by +90° in XY."""
        t = getattr(self, "board_dir_xy", np.array([1.0, 0.0]))
        return np.array([-t[1], t[0]])

    def move_relative_xy(self, dx: float, dy: float):
        """Move EE by a small offset in XY, preserving Z and orientation."""
        cur = self.get_current_ee_pose()
        if cur is None:
            self.log_with_time('error', "Current EE pose unavailable.")
            return
        pose = self.create_pose(
            (cur.position.x + float(dx), cur.position.y + float(dy), cur.position.z),
            (cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w)
        )
        self.send_move_request(pose, is_cartesian=False)

    def move_to_board_end(self, which="left"):
        if None in (self.board_dx1_mm, self.board_dy1_mm, self.board_dx2_mm, self.board_dy2_mm):
            self.log_with_time('warn', "Board endpoints not yet received.")
            return
        dx_mm, dy_mm = (self.board_dx1_mm, self.board_dy1_mm) if which == "left" else (self.board_dx2_mm, self.board_dy2_mm)
        dx_m, dy_m = -dx_mm / 1000.0, -dy_mm / 1000.0
        self.move_relative_xy(dx_m, dy_m)

    def get_marker_pose_in_base(self) -> Pose:
        """
        Return the latest marker pose expressed in base_link.
        Raises RuntimeError if unavailable.
        """
        if self.pose_in_camera is None:
            raise RuntimeError("No ArUco pose from camera yet (self.pose_in_camera is None).")

        return self._transform_pose(
            self.pose_in_camera,
            source_frame="camera_color_optical_frame",
            target_frame="base_link"
        )

    def correct_marker_pose_for_joints(self,pose_b: Pose, j0_angle: float, j5_angle: float) -> Pose:
        """
        Rotates the marker pose around base Z by (J0 + J5) radians.
        Use when camera_link is fixed to base_link instead of being downstream of the wrist.
        """

        # 1) Build rotation matrix about Z
        total_angle = j0_angle + j5_angle
        R_corr = tf_transformations.rotation_matrix(total_angle, (0, 0, 1))[0:3, 0:3]

        # 2) Pose → matrix
        q = [pose_b.orientation.x, pose_b.orientation.y, pose_b.orientation.z, pose_b.orientation.w]
        R_pose = tf_transformations.quaternion_matrix(q)[0:3, 0:3]
        p = np.array([pose_b.position.x, pose_b.position.y, pose_b.position.z])

        # 3) Apply rotation about base Z to both R and p
        p_corr = R_corr.dot(p)
        R_corr_total = R_corr.dot(R_pose)

        # 4) Back to Pose
        q_corr = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([R_corr_total, np.zeros((3,1))]), [0,0,0,1]])
        )
        pose_out = Pose()
        pose_out.position.x, pose_out.position.y, pose_out.position.z = p_corr
        pose_out.orientation.x, pose_out.orientation.y, pose_out.orientation.z, pose_out.orientation.w = q_corr
        return pose_out

    def compute_board_edges_from_marker(
        self,
        col_spacing_m: float = 0.040,     # center-to-center spacing
        num_cols: int = 7,
        axis_along_columns: str = "x",    # marker axis that runs along the board columns
        axis_sign: int = +1,
        board_clearance_m: float = 0.030, # from marker center toward the board face
        axis_board_normal: str = "y",     # marker axis that points from marker toward the board
        normal_sign: int = +1,
        z_mode: str = "keep_ee"           # "keep_ee" | "marker_z" | float(z_value)
    ):
        """
        Returns two edge points (left, right) in base_link.
        All stepping is done in the XY plane so image/camera tilt does not distort XY.
        """

        j0_angle = self.get_joint_angle("joint_1") or 0.0
        j5_angle = self.get_joint_angle("joint_6") or 0.0

        pose_b = self.get_marker_pose_in_base()
        pose_b = self.correct_marker_pose_for_joints(pose_b, j0_angle, j5_angle)

        # Rotation: columns are columns of R (marker axes in base)
        q = [pose_b.orientation.x, pose_b.orientation.y, pose_b.orientation.z, pose_b.orientation.w]
        R = tf_transformations.quaternion_matrix(q)[0:3, 0:3]

        # Helpers: project a 3D vector to XY and normalize there
        def proj_xy(v):
            vx, vy = float(v[0]), float(v[1])
            n = (vx*vx + vy*vy) ** 0.5
            if n < 1e-12:
                return np.array([1.0, 0.0])  # fallback
            return np.array([vx/n, vy/n])

        # Column direction (in XY)
        idx = {"x": 0, "y": 1, "z": 2}[axis_along_columns.lower()]
        col_dir_xy = proj_xy(axis_sign * R[:, idx])

        # Normal toward the board (in XY) – just to nudge from the marker to the board face
        nidx = {"x": 0, "y": 1, "z": 2}[axis_board_normal.lower()]
        n_dir_xy = proj_xy(normal_sign * R[:, nidx])

        # Center on board face (XY only)
        c = np.array([pose_b.position.x, pose_b.position.y], dtype=float)
        center_xy = c + board_clearance_m * n_dir_xy

        # Half span in XY
        half_span = ((num_cols - 1) * col_spacing_m) / 2.0
        left_xy  = center_xy - half_span * col_dir_xy   # col 0
        right_xy = center_xy + half_span * col_dir_xy   # col N-1

        # Choose Z
        if z_mode == "keep_ee":
            ee = self.get_current_ee_pose()
            z_val = float(ee.position.z) if ee is not None else float(pose_b.position.z)
        elif z_mode == "marker_z":
            z_val = float(pose_b.position.z)
        else:
            # z_mode is a numeric value
            try:
                z_val = float(z_mode)
            except Exception:
                z_val = float(pose_b.position.z)

        edge_left  = Point(x=float(left_xy[0]),  y=float(left_xy[1]),  z=z_val)
        edge_right = Point(x=float(right_xy[0]), y=float(right_xy[1]), z=z_val)
        return edge_left, edge_right

    def on_compute_drop_points(self):
        """
        Compute and save board endpoints into the GUI fields **only** from the
        `board_end_points` topic values. No marker fallback.

        Expects the subscriber to have populated:
        self.board_dx1_mm, self.board_dy1_mm,
        self.board_dx2_mm, self.board_dy2_mm
        """
        try:
            # 1) Ensure we have topic data
            if any(v is None for v in (
                getattr(self, 'board_dx1_mm', None),
                getattr(self, 'board_dy1_mm', None),
                getattr(self, 'board_dx2_mm', None),
                getattr(self, 'board_dy2_mm', None),
            )):
                self.log_with_time('warn', "No board_end_points topic data yet. Press again after a message arrives.")
                return

            # 2) Need current EE pose for absolute XY in base_link
            ee = self.get_current_ee_pose()
            if ee is None:
                self.log_with_time('error', "Cannot read current EE pose.")
                return

            # 3) Convert mm→m using SAME sign convention you use when moving by dx/dy:
            #    move_to_board_end() uses (-dx_mm, -dy_mm) in base XY.
            p1_xy = (float(ee.position.x) - float(self.board_dx1_mm)/1000.0,
                    float(ee.position.y) - float(self.board_dy1_mm)/1000.0)
            p2_xy = (float(ee.position.x) - float(self.board_dx2_mm)/1000.0,
                    float(ee.position.y) - float(self.board_dy2_mm)/1000.0)

            # 4) Save to state & GUI (same fields your UI already uses)
            self.drop_p1_xy = p1_xy
            self.drop_p2_xy = p2_xy
            self._set_xy_entry(self.drop_p1_entry, p1_xy)
            self._set_xy_entry(self.drop_p2_entry, p2_xy)

            # 5) Recompute direction used by the nudge buttons
            self._recompute_board_dir_xy()

            self.log_with_time(
                'info',
                f"Board endpoints (from topic): "
                f"P1=({p1_xy[0]:.3f},{p1_xy[1]:.3f}), "
                f"P2=({p2_xy[0]:.3f},{p2_xy[1]:.3f})"
            )

        except Exception as e:
            self.log_with_time('error', f"compute_drop_points (topic) failed: {e}")

        self._save_board_state()

    def move_to_marker_towards_bottom_edge(self, distance_m: float = 0.04, z_mode: str = "keep_ee", sign_correction: int = +1):
        """
        Move the EE to a point that is `distance_m` (default 4 cm) from the ArUco
        marker center, *toward the bottom/red edge* (perpendicular to that edge).
        
        Assumptions:
        - The 'red line' is the edge whose midpoint has the largest image Y
            (bottom edge in the processed image).
        - That direction corresponds to the camera/marker +Y image direction.
        - We map that to base_link by using the marker's +Y axis (column 1 of R)
            and projecting to the base XY plane. If your rig’s sign ends up inverted,
            set sign_correction = -1.

        Args:
        distance_m: how far from the center to move (default 0.04 m).
        z_mode: "keep_ee" | "marker_z" | float (numeric Z). Same semantics as compute_board_edges_from_marker.
        sign_correction: +1 or -1 to flip direction if needed on your setup.
        """
        try:
            # 1) Marker pose in base (with your joint correction)
            pose_b = self.get_current_ee_pose()

            # 2) Rotation matrix (marker -> base), take marker +Y axis, project to base XY
            q = [pose_b.orientation.x, pose_b.orientation.y, pose_b.orientation.z, pose_b.orientation.w]
            R = tf_transformations.quaternion_matrix(q)[0:3, 0:3]
            y_axis = R[:, 1]  # marker +Y expressed in base

            # Project to XY plane & normalize
            dir_xy = np.array([float(y_axis[0]), float(y_axis[1])], dtype=float)
            n = np.linalg.norm(dir_xy)
            if n < 1e-9:
                self.log_with_time('warn', "Marker +Y nearly vertical; fallback to base +X.")
                dir_xy = np.array([1.0, 0.0], dtype=float)
            else:
                dir_xy /= n

            # We want "toward the red/bottom edge" → along image +Y.
            # If your physical mapping is flipped, invert with sign_correction.
            dir_xy *= float(sign_correction)

            # 3) Target XY = marker center + distance * dir_xy
            cx = float(pose_b.position.x) + distance_m * dir_xy[0]
            cy = float(pose_b.position.y) + distance_m * dir_xy[1]

            # 4) Choose Z
            if z_mode == "keep_ee":
                cur = self.get_current_ee_pose()
                if cur is None:
                    self.log_with_time('error', "Current EE pose unavailable.")
                    return
                z_val = float(cur.position.z)
                yaw_src = cur  # keep current yaw
            elif z_mode == "marker_z":
                z_val = float(pose_b.position.z)
                yaw_src = self.get_current_ee_pose() or pose_b
            else:
                try:
                    z_val = float(z_mode)
                except Exception:
                    z_val = float(pose_b.position.z)
                yaw_src = self.get_current_ee_pose() or pose_b

            # 5) Keep face-down orientation but preserve yaw from current EE
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([
                yaw_src.orientation.x, yaw_src.orientation.y, yaw_src.orientation.z, yaw_src.orientation.w
            ])
            q_down = tf_transformations.quaternion_from_euler(-np.pi, 0.0, yaw)

            # 6) Build pose & move
            target = self.create_pose(
                (cx, cy, z_val),
                (q_down[0], q_down[1], q_down[2], q_down[3])
            )
            self.log_with_time('info', f"Moving toward marker bottom edge by {distance_m*100:.0f} mm → ({cx:.3f}, {cy:.3f}, {z_val:.3f})")
            self.send_move_request(target, is_cartesian=False)

        except Exception as e:
            self.log_with_time('error', f"move_to_marker_towards_bottom_edge failed: {e}")


    def on_board_middle(self):
        self.refine_pose_with_ee_camera (0.0)
        self.move_to_marker_towards_bottom_edge(distance_m=0.0, z_mode="keep_ee", sign_correction=+1)


    def _update_active_point_from_current_pose(self):
        """
        Unified update-function:
        - If editing endpoint → update drop_p1 or drop_p2
        - If editing chip     → update chip_positions[self.chip_index]
        """

        ee = self.get_current_ee_pose()
        if ee is None:
            self.log_with_time("error", "EE pose unavailable.")
            return False

        x = round(ee.position.x, 3)
        y = round(ee.position.y, 3)

        # ---------------------------
        # Case 1: Editing a board endpoint
        # ---------------------------
        if self.editing_endpoint is not None:
            if self.editing_endpoint == 1:
                self.drop_p1_xy = (x, y)
                self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)
            elif self.editing_endpoint == 2:
                self.drop_p2_xy = (x, y)
                self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

            self._recompute_board_dir_xy()
            self._save_board_state()

            self.log_with_time("info",
                f"Updated board endpoint {self.editing_endpoint} → ({x:.3f}, {y:.3f})")
            return True

        # ---------------------------
        # Case 2: Editing a chip pickup position
        # ---------------------------
        if self.editing_chip is True:
            if len(self.chip_positions) > 0:
                self.chip_positions[self.cur_chip_index] = (x, y)
                self.save_chip_positions_to_json()

                self.log_with_time("info",f"Updated chip position #{self.cur_chip_index} → ({x:.3f}, {y:.3f})")
                return True
        return False


    def _update_active_point_from_current_pose_old(self):
        """
        After a motion completes, read the current EE pose and write its (x,y)
        into the active stored endpoint (drop_p1_xy or drop_p2_xy). Also updates UI.
        """
        if getattr(self, "active_drop_idx", None) not in (1, 2):
            self.log_with_time('warn', "No active drop point to update (press 'Goto 1st/2nd point' first).")
            return

        pose = self.get_current_ee_pose()
        if pose is None:
            self.log_with_time('error', "Cannot read current EE pose to update endpoint.")
            return

        new_xy = (float(pose.position.x), float(pose.position.y))
        attr = "drop_p1_xy" if self.active_drop_idx == 1 else "drop_p2_xy"
        setattr(self, attr, new_xy)

        # reflect in UI
        entry = self.drop_p1_entry if self.active_drop_idx == 1 else self.drop_p2_entry
        self._set_xy_entry(entry, new_xy)

        # keep direction vector in sync when both points exist
        self._recompute_board_dir_xy()

        self.log_with_time('info',
            f"Updated point #{self.active_drop_idx} from EE pose → ({new_xy[0]:.3f}, {new_xy[1]:.3f})")
        self._save_board_state()


    def on_board_nudge(self, direction: str):
        # step meters from entry (default 1 cm)
        try:
            step = float(self.board_step_entry.get().strip())
        except Exception:
            step = 0.005

        # compute directions
        self._recompute_board_dir_xy()
        t = self.board_dir_xy
        n = self._board_normal_xy()

        # choose direction
        if direction == "up":
            d = t * step
        elif direction == "down":
            d = -t * step
        elif direction == "left":
            d = n * step
        elif direction == "right":
            d = -n * step
        else:
            return

        # move EE
        self.move_relative_xy(d[0], d[1])

        #update endpoints only when editing ---
        if self.editing_endpoint in (1,2) or self.editing_chip is True:
            self._update_active_point_from_current_pose()
                # ---------- Chip-position editing mode ----------
        

    

    def brick_top_infos_callback(self, msg: Float32MultiArray):
        """
        Callback for receiving batched brick detections:
        Each object is [type, dx_mm, dy_mm, angle_deg, est_height_mm, cx_m, cy_m]
        """
        data = list(msg.data)
        n = len(data)
        self.latest_bricks = []

        if n == 0:
            return

        # Expect exactly 7 values per detected object
        if n % 7 != 0:
            self.log_with_time('warn', f"brick_top_infos unexpected length: {n}")
            return

        stride = 7

        for i in range(0, n, stride):
            obj_type = int(data[i + 0])
            dx_mm    = float(data[i + 1])
            dy_mm    = float(data[i + 2])
            angle    = float(data[i + 3])
            height   = float(data[i + 4])
            cx_m     = float(data[i + 5])
            cy_m     = float(data[i + 6])

            self.latest_bricks.append({
                "type": obj_type,           
                "dx_mm": dx_mm,
                "dy_mm": dy_mm,
                "angle_deg": angle,
                "est_height_mm": height,
                "cx_m": cx_m,
                "cy_m": cy_m,
            })

    def move_servo_to_angle(self, angle_deg: float):
        if not self.move_servo_client.wait_for_service(timeout_sec=1.0):
            self.log_with_time('error', "move_servo_to_angle service not available")
            return False

        req = MoveServoToAngle.Request()
        req.angle_deg = float(angle_deg)

        # Use your existing blocking helper:
        resp = self.call_service_blocking(self.move_servo_client, req, timeout_sec=5.0)
        time.sleep(1)
        if resp is None:
            self.log_with_time('error', "move_servo_to_angle timed out / failed")
            return False

        if getattr(resp, "success", False):
            self.log_with_time('info', f"Servo moved to {angle_deg:.1f}°")
            self.gui_queue.put(lambda: self.status_label.config(
                                    text=f"Servo → {angle_deg:.1f}°"
                                ))

            return True
        else:
            msg = getattr(resp, "message", "(no message)")
            self.log_with_time('warn', f"move_servo_to_angle failed: {msg}")
            self.gui_queue.put(lambda: self.status_label.config(
                text=f"Servo move failed: {msg}"
            ))
            return False


    def log_with_time(self, level, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        if level == 'info':
            # self.get_logger().info(full_message)
            print(full_message)  # For GUI visibility
        elif level == 'error':
            # self.get_logger().error(full_message)
            print(full_message)  # For GUI visibility
        elif level == 'warn':
            # self.get_logger().warn(full_message)
            print(full_message)  # For GUI visibility
        else:
            #self.get_logger().debug(full_message)    
            print(full_message)  # For GUI visibility



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
        if len(msg.data) != 5:
            self.log_with_time('warn' ,f"Received brick_top_info with unexpected length: {len(msg.data)}")
            return

        obj_type, dx, dy, angle, height = msg.data

        self.top_type = int(obj_type)         
        self.top_dx_mm = float(dx)
        self.top_dy_mm = float(dy)
        self.top_angle_deg = float(angle)
        self.top_est_height_mm = float(height)


    def move_to_chip_start(self):
        

        x = self.chip_start_x 
        y = self.chip_start_y
        z = self.chip_start_z
        
        self.log_with_time("info", f"Moving to chip start: ({x:.3f}, {y:.3f}, {z:.3f})")
        ok = self.move_to_xy(x, y, z)
        self.log_with_time("info", f"Chip start move result: {ok}")
        return ok

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

        # Container frame for the grid
        zgrid = tk.Frame(zgrp)
        zgrid.pack()

        buttons = [
            ("45 cm", 0.45),
            ("40 cm", 0.40),
            ("37 cm", 0.37),
            ("35 cm", 0.35),
            ("30 cm", 0.30),
            ("25 cm", 0.25),
            ("21 cm", 0.21),
            ("20 cm", 0.20),
            ("17 cm", 0.17),
            ("14 cm", 0.14),
            ("13 cm", 0.13),
            ("12 cm", 0.12),
        ]
        count_one_line = int(buttons.__len__() / 2)
        # Build two rows
        for i, (label, height) in enumerate(buttons):
            row = 0 if i < count_one_line else 1    # first 5 buttons in row 0, rest in row 1
            col = i if i < count_one_line else (i - count_one_line)
            tk.Button(
                zgrid,
                text=label,
                width=8,
                command=lambda h=height: self.move_to_height(h)
            ).grid(row=row, column=col, padx=4, pady=5)



        newgrp = tk.LabelFrame(self.tab_tools, text="Other Controls")
        newgrp.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(newgrp, text="Chip Start", command=self.move_to_chip_start, width=8).pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(newgrp, text="Start", command=self.start_collection, width=8).pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(newgrp, text="Tower", command=self.stack_all_bricks, width=8).pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(newgrp, text="Refine", command=self.refine_pos, width=8).pack(side=tk.LEFT, padx=4, pady=5)
        tk.Button(newgrp, text="disc. EPs", command=self.disconnect_endpoints, width=8).pack(side=tk.LEFT, padx=4, pady=5)


        # --------------------------------------------------
        # CHIP CONTROLS (CLEAN LAYOUT)
        # --------------------------------------------------
        chip_grp = tk.LabelFrame(self.tab_tools, text="Chip Controls")
        chip_grp.pack(fill=tk.X, padx=8, pady=6)

        # ---------- Top actions ----------
        top = tk.Frame(chip_grp)
        top.pack(fill=tk.X, pady=(4, 8))

        tk.Button(top, text="Collect Chip",
                command=self.collect_chip, width=9)\
            .pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Next Move",
                command=self.on_next_move, width=9)\
            .pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Set Chip Positions",
                command=self.on_set_chip_positions, width=12)\
            .pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Next Chip Position",
                command=self.on_next_chip_position, width=13)\
            .pack(side=tk.LEFT, padx=4)
        
        tk.Button(top, text="Set Box",
                command=self.on_set_box, width=9)\
            .pack(side=tk.LEFT, padx=4)        

        # ---------- Board drop points section ----------
        box = tk.LabelFrame(chip_grp, text="Drop Line Calibration")
        box.pack(fill=tk.X, padx=4, pady=8)

        # --- Row 0: Compute + step ---
        tk.Button(box, text="Compute Drop Points",
                command=self.on_compute_drop_points, width=20)\
            .grid(row=0, column=0, padx=4, pady=4, sticky="w")

        tk.Label(box, text="Step (m):")\
            .grid(row=0, column=1, padx=4, pady=4, sticky="e")

        self.board_step_entry = tk.Entry(box, width=10)
        self.board_step_entry.grid(row=0, column=2, padx=4, pady=4, sticky="w")
        self.board_step_entry.insert(0, "0.01")

        # Quick step buttons
        quick = tk.Frame(box)
        quick.grid(row=0, column=3, padx=4, pady=4, sticky="w")
        for val in [0.01, 0.005, 0.002]:
            tk.Button(quick, text=f"{val:.3f}", width=6,
                    command=lambda v=val: (self.board_step_entry.delete(0, tk.END),
                                            self.board_step_entry.insert(0, f"{v:.3f}"))
                    ).pack(side=tk.LEFT, padx=1)


        # --- Row 1 + 2 : entry boxes ---
        tk.Label(box, text="Point 1 (x,y):")\
            .grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.drop_p1_entry = tk.Entry(box, width=28)
        self.drop_p1_entry.grid(row=1, column=1, columnspan=3, padx=4, pady=2, sticky="w")

        tk.Label(box, text="Point 2 (x,y):")\
            .grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self.drop_p2_entry = tk.Entry(box, width=28)
        self.drop_p2_entry.grid(row=2, column=1, columnspan=3, padx=4, pady=2, sticky="w")


        # ---------- Nudges ----------
        nudges = tk.LabelFrame(box, text="Adjust Board Line")
        nudges.grid(row=3, column=0, columnspan=4, pady=(10, 4), padx=4, sticky="w")

        # Row 0
        tk.Button(nudges, text="Up", width=7,
                command=lambda: self.on_board_nudge("up")).grid(row=0, column=0, padx=2, pady=2)

        tk.Button(nudges, text="Down", width=7,
                command=lambda: self.on_board_nudge("down")).grid(row=0, column=1, padx=2, pady=2)

        tk.Button(nudges, text="Left", width=7,
                command=lambda: self.on_board_nudge("left")).grid(row=0, column=2, padx=2, pady=2)

        tk.Button(nudges, text="Right", width=7,
                command=lambda: self.on_board_nudge("right")).grid(row=0, column=3, padx=2, pady=2)

        tk.Button(nudges, text="Middle", width=7,
                command=self.on_board_middle).grid(row=0, column=4, padx=4, pady=2)

        tk.Button(nudges, text="Next Column", width=12,
                command=self.on_next_board_column).grid(row=0, column=5, padx=6, pady=2)


        # Row 1
        tk.Button(nudges, text="CW", width=7,
                command=lambda: self.nudge_ee_yaw(-2.0)).grid(row=1, column=0, padx=2, pady=2)

        tk.Button(nudges, text="CCW", width=7,
                command=lambda: self.nudge_ee_yaw(+2.0)).grid(row=1, column=1, padx=2, pady=2)

        tk.Button(nudges, text="Pickup Chip", width=12,
                command=self.pickup_chip).grid(row=1, column=2, padx=2, pady=2)

        # Row 2: refine
        self.refine_btn = tk.Button(nudges,
                text="Refine Column\n&& Update Endpoints",
                command=self.on_refine_column_update_endpoints,
                bg="#d0ffd0", width=28)
        self.refine_btn.grid(row=2, column=0, columnspan=6, pady=8)

        # ---------- Go-to buttons ----------
        goto = tk.Frame(box)
        goto.grid(row=4, column=0, columnspan=4, pady=(8,4), padx=4, sticky="w")

        tk.Button(goto, text="Go to 1st point", width=16,
                command=lambda: self.on_goto_point(1)).pack(side=tk.LEFT, padx=4)

        tk.Button(goto, text="Go to 2nd point", width=16,
                command=lambda: self.on_goto_point(2)).pack(side=tk.LEFT, padx=4)

        # ---------- Align row ----------
        align = tk.Frame(box)
        align.grid(row=5, column=0, columnspan=4, pady=(4,8), padx=4, sticky="w")

        tk.Button(align, text="Align marker (horiz.)", width=18,
                command=lambda: self.align_marker_with_image(axis_in_marker="x",
                                                            target="horizontal"))\
            .pack(side=tk.LEFT, padx=4)

        tk.Button(align, text="Align marker (vert.)", width=18,
                command=lambda: self.align_marker_with_image(axis_in_marker="x",
                                                            target="vertical"))\
            .pack(side=tk.LEFT, padx=4)



        # Start periodic updates
        self.update_position_label()

    def on_set_chip_positions(self):
        """
        Move to identification pose, detect chips, compute base-frame positions,
        round to 3 decimals, and save to JSON.
        """

        self.log_with_time("info", "Setting chip positions...")

        # # 1) Move robot to identification viewpoint
        # x, y, z = self.chip_ident_x, self.chip_ident_y, self.chip_ident_z
        # ok = self.move_to_xy_align_board(x, y, z)
        # if not ok:
        #     self.log_with_time("error", "Failed to move to identification pose.")
        #     return

        time.sleep(self.wait_before_refine_sec)

        # 2) Use ALL available chips
        if not hasattr(self, "latest_bricks") or len(self.latest_bricks) == 0:
            self.log_with_time("warn", "No chips detected.")
            return

        bricks = self.latest_bricks

        # 3) Get EE pose
        ee = self.get_current_ee_pose()
        if ee is None:
            self.log_with_time("error", "EE pose unavailable.")
            return

        ex = ee.position.x
        ey = ee.position.y

        found_positions = []

        # 4) For each chip: apply dx/dy directly (already in base frame)
        for b in bricks:
            dx = round(b["dx_mm"] / 1000.0, 3)  # m
            dy = round(b["dy_mm"] / 1000.0, 3)  # m

            chip_x = round(ex + dx, 3)
            chip_y = round(ey + dy, 3)

            found_positions.append((chip_x, chip_y))

            self.log_with_time(
                "info",
                f"Chip dx={dx:.3f}m dy={dy:.3f}m → base ({chip_x:.3f}, {chip_y:.3f})"
            )

        # 5) Save final array (flexible size)
        self.chip_positions = found_positions
        self.save_chip_positions_to_json()

        self.log_with_time("info", f"Stored chip positions ({len(found_positions)} chips).")

    def on_next_move(self):
        self.load_robot_params()
        self.send_rotated_board_box()
        if self.latest_board is None:
            self.get_logger().warn("No board state available yet!!!!!!!!!!!!!")
            return

        if not self.ai_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Connect4 AI service unavailable!")
            return

        # Prepare request
        req = GetNextMove.Request()
        req.board = self.latest_board.flatten().tolist()

        # Decide which player you are
        # P1=1, P2=2 — your robot is P2?
        req.player = 2

        future = self.ai_client.call_async(req)
        future.add_done_callback(self.after_ai_move)

    def after_ai_move(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"AI service failed: {e}")
            return

        col = res.column

        if col < 0 or col > 6:
            self.get_logger().error("AI returned invalid column.")
            return

        self.get_logger().info(f"AI selected column {col}")

        # Now command the robot to place a piece in this column
        # self.place_chip(col)
        self.gui_queue.put(lambda: self.place_chip(col))


    def place_chip(self, column):
        """
        Move the arm to the correct drop position for the chosen column.
        """
        if self.current_state != STATE_COLLECTED_CHIP:
            self.move_to_chip_start()   # already exists :contentReference[oaicite:1]{index=1}
            self.increment_chip_index()
            ok = self.collect_chip()
            if not ok:
                self.log_with_time("error", "Failed to collect chip before placing.")
                return

        self.goto_board_column(column)
        self.drop_chip_at_current_column() 
        self.current_state = STATE_DROPPED_CHIP

        # self.move_to_chip_start()   # already exists :contentReference[oaicite:1]{index=1}

        ok = self.collect_chip()
        if not ok:
            self.log_with_time("error", "Failed to collect chip before placing.")
            return        


    def disconnect_endpoints(self):
        self.editing_endpoint = None

    def refine_current_column_center(self):
        """
        Refine the robot's XY so the EE moves to the true center of
        the column directly underneath, using two refine_pos() iterations.
        Returns: (cx, cy) in base frame.
        """

        # Call refine twice for better accuracy
        for _ in range(2):
            time.sleep(0.5)
            self._refine_pos()

        # Retrieve the current EE pose (geometry_msgs/Pose)
        pose = self.get_current_ee_pose()

        if pose is None:
            self.log_with_time("error", "refine_current_column_center: EE pose unavailable!")
            return None, None

        cx = pose.position.x
        cy = pose.position.y

        if cx is None or cy is None:
            self.log_with_time("error", "refine_current_column_center: invalid EE XY!")
            return None, None

        return cx, cy



    def on_refine_column_update_endpoints(self):
        """
        Refines the robot position to the true center of the current column,
        then updates board endpoints accordingly.
        Assumes the arm is already above some column and the gripper is open.
        """
        try:
            # 1. Get EE pose
            pose = self.get_current_ee_pose()
            if pose is None:
                self.log_with_time("error", "EE pose unavailable, cannot refine.")
                return

            # 2. Identify closest column from current XY
            col = self.find_closest_column()
            if col is None:
                self.log_with_time("warn", "Cannot determine closest column.")
                return

            self.log_with_time("info", f"Refining at column {col}")

            # 3. Refine twice (your requirement)
            refined_x, refined_y = self.refine_current_column_center()

            # 4. Update endpoints based on refined center
            self.update_endpoints_after_refine(refined_center=(refined_x, refined_y),
                                            column_index=col)

            # 5. Recompute direction and force UI update
            self._recompute_board_dir_xy()
            
            self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)
            self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

            self.log_with_time("info",
                f"Endpoints updated after refinement at column {col}.\n"
                f"New P1: {self.drop_p1_xy}, P2: {self.drop_p2_xy}"
            )

        except Exception as e:
            self.log_with_time("error", f"Refinement error: {e}")


    def find_closest_column(self):
        pose = self.get_current_ee_pose()
        if pose is None:
            return None

        ee_x = pose.position.x
        ee_y = pose.position.y

        best_i = None
        best_dist = float("inf")

        for i in range(self.num_board_columns):
            xy = self.compute_board_column_xy(i)
            if xy is None:
                continue

            d = np.hypot(ee_x - xy[0], ee_y - xy[1])
            if d < best_dist:
                best_dist = d
                best_i = i

        return best_i


    def compute_shift(self, col_idx, refined_xy):
        expected = self.compute_board_column_xy(col_idx)
        if expected is None:
            return None

        ex, ey = expected
        rx, ry = refined_xy

        dx = rx - ex
        dy = ry - ey

        self.log_with_time("info", f"Shift Δ = ({dx:.4f}, {dy:.4f})")
        return (dx, dy)

    def apply_shift_to_endpoints(self, shift):
        if shift is None:
            return

        dx, dy = shift

        # P1
        if self.drop_p1_xy:
            x1, y1 = self.drop_p1_xy
            self.drop_p1_xy = (x1 + dx, y1 + dy)
            self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)

        # P2
        if self.drop_p2_xy:
            x2, y2 = self.drop_p2_xy
            self.drop_p2_xy = (x2 + dx, y2 + dy)
            self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

        self._recompute_board_dir_xy()
        self._save_board_state()

        self.log_with_time("info", "Endpoints shifted according to refined center")



    def on_next_board_column(self):
        """Cycle 0→6 and move to that column."""
        self.editing_endpoint = None   # disable endpoint editing session
        self.send_rotated_board_box()
        self.current_column_idx = (self.current_column_idx + 1) % self.total_columns
        self.goto_board_column(self.current_column_idx)

    def move_to_xy(self, x: float, y: float, z: float = None, is_cartesian: bool = False):
        """Move EE to (x,y) keeping current Z and orientation."""
        cur = self.get_current_ee_pose()
        if cur is None:
            self.log_with_time('error', "Cannot read current end-effector pose.")
            return
        target_z = z if z is not None else float(cur.position.z)
        pose = self.create_pose(
            (float(x), float(y), float(target_z)),
            (cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w)
        )
        return self.send_move_request(pose, is_cartesian=is_cartesian)

    def on_goto_point(self, idx: int):
        self.editing_endpoint = idx

        """Move to one of the two stored drop points, always using stored values."""
        # Select the correct stored tuple
        xy = getattr(self, "drop_p1_xy" if idx == 1 else "drop_p2_xy", None)

        if xy is None:
            self.log_with_time('warn', f"No stored point #{idx} available.")
            return
        self.active_drop_idx = idx  # <-- remember which one is active now

        try:
            x, y = float(xy[0]), float(xy[1])
        except Exception as e:
            self.log_with_time('error', f"Invalid stored point #{idx}: {e}")
            return

        self.log_with_time('info', f"Going to stored point #{idx}: ({x:.3f}, {y:.3f})")
        self.move_to_xy_align_board(float(xy[0]), float(xy[1]), self.board_hover_z, is_cartesian=False)
        self.send_rotated_board_box()
        self.publish_safe_points()

    def update_endpoints_after_refine(self, refined_center, column_index):
        """
        After refining the true center of the current column, shift the drop-line endpoints.

        - If the refined column is the FIRST column → move only drop_p1_xy to refined_center.
        - If the refined column is the LAST column → move only drop_p2_xy to refined_center.
        - Otherwise → shift BOTH endpoints by (dx, dy) so the refined center matches.

        refined_center: (x, y)
        column_index: integer index of the column that was refined
        """

        old_center = self.compute_board_column_xy(column_index)
        if old_center is None:
            self.log_with_time('warn', f"Cannot update endpoints: column {column_index} invalid.")
            return

        new_x, new_y = refined_center
        old_x, old_y = old_center

        dx = new_x - old_x
        dy = new_y - old_y

        num_cols = self.num_columns if hasattr(self, "num_columns") else 7

        # ============================================================
        # CASE 1: FIRST COLUMN → update only drop_p1_xy
        # ============================================================
        if column_index == 0:
            self.drop_p1_xy = (new_x, new_y)
            self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)

            # don't modify drop_p2_xy
            self._recompute_board_dir_xy()
            self._save_board_state()
            self.log_with_time(
                'info',
                f"Refined FIRST column → updated drop_p1 only to ({new_x:.4f},{new_y:.4f})."
            )
            return

        # ============================================================
        # CASE 2: LAST COLUMN → update only drop_p2_xy
        # ============================================================
        if column_index == (num_cols - 1):
            self.drop_p2_xy = (new_x, new_y)
            self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

            # don't modify drop_p1_xy
            self._recompute_board_dir_xy()
            self._save_board_state()
            self.log_with_time(
                'info',
                f"Refined LAST column → updated drop_p2 only to ({new_x:.4f},{new_y:.4f})."
            )
            return

        # ============================================================
        # CASE 3: INNER COLUMN → shift both endpoints by (dx, dy)
        # ============================================================
        if self.drop_p1_xy is not None:
            x1, y1 = self.drop_p1_xy
            self.drop_p1_xy = (x1 + dx, y1 + dy)
            self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)

        if self.drop_p2_xy is not None:
            x2, y2 = self.drop_p2_xy
            self.drop_p2_xy = (x2 + dx, y2 + dy)
            self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

        self._recompute_board_dir_xy()
        self._save_board_state()

        self.log_with_time(
            'info',
            f"Endpoints shifted → dx={dx:.4f}, dy={dy:.4f} for refined column {column_index}."
        )


    def pickup_chip(self):
        """
        Full automatic sequence:
        1) Move to chip start
        2) Collect one chip       (same as pressing 'Collect Chip')
        3) Move to the NEXT Connect-4 column
        4) Drop the chip
        """
        self.log_with_time("info", "=== PLACE ONE CHIP SEQUENCE START ===")

        # 1) Move to chip start
        self.move_to_chip_start()   # already exists :contentReference[oaicite:1]{index=1}
        self.latest_bricks = []  # clear previous bricks
        time.sleep(0.15)
        # 2) Collect chip (same function called from the Collect Chip button)


        ok = self.collect_chip()

        if not ok:
            self.log_with_time("error", "Failed to collect chip")
            return False

        # 3) Move to next column (uses your existing logic)
        # self.on_next_board_column()  # already exists :contentReference[oaicite:3]{index=3}
        # self.drop_chip_at_current_column()   
        

    def drop_chip_at_current_column(self):
        # 4) Drop the chip (open gripper)
        self.open_gripper_srv()      # already exists :contentReference[oaicite:4]{index=4}
        self.log_with_time("info", "✅ Chip dropped!")

        time.sleep(self.wait_before_refine_sec)  # allow camera to stabilize after gripper motion
        if len(self.latest_bricks) >1:
            self.log_with_time("warn", f"Multiple bricks detected after dropping chip, skipping alignment: {len(self.latest_bricks)}")
            return

         # --- Determine closest column from EE pose ---
        col = self.find_closest_column()
        if col is None:
            self.log_with_time('warn', "Could not determine closest column — skipping refinement.")
            return

        for i in range(1):
            # --- Refine column center ---
            refined_x, refined_y = self.refine_current_column_center()

            if refined_x is None:
                self.log_with_time('warn', "Refinement failed — skipping endpoint update.")
                return

            # --- Update endpoints to realign the column line ---
            self.update_endpoints_after_refine((refined_x, refined_y), col)

        self.log_with_time(
            'info',
            f"Refinement complete. Column {col} center=({refined_x:.3f},{refined_y:.3f}). Endpoints updated."
        )

    def increment_chip_index(self):
        """Increment cur_chip_index, wrapping around if necessary."""
        if not hasattr(self, "chip_positions") or len(self.chip_positions) == 0:
            self.cur_chip_index = 0
            return

        self.cur_chip_index = (self.cur_chip_index + 1) % len(self.chip_positions)
        self.log_with_time(
            "info",
            f"Current chip index set to {self.cur_chip_index}."
        )
        
    def _get_closest_chip_to_ee(self):
        """
        Return (cx, cy) [meters] of the brick currently closest to the end effector,
        or None if none are available.
        """
        all_bricks = list(getattr(self, "latest_bricks", []))

        chips = [b for b in all_bricks if b.get("type") == 1]

        if not chips:
            self.log_with_time('warn', "No chips available to collect.")
            return None

        cur = self.get_current_ee_pose()
        if not cur:
            self.log_with_time('error', "Current EE pose unavailable.")
            return None

        def dist_to_ee(b):
            if b.get("cx_m") is None or b.get("cy_m") is None:
                return float("inf")
            return ((b["cx_m"] - cur.position.x) ** 2 +
                    (b["cy_m"] - cur.position.y) ** 2) ** 0.5

        target = min(chips, key=dist_to_ee)
        if target.get("cx_m") is None or target.get("cy_m") is None:
            self.log_with_time('warn', "Closest chip missing coordinates.")
            return None

        cx, cy = float(target["cx_m"]), float(target["cy_m"])
        self.log_with_time('info', f"Closest chip at ({cx:.3f}, {cy:.3f})")
        return cx, cy

    def collect_chip(self):
        
        """
        Try each chip_position until a chip is detected at that location.
        If detected → refine + pick (chip_index stays the same).
        If none detected in any location → return False.
        """
        self.load_robot_params()
        self.send_rotated_board_box()
        # ---------------------------------------
        # Validate chip_positions
        # ---------------------------------------
        if not hasattr(self, "chip_positions") or len(self.chip_positions) == 0:
            self.log_with_time("error", "No chip_positions stored. Run 'Set Chip Positions' first.")
            return False

        num_positions = len(self.chip_positions)

        # ---------------------------------------
        # Get EE yaw (face-down)
        # ---------------------------------------
        cur = self.get_current_ee_pose()
        if not cur:
            self.log_with_time("error", "Current EE pose unavailable.")
            return False

        _, _, yaw = tf_transformations.euler_from_quaternion([
            cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w
        ])

        def face_down_pose(x, y, z, yaw_rad):
            q = tf_transformations.quaternion_from_euler(-np.pi, 0.0, yaw_rad)
            return self.create_pose(
                (float(x), float(y), float(z)),
                (q[0], q[1], q[2], q[3])
            )

        prep_z = float(self.prepare_to_pick_z)

        # =====================================================
        # LOOP THROUGH ALL CHIP POSITIONS UNTIL A CHIP IS FOUND
        # =====================================================
        attempts = 0

        while attempts < num_positions:

            tx, ty = self.chip_positions[self.cur_chip_index]

            self.log_with_time(
                "info",
                f"Checking chip-position index {self.cur_chip_index}: ({tx:.3f}, {ty:.3f})"
            )

            # ---------------------------------------
            # Move to this candidate location
            # ---------------------------------------
            pose = face_down_pose(tx, ty, prep_z, yaw)
            self.open_gripper_srv()

            ok = self.send_move_request(pose, is_cartesian=False)
            if not ok:
                self.log_with_time("error", f"Failed to move to chip position #{self.cur_chip_index}.")
                if self.move_to_chip_start() == True:
                    ok = self.send_move_request(pose, is_cartesian=False)
                    if not ok:
                        self.log_with_time("error", f"Retry also failed to move to chip position #{self.cur_chip_index}.")
                        return False
                else:
                    self.log_with_time("error", "Also failed to return to chip start position.")
                    return False
            self.latest_bricks = []  # clear previous bricks
            # Give camera time after stopping
            time.sleep(self.wait_before_refine_sec)

            # Try to detect a chip
            latest_target = self._get_closest_chip_to_ee()

            if latest_target is not None:
                # ---------------------------------------
                # ✔ CHIP FOUND at this chip_index
                # ---------------------------------------
                cx, cy = latest_target

                self.log_with_time(
                    "info",
                    f"Chip detected at index {self.cur_chip_index} → refining towards ({cx:.3f},{cy:.3f})"
                )

                # First refinement step (still above board)
                self.refine_pose_with_ee_camera_dx_dy(cx, cy, prep_z)
                time.sleep(self.wait_before_refine_sec)

                latest_target = self._get_closest_chip_to_ee()
                cx, cy = latest_target
                self.refine_pose_with_ee_camera_dx_dy(cx, cy, prep_z)


                # Recompute yaw after refine
                cur = self.get_current_ee_pose()
                _, _, yaw = tf_transformations.euler_from_quaternion([
                    cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w
                ])

                # Get updated detection again
                time.sleep(self.wait_before_refine_sec)
                latest_target = self._get_closest_chip_to_ee()
                if latest_target:
                    cx, cy = latest_target

                # Move down to pick
                pick_pose = face_down_pose(cx, cy, float(self.pick_z3), yaw)
                if not self.send_move_request(pick_pose, is_cartesian=False):
                    self.log_with_time("error", "Pick move failed.")
                    return False

                # Grab
                self.close_gripper_srv()

                # Lift
                self.move_to_height(float(self.lift_z), is_cartesian=True)

                self.current_state = STATE_COLLECTED_CHIP
                self.log_with_time("info", "Chip collected successfully.")
                return True

            # ---------------------------------------
            # ❌ No chip found → move to next chip_index
            # ---------------------------------------
            self.log_with_time(
                "info",
                f"No chip detected at index {self.cur_chip_index}. Trying next."
            )

            self.increment_chip_index()
            attempts += 1

        # =====================================================
        # After checking all positions → no chips anywhere
        # =====================================================
        self.move_to_chip_start()
        self.log_with_time("info", "No chips detected at ANY chip position.")
        return False

    def start_collection(self):
        """
        Move to the start_position defined in robot_params.json.
        Orientation remains the original hard-coded quaternion.
        """

        # Ensure params are loaded
        self.load_robot_params()

        # Read XYZ from JSON
        sx = float(self.start_x)
        sy = float(self.start_y)
        sz = float(self.start_z)

        # Keep your original orientation
        qx, qy, qz, qw = 0.000, 1.000, 0.002, 0.000

        pose = self.make_pose_xyz_quat(sx, sy, sz, qx, qy, qz, qw)

        ok = self.send_move_request(pose, is_cartesian=False)

        if ok:
            self.log_with_time(
                "info",
                f"Moved to start position ({sx:.3f}, {sy:.3f}, {sz:.3f})"
            )
        else:
            self.log_with_time("error", "Failed to move to start position.")

        return ok


    def refine_pos(self):
        self.send_rotated_board_box()
        self._refine_pos()

        self.refine_board_endpoints()
        self.refine_chip_position()


    def refine_chip_position(self):
        """
        If in chip-position edit mode, update the chip position at chip_index
        based on the current EE pose.
        """

        # ---------------------------------------------------
        # Only refine if we're in chip-position edit mode
        # ---------------------------------------------------
        if not getattr(self, "editing_chip", False):
            return

        # ---------------------------------------------------
        # Get the EE pose
        # ---------------------------------------------------
        pose = self.get_current_ee_pose()
        if pose is None:
            return

        # Extract XY from geometry_msgs/Pose OR tuple
        if hasattr(pose, "position"):
            ee_x = pose.position.x
            ee_y = pose.position.y
        else:
            ee_x = pose[0]
            ee_y = pose[1]

        if ee_x is None or ee_y is None:
            return

        # ---------------------------------------------------
        # Check chip_positions list
        # ---------------------------------------------------
        if not hasattr(self, "chip_positions") or len(self.chip_positions) == 0:
            self.log_with_time("warn", "No chip positions to refine.")
            return

        # ---------------------------------------------------
        # Update current chip_index entry
        # ---------------------------------------------------
        new_x = round(ee_x, 3)
        new_y = round(ee_y, 3)

        self.chip_positions[self.cur_chip_index] = (new_x, new_y)

        # Save immediately to JSON
        self.save_chip_positions_to_json()

        self.log_with_time(
            "info",
            f"Refined chip position #{self.cur_chip_index} to ({new_x:.3f}, {new_y:.3f})"
        )


    def refine_board_endpoints(self):
        if self.editing_endpoint is not None:
            pose = self.get_current_ee_pose()   # <-- NEW

            if pose is None:
                return

            # Extract XY depending on pose structure
            if hasattr(pose, "position"):  
                # geometry_msgs/Pose
                ee_x = pose.position.x
                ee_y = pose.position.y
            else:
                # tuple-like: (x, y, z, qx, qy, qz, qw)
                ee_x = pose[0]
                ee_y = pose[1]

            if ee_x is not None and ee_y is not None:
                if self.editing_endpoint == 1:
                    self.drop_p1_xy = (ee_x, ee_y)
                    self._set_xy_entry(self.drop_p1_entry, self.drop_p1_xy)

                elif self.editing_endpoint == 2:
                    self.drop_p2_xy = (ee_x, ee_y)
                    self._set_xy_entry(self.drop_p2_entry, self.drop_p2_xy)

                self._recompute_board_dir_xy()
                self._save_board_state()

                self.log_with_time(
                    "info",
                    f"Updated drop point #{self.editing_endpoint} to ({ee_x:.4f}, {ee_y:.4f}) after refinement."
                )


    def _refine_pos(self):
        self.refine_pose_with_ee_camera (0.0)

    def stack_all_bricks(
        self,
        brick_thickness=0.0746,   # 7.46 mm in meters
        base_policy="farthest",    # "nearest" or "farthest"
        approach_pick_z=0.20,      # hover above pickup
        align_pick_z=0.19,         # refine height before grasp
        grasp_z=0.13,              # close gripper here
        approach_base_z=0.10,      # exactly 10 cm above base for the *initial* measurement
        retreat_z=0.30             # rise after pick/place
    ):
        """
        Stack all visible bricks on a chosen base.
        - Snapshot bricks once (do not use live latest_bricks after moving).
        - Measure base once at 10 cm (stores precise x,y and yaw).
        - For each remaining brick: pick with local refine; then place at stored base x,y
        and increasing Z (brick_thickness per level). No re-observe of base.
        """
        # 1) Snapshot bricks (positions are from the start view)
        bricks = list(getattr(self, "latest_bricks", []))
        bricks = [b for b in bricks if b.get("cx_m") is not None and b.get("cy_m") is not None]
        if len(bricks) < 2:
            self.log_with_time('warn', f"Need >=2 bricks, got {len(bricks)}.")
            return False

        cur = self.get_current_ee_pose()
        if not cur:
            self.log_with_time('error', "Current EE pose unavailable.")
            return False

        # Choose base: nearest/farthest to current EE
        def dist_to_ee(b):
            return ((b["cx_m"] - cur.position.x)**2 + (b["cy_m"] - cur.position.y)**2) ** 0.5

        bricks.sort(key=dist_to_ee)
        if base_policy == "nearest":
            base = bricks[0]; movers = bricks[1:]
        else:
            base = bricks[-1]; movers = bricks[:-1]

        base_x0, base_y0 = float(base["cx_m"]), float(base["cy_m"])
        self.log_with_time('info', f"Base brick @ ({base_x0:.3f}, {base_y0:.3f}); stacking {len(movers)} bricks.")

        import numpy as np, tf_transformations
        # Keep face-down; keep yaw from the *measured* base pose
        def quat_face_down_with_yaw(yaw_rad):
            return tf_transformations.quaternion_from_euler(-np.pi, 0.0, yaw_rad)

        # Helper to build poses
        def pose_xy_z_quat(x, y, z, q):
            return self.create_pose((float(x), float(y), float(z)),
                                    (float(q[0]), float(q[1]), float(q[2]), float(q[3])))

        # 2) Measure base precisely ONCE (10 cm above), then freeze that pose
        #    Uses the refined EE pose to lock base_x/base_y and yaw
        self.send_move_request(
            pose_xy_z_quat(base_x0, base_y0, approach_base_z+brick_thickness, quat_face_down_with_yaw(
                tf_transformations.euler_from_quaternion([cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w])[2]
            )),
            is_cartesian=False
        )
        # refine here is OK (camera clear); it updates EE to the true base center/orientation
        self.refine_pose_with_ee_camera_dx_dy(base_x0, base_y0, height=approach_base_z+brick_thickness)

        pose_after_refine = self.get_current_ee_pose()
        if not pose_after_refine:
            self.log_with_time('error', "Failed to read pose after base refine.")
            return False

        base_ref_x = float(pose_after_refine.position.x)
        base_ref_y = float(pose_after_refine.position.y)
        # freeze yaw from the refined base pose
        _, _, base_ref_yaw = tf_transformations.euler_from_quaternion([
            pose_after_refine.orientation.x,
            pose_after_refine.orientation.y,
            pose_after_refine.orientation.z,
            pose_after_refine.orientation.w
        ])
        q_down_base = quat_face_down_with_yaw(base_ref_yaw)

        self.log_with_time('info', f"Frozen base @ ({base_ref_x:.3f}, {base_ref_y:.3f}), yaw={np.degrees(base_ref_yaw):.1f}°, z={approach_base_z:.2f} m")

        # 3) Stack: for each other brick → pick (with local refine), then place at frozen base (no re-observe)
        levels_on_base = 1  # base brick already there

        for i, b in enumerate(movers):
            bx, by = float(b["cx_m"]), float(b["cy_m"])
            self.log_with_time('info', f"[{i+1}/{len(movers)}] pick from ({bx:.3f}, {by:.3f})")

            # --- PICK sequence (local refine OK; camera sees the loose brick) ---
            self.send_move_request(pose_xy_z_quat(bx, by, approach_pick_z, q_down_base), is_cartesian=False)
            self.open_gripper_srv()
            self.refine_pose_with_ee_camera_dx_dy(bx, by, height=align_pick_z)
            self.refine_pose_with_ee_camera_dx_dy(bx, by, height=grasp_z)
            self.close_gripper_srv()
            self.move_to_height(retreat_z, is_cartesian=False)

            # --- PLACE sequence (DO NOT re-observe base; camera may be occluded) ---
            # compute stack-top Z
            place_z = grasp_z + (levels_on_base) * brick_thickness

            # go above frozen base x,y, then descend to place_z, then open
            self.send_move_request(pose_xy_z_quat(base_ref_x, base_ref_y, retreat_z, q_down_base), is_cartesian=False)
            self.send_move_request(pose_xy_z_quat(base_ref_x, base_ref_y, place_z,  q_down_base), is_cartesian=False)
            self.open_gripper_srv()

            levels_on_base += 1

            # retreat, optionally pre-position over next brick
            self.move_to_height(retreat_z, is_cartesian=False)
            if i + 1 < len(movers):
                nb = movers[i + 1]
                self.send_move_request(pose_xy_z_quat(float(nb["cx_m"]), float(nb["cy_m"]), retreat_z, q_down_base),
                                    is_cartesian=False)

        self.log_with_time('info', "stack_all_bricks_precise_frozen_base: done.")
        return True



    # Helper: build a face-down pose at (x,y,z) with given yaw
    def face_down_pose(self,x, y, z, yaw_rad):
            q = tf_transformations.quaternion_from_euler(-np.pi, 0.0, yaw_rad)
            return self.create_pose((x, y, z), (q[0], q[1], q[2], q[3]))

    def collect_bricks(self):
        # Snapshot current bricks (list of dicts with cx_m, cy_m, etc.)
        bricks = list(getattr(self, "latest_bricks", []))
        if not bricks:
            self.log_with_time('warn', "No bricks to collect (snapshot empty).")
            return

        cur = self.get_current_ee_pose()
        if not cur:
            self.log_with_time('error', "Current EE pose unavailable.")
            return



        # Keep current yaw for all “hover”/drop/above-next moves
        _, _, yaw = tf_transformations.euler_from_quaternion([
            cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w
        ])

        HOVER_Z = 0.30      # safe travel height
        PICK_Z1  = 0.21      # approach height used by refine function
        PICK_Z2  = 0.15      # approach height used by refine function
        PICK_Z3  = 0.13      # approach height used by refine function

        DROP_Z  = 0.30

        # Bin/drop location (adjust to your setup)
        drop_xy = (-0.25, -0.30)
        drop_pose = self.face_down_pose(drop_xy[0], drop_xy[1], DROP_Z, yaw)

        # Iterate by index so we can look ahead to the “next” brick
        for i in range(len(bricks)):
            b = bricks[i]
            cx = b.get("cx_m")
            cy = b.get("cy_m")
            if cx is None or cy is None:
                self.log_with_time('warn', f"Brick #{i+1}: missing center base coords, skipping.")
                continue

            self.log_with_time('info', f"Collecting brick #{i+1} at ({cx:.3f}, {cy:.3f})")

            # Open, approach & align, close, lift
            self.open_gripper_srv()
            self.refine_pose_with_ee_camera_dx_dy(cx, cy, PICK_Z1)   # aligns to that brick at given height
            self.refine_pose_with_ee_camera_dx_dy(cx, cy, PICK_Z1)   # aligns to that brick at given height
            self.move_to_height(PICK_Z3, is_cartesian=False)
            self.close_gripper_srv()
            self.move_to_height(DROP_Z, is_cartesian=False)

            # Go to drop pose and release
            self.send_move_request(drop_pose, is_cartesian=False)
            self.open_gripper_srv()

            # Move up to travel height
            self.move_to_height(HOVER_Z, is_cartesian=False)

            # If there is a next brick, go directly above it at hover height
            if i + 1 < len(bricks):
                nb = bricks[i+1]
                n_cx = nb.get("cx_m")
                n_cy = nb.get("cy_m")
                if n_cx is not None and n_cy is not None:
                    above_next = self.face_down_pose(n_cx, n_cy, HOVER_Z, yaw)
                    self.log_with_time('info',
                        f"Moving above next brick #{i+2} at ({n_cx:.3f}, {n_cy:.3f})")
                    self.send_move_request(above_next, is_cartesian=False)

        self.log_with_time('info', "collect_bricks: done.")


    def open_gripper_srv(self):
        self.load_robot_params()
        self.current_state = STATE_NONE
        self.move_servo_to_angle(self.GRIP_FORCE_OPEN)

    def close_gripper_srv(self):
        self.load_robot_params()
        self.current_state = STATE_COLLECTED_CHIP
        self.move_servo_to_angle(self.GRIP_FORCE_CLOSE)

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

    def make_pose_xyz_quat(self,x: float, y: float, z: float,
                        qx: float, qy: float, qz: float, qw: float) -> Pose:
        """Create a Pose from numeric xyz and quaternion (qx, qy, qz, qw)."""
        # (Optional) normalize quaternion to be safe
        mag = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if mag > 0:
            qx, qy, qz, qw = (qx/mag, qy/mag, qz/mag, qw/mag)

        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        p.orientation.x = float(qx)
        p.orientation.y = float(qy)
        p.orientation.z = float(qz)
        p.orientation.w = float(qw)
        return p

    def refine_pose_with_ee_camera_dx_dy(self, cx: float, cy: float, height: float = 0.15):
        """
        Pick the brick whose published center is closest to (cx, cy) [meters, base_link]
        and move the EE by that brick's (dx, dy) plus orient EE by its angle.

        Requires self.latest_bricks from 'brick_top_infos'.
        Falls back to computing center from current EE pose if cx/cy per-brick aren't published.
        """
        # Need current EE pose
        ee_pose = self.get_current_ee_pose()
        if ee_pose is None:
            self.log_with_time('error', "Cannot read current end-effector pose.")
            return

        if not self.latest_bricks:
            self.log_with_time('warn', "No bricks available from 'brick_top_infos'.")
            return

        # Decide a center for each brick (prefer broker-provided cx/cy, otherwise compute)
        def brick_center_m(b):
            if b.get("cx_m") is not None and b.get("cy_m") is not None:
                return b["cx_m"], b["cy_m"]
            # fallback: EE + (dx,dy) (mm->m). dx/dy are defined as "how much to move EE"
            return (ee_pose.position.x + b["dx_mm"] / 1000.0,
                    ee_pose.position.y + b["dy_mm"] / 1000.0)

        # Pick nearest brick to requested (cx, cy)
        best = None
        best_d2 = float("inf")
        for b in self.latest_bricks:
            bx, by = brick_center_m(b)
            if bx is None or by is None:
                continue
            d2 = (bx - cx) ** 2 + (by - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = b

        if best is None:
            self.log_with_time('warn', "No valid brick centers to compare.")
            return

        # Convert moves mm -> m
        dx_m = best["dx_mm"] / 1000.0
        dy_m = best["dy_mm"] / 1000.0
        rel_rotation_deg = -float(best["angle_deg"])   # same convention as before

        # Current orientation -> set face-down roll/pitch, adjust yaw by rel rotation
        quat = [
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        ]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        roll = -np.pi    # face down
        pitch = 0.0
        current_yaw_deg = float(np.degrees(yaw)) % 360.0
        new_yaw_deg = (current_yaw_deg + rel_rotation_deg) % 360.0

        # Build corrected pose
        corrected = Pose()
        corrected.position.x = ee_pose.position.x + dx_m
        corrected.position.y = ee_pose.position.y + dy_m
        corrected.position.z = float(height)

        yaw_rad = np.radians(new_yaw_deg)
        qf = tf_transformations.quaternion_from_euler(roll, pitch, yaw_rad)
        corrected.orientation.x = qf[0]
        corrected.orientation.y = qf[1]
        corrected.orientation.z = qf[2]
        corrected.orientation.w = qf[3]

        # Go!
        self.send_move_request(corrected, is_cartesian=False)

        # Optional: reflect final pose back to GUI
        final_pose = self.get_current_ee_pose()
        if final_pose:
            pos = final_pose.position
            ori = final_pose.orientation
            pos_str = f"({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
            ori_str = f"({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})"

            def update_entries():
                self.translation_entry.delete(0, tk.END)
                self.translation_entry.insert(0, pos_str)
                self.rotation_entry.delete(0, tk.END)
                self.rotation_entry.insert(0, ori_str)
            self.gui_queue.put(update_entries)


    def refine_pose_with_ee_camera(self, height=0.15):
        if None in (self.top_dx_mm, self.top_dy_mm, self.top_angle_deg, self.top_est_height_mm):
            self.log_with_time('warn', "No brick info from end-effector camera.")
            return

        dx_cam = self.top_dx_mm / 1000.0
        dy_cam = self.top_dy_mm / 1000.0
        rel_rotation_deg = -self.top_angle_deg  # from vision

        ee_pose = self.get_current_ee_pose()
        if ee_pose is None:
            self.log_with_time('error', "Cannot read current end-effector pose.")
            return

        quat = [
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        ]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        roll = -np.pi   # face down
        pitch = 0       # face down
        current_yaw_deg = np.degrees(yaw) % 360

        # --- Decide behavior based on height ---
        if height == 0:
            # Keep yaw and z as-is
            new_yaw_deg = current_yaw_deg
            target_z = ee_pose.position.z
            print('info', f"[EE Refine] Height=0 → Keeping yaw ({current_yaw_deg:.2f}°) and Z ({target_z:.3f} m)")
        else:
            # Apply correction both in yaw and Z
            new_yaw_deg = (current_yaw_deg + rel_rotation_deg) % 360
            target_z = height
            print('info',
                f"[EE Rotation] Current yaw: {current_yaw_deg:.2f}°, "
                f"Relative: {rel_rotation_deg:.2f}°, "
                f"Target: {new_yaw_deg:.2f}° | New Z={target_z:.3f} m")

        # Transform dx/dy to base frame
        R = tf_transformations.quaternion_matrix(quat)[0:3, 0:3]
        offset_base = R @ np.array([dx_cam, dy_cam, 0.0])

        corrected_pose = Pose()
        corrected_pose.position.x = ee_pose.position.x + dx_cam
        corrected_pose.position.y = ee_pose.position.y + dy_cam
        corrected_pose.position.z = target_z

        # Orientation
        yaw_rad = np.radians(new_yaw_deg)
        final_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw_rad)
        corrected_pose.orientation.x = final_quat[0]
        corrected_pose.orientation.y = final_quat[1]
        corrected_pose.orientation.z = final_quat[2]
        corrected_pose.orientation.w = final_quat[3]

        self.send_move_request(corrected_pose, is_cartesian=False)
        
        # --- Feedback ---
        final_pose = self.get_current_ee_pose()
        if final_pose:
            _, _, final_yaw = tf_transformations.euler_from_quaternion([
                final_pose.orientation.x,
                final_pose.orientation.y,
                final_pose.orientation.z,
                final_pose.orientation.w,
            ])
            final_yaw_deg = np.degrees(final_yaw) % 360
            print('info', f"[EE Rotation] Final yaw: {final_yaw_deg:.2f}°")

            # Update GUI
            pos_str = f"({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f})"
            ori_str = f"({final_pose.orientation.x:.3f}, {final_pose.orientation.y:.3f}, {final_pose.orientation.z:.3f}, {final_pose.orientation.w:.3f})"

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
        """Thread-safe label update. Always runs in the Tk main thread."""
        def update():
            if self.arm_is_moving:
                self.status_label.config(text="Arm Is Moving", fg="red")
            else:
                self.status_label.config(text="Waiting", fg="black")

        # Schedule GUI update to main thread via the queue
        self.gui_queue.put(update)



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
                self.joint_states = msg
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
        # self.update_gui_once()
        future.add_done_callback(_on_response)

        if not done_event.wait(timeout=timeout_sec):
            self.log_with_time('error' ,"Service call timed out")
            return None

        self._update_status_label()
        # self.update_gui_once()
        
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
            pose.position = Point(
                x=float(trans.transform.translation.x),
                y=float(trans.transform.translation.y),
                z=float(trans.transform.translation.z),
            )
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

    def align_marker_with_image(
        self,
        axis_in_marker: str = "x",   # "x" or "y" — which marker axis to align in the image
        target: str = "horizontal",  # "horizontal" or "vertical" final orientation in the image
        sign_correction: int = +1    # flip to -1 if rotation goes the wrong way on your rig
    ):
        """
        Rotate the end-effector (camera) about its viewing axis so the chosen marker axis
        appears perfectly horizontal/vertical in the camera image.

        Frames (ROS optical): +X right, +Y down, +Z forward (into the scene).

        Steps:
        1) Use marker pose in the camera frame (self.pose_in_camera).
        2) Project the chosen marker axis onto the image plane (X–Y of camera frame).
        3) Compute its angle in the image.
        4) Rotate the EE around the viewing axis (approx. a pure yaw update) by -angle
            (or to ±90° for vertical) and keep XYZ fixed.

        Notes:
        * If the rotation direction is inverted for your camera mounting, set sign_correction=-1.
        * This assumes the camera is rigidly fixed to the EE; a yaw about the EE roughly
            equals a roll about the camera Z (optical axis).
        """
        # --- 0) sanity ---
        if self.pose_in_camera is None:
            self.log_with_time('warn', "No ArUco pose in camera frame yet.")
            return

        # --- 1) marker pose in camera frame ---
        pose_c = self.pose_in_camera

        # --- 2) rotation matrix of marker in camera frame ---
        q = [pose_c.orientation.x, pose_c.orientation.y, pose_c.orientation.z, pose_c.orientation.w]
        R = tf_transformations.quaternion_matrix(q)[0:3, 0:3]  # marker axes expressed in camera frame

        idx = {"x": 0, "y": 1, "z": 2}[axis_in_marker.lower()]
        a_cam = R[:, idx]           # chosen marker axis, in camera coords

        # Project to the image plane: use (X_right, Y_down), drop Z
        ax, ay = float(a_cam[0]), float(a_cam[1])

        # If the axis is nearly perpendicular to the image plane, bail
        if ax*ax + ay*ay < 1e-8:
            self.log_with_time('warn', "Marker axis nearly along camera Z; angle undefined.")
            return

        # --- 3) current angle of that axis in the image (radians).
        # angle 0 = pointing to the right (horizontal), positive = clockwise (since Y is down).
        angle = math.atan2(ay, ax)  # [-pi, pi]

        # Desired angle in the image:
        if target.lower() == "horizontal":
            angle_target = 0.0
        elif target.lower() == "vertical":
            # vertical means pointing straight down (or up). In ROS optical, +Y is DOWN,
            # so choose +90° (down) as canonical. If you want up, use -90°.
            angle_target = math.pi / 2.0
        else:
            self.log_with_time('warn', f"Unknown target '{target}', use 'horizontal' or 'vertical'.")
            return

        # The delta to apply to the camera about its viewing axis (image rotation)
        delta_img = (angle_target - angle)

        # Optional: wrap to [-pi, pi] for minimal rotation
        while delta_img > math.pi:
            delta_img -= 2*math.pi
        while delta_img < -math.pi:
            delta_img += 2*math.pi

        delta_img *= float(sign_correction)

        # --- 4) apply that as a yaw change to the EE pose in base_link ---
        cur = self.get_current_ee_pose()
        if cur is None:
            self.log_with_time('error', "Current EE pose unavailable.")
            return

        # Keep position, keep roll/pitch face-down if you like, only adjust yaw.
        roll, pitch, yaw = tf_transformations.euler_from_quaternion([
            cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w
        ])

        # If your tool is face-down (-pi roll) and camera optical Z aligns with EE -Z,
        # a change in camera image rotation typically maps to a change in EE yaw by the same sign.
        new_yaw = yaw + delta_img

        q_new = tf_transformations.quaternion_from_euler(roll, pitch, new_yaw)
        pose = self.create_pose(
            (cur.position.x, cur.position.y, cur.position.z),
            (q_new[0], q_new[1], q_new[2], q_new[3])
        )
        self.send_move_request(pose, is_cartesian=False)

        self.log_with_time(
            'info',
            f"Aligned marker to {target} via Δyaw={math.degrees(delta_img):+.1f}° (img angle {math.degrees(angle):+.1f}°)."
        )



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
            # Transform the pose into target_frame
            transformed_pose = do_transform_pose(stamped_pose_in.pose, transform)

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

    def send_move_request(self, pose, is_cartesian=True, via_points=None) -> bool:
        self.arm_is_moving = True
        self.update_status_label()
        # pose_goal = PoseStamped() 
        # pose_goal.header.frame_id = "base_link"
        # pose_goal.pose = Pose(position = pose.position, orientation = pose.orientation)
        pose = Pose(position = pose.position, orientation = pose.orientation)
        print ("starting to move")
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        motion_type = "Cartesian" if is_cartesian else "Joint-space"
        if not (-0.5 <= pose.position.x <= 0.5 and -0.54 <= pose.position.y <= 0.6 and 0.06 <= pose.position.z <= 0.55):
            print(f"[{now}] 🚫 Refused Motion: {motion_type} target out of bounds! Target: pos=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) ori={pose.orientation}")
            self.arm_is_moving = False
            return False

        request = self.get_move_request(pose, is_cartesian=is_cartesian)
        # Send the request


        response = self.call_service_blocking(self.move_arm_client, request,timeout_sec=4448.0)
        print("Got response:", response)

        if response is None:
            print(f"[{now}] ❌ Motion Failed: No response from service for {motion_type} move to pos=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) ori={pose.orientation}")
            return False
        is_ok = response.success
        if is_ok:
            print(f"[{now}] ✅ Motion Succeeded: {motion_type} move to pos=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) ori={pose.orientation}")
        else:
            print(f"[{now}] ❌ Motion Failed: {motion_type} move to pos=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) ori={pose.orientation} | Message: {response.message}")
        return is_ok

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


    def get_joint_angle(self, joint_name: str):
        """Return the angle (in degrees) for a given joint name, or None if not found."""
        if not self.joint_states:
            return None
        try:
            idx = self.joint_states.name.index(joint_name)
            return np.degrees(self.joint_states.position[idx])
        except ValueError:
            self.get_logger().warn(f"{joint_name} not found in joint states")
            return None

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
    debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    print("Debugger connected.")
   
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
