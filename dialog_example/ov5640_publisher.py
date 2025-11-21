#!/home/alon/venv/bin/python

import sys
print(f"[DEBUG] Running with Python interpreter: {sys.executable}")
import debugpy
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
import os
from ultralytics import YOLO
from glob import glob
import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState  # already available in ROS 2
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout
import tf_transformations
import torch
import numpy as np

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

CAMERA_INDEX = 8  # Change to 1, 2, etc., if needed
class OV5640Publisher(Node):
    def __init__(self):
        super().__init__('ov5640_publisher')
        self.counter = 0
        self.joint_states = None  # Initialize joint states to None
        MARKER_SIDE_MM = 70.0  # ArUco marker side length (7 cm)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.info_publisher = self.create_publisher(Float32MultiArray, 'brick_top_info', 10)
        self.infos_publisher = self.create_publisher(Float32MultiArray, 'brick_top_infos', 10)  # NEW (batched)
        self.board_end_points_pub = self.create_publisher(Float32MultiArray, 'board_end_points', 10)

        self.setup_aruco()

        self.aruco_corners_pub = self.create_publisher(Float32MultiArray, 'aruco_corners', 10)

        self.joint_positions = [None, None, None, None, None, None]  # store last positions
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.bridge = CvBridge()

        self.latest_frame = None
        self.create_subscription(
            Image,
            '/ov5640/image_raw',   
            self.image_callback,
            10)


        self.capture_folder = "captured_images"
        os.makedirs(self.capture_folder, exist_ok=True)
        self.capture_count = 0

        self.model_chip = YOLO("/home/alon/Documents/Projects/rchip_table_train/runs/obb/train6/weights/best.pt")
        self.model_board = YOLO("/home/alon/Documents/Projects/board_top_train/runs/obb/train4/weights/best.pt")

        self.handle_camera_calibration()

    def image_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not hasattr(self, "map1"):  # run calibration only once
                self.get_logger().info("Running camera calibration with first received frame")
                self.handle_camera_calibration()
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")



    def get_ee_xy(self):
        try:
            # base_link <- ee_link transform (now)
            trans = self.tf_buffer.lookup_transform(
                'base_link', 'ee_link', rclpy.time.Time())
            x = float(trans.transform.translation.x)  # meters
            y = float(trans.transform.translation.y)  # meters
            return x, y
        except Exception as e:
            self.get_logger().warn(f"TF lookup (base_link<-ee_link) failed: {e}")
            return None, None
        
    def handle_camera_calibration(self):
        # Wait for a frame from subscriber
        if self.latest_frame is None:
            self.get_logger().warn("No frame yet for calibration — using black frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame = self.latest_frame.copy()

        # === Load calibration data ===
        calib_path = "/home/alon/ros_ws/src/dialog_example/dialog_example/camera_calibration.npz"
        if not os.path.exists(calib_path):
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            rclpy.shutdown()

        data = np.load(calib_path)
        self.K = data["K"]
        self.dist = data["dist"]

        # Precompute undistortion map
        if self.latest_frame is None:
            self.get_logger().warn("No frame received yet")
            return
        frame = self.latest_frame.copy()

        h, w = frame.shape[:2]
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)
        self.optical_center = (self.new_K[0, 2], self.new_K[1, 2])  # (cx, cy)

    def _affine_to_3x3(self,M2x3: np.ndarray) -> np.ndarray:
        A = np.eye(3, dtype=np.float32)
        A[:2, :3] = M2x3.astype(np.float32)
        return A

    def _compose_affine(self,Mb_2x3: np.ndarray, Ma_2x3: np.ndarray) -> np.ndarray:
        """Compose Mb∘Ma (apply Ma first, then Mb). Returns 2x3."""
        Mb = self._affine_to_3x3(Mb_2x3)
        Ma = self._affine_to_3x3(Ma_2x3)
        Mc = Mb @ Ma
        return Mc[:2, :3]

    def _apply_affine_points(self, ptsNx2: np.ndarray, M2x3: np.ndarray) -> np.ndarray:
        """
        Apply a 2x3 affine transform to N 2D points.

        ptsNx2: array-like of shape (N, 2)
        M2x3:   affine matrix of shape (2, 3)
        returns: transformed points of shape (N, 2)
        """
        pts = np.asarray(ptsNx2, dtype=np.float32).reshape(-1, 2)
        M = np.asarray(M2x3, dtype=np.float32).reshape(2, 3)

        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])        # (N,3)
        out = pts_h @ M.T                     # (N,2)
        return out



    def joint_state_callback(self, msg: JointState):
        self.joint_states = msg

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

    def offset_cam_to_base(self, dx_mm, dy_mm, joint1_deg, joint6_deg):
        """
        Convert camera-plane offset to base frame using J1 + J6 + image-plane vector angle.
        dx_mm, dy_mm: offset in camera frame (mm) - can be None
        joint1_deg, joint6_deg: angles in degrees - can be None
        Returns: (dx_base_mm, dy_base_mm) or (None, None) if invalid inputs
        """

        joint1_deg = joint6_deg = 0.0 #these were taken into account when rotating the frame
        # Validate inputs
        inputs = [dx_mm, dy_mm, joint1_deg, joint6_deg]
        if any(val is None for val in inputs):
            return None, None

        try:
            # Distance and direction in camera plane
            dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)
            alpha_cam = np.degrees(np.arctan2(dy_mm, dx_mm))  # angle from EE's +X axis

            # Total yaw from base frame
            total_yaw_deg = joint1_deg + joint6_deg + alpha_cam
            total_yaw_rad = np.radians(total_yaw_deg)

            # Map to base frame
            dx_base = dist_mm * np.cos(total_yaw_rad)
            dy_base = dist_mm * np.sin(total_yaw_rad)

            return dx_base, dy_base

        except Exception:
            return None, None

    def calculate_brick_yaw(self,points):
        """
        Calculate the brick yaw angle (long edge orientation) from 4 OBB points.

        Args:
            points (np.ndarray): Array of shape (4, 2) representing the OBB vertices.

        Returns:
            float: Yaw angle in degrees, constrained to [-90, 90].
        """
        if points.shape != (4, 2) or np.isnan(points).any():
            return None

        # Determine which edge is longer
        edge1 = np.linalg.norm(points[1] - points[0])
        edge2 = np.linalg.norm(points[2] - points[1])

        # Get dx, dy for the longer edge
        if edge1 >= edge2:
            dx_edge, dy_edge = points[1][0] - points[0][0], points[1][1] - points[0][1]
        else:
            dx_edge, dy_edge = points[2][0] - points[1][0], points[2][1] - points[1][1]

        # Calculate angle
        angle_rad = np.arctan2(dy_edge, dx_edge)
        angle_deg = float(np.degrees(angle_rad))

        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        return angle_deg

    def compensate_angle(self,detected_angle, total_angle):
        """
        Compensate the brick yaw angle for preprocessing transforms.
        
        detected_angle : float (degrees) - from calculate_brick_yaw(points)
        total_angle    : float (degrees) - J1 + J6 rotation applied to image
        """
        # Undo last rotation
        corrected = detected_angle + total_angle

        # Undo horizontal flip
        corrected = -corrected

        # Undo initial 90° clockwise rotation
        corrected += 90.0

        # Normalize to [-90, 90]
        while corrected > 90:
            corrected -= 180
        while corrected < -90:
            corrected += 180

        return corrected

    def draw_optical_center(self, image):
        """
        Draws a red crosshair and label at the optical center of the camera.

        Args:
            image (np.ndarray): The image to annotate (typically annotated_frame)
        """
        if not hasattr(self, 'optical_center'):
            return  # skip if not yet initialized

        optical_x = int(self.optical_center[0])
        optical_y = int(self.optical_center[1])

        # Draw cross
        cv2.drawMarker(
            image,
            (optical_x, optical_y),
            (0, 0, 255),  # Red
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
            line_type=cv2.LINE_AA
        )

        # Add label
        cv2.putText(
            image,
            "Optical Center",
            (int(optical_x) + 10, int(optical_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    def _rotate_image(self, img, angle_deg):
        """Rotate image around its center. Returns (rotated_img, M, M_inv)."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        M_inv = cv2.invertAffineTransform(M)  # 2x3
        return rotated, M, M_inv

    def _has_obb(self, yolo_result):
        """Check if YOLO result has at least one OBB polygon."""
        return (
            hasattr(yolo_result, "obb")
            and yolo_result.obb is not None
            and getattr(yolo_result.obb, "xyxyxyxy", None) is not None
            and len(yolo_result.obb.xyxyxyxy) > 0
        )

    def detect_aruco(self, frame):
        """Detect ArUco markers using only DICT_4X4_250."""
        # assumes setup_aruco() already did:
        # self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        # self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # self.get_logger().info(f"Detected ArUco (DICT_4X4_250): IDs={ids.flatten().tolist()}")
            return corners, ids, rejected

        # nothing found
        return [], None, []

    # ---------------- ArUco helpers ----------------
    def setup_aruco(self):
        """Initialize the ArUco detector with predefined dictionary and parameters."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def estimate_pose(self, corners):
        """Estimate pose from ArUco corners using calibration if available.
        Returns (tvec, rvec, quat) or (None, None, None)."""
        obj_points = np.array([
            [-0.035,  0.035, 0],
            [ 0.035,  0.035, 0],
            [ 0.035, -0.035, 0],
            [-0.035, -0.035, 0]
        ], dtype=np.float32)

        c = corners.reshape((4, 2)).astype(np.float32)

        if hasattr(self, "K") and self.K is not None:
            ret, rvec, tvec = cv2.solvePnP(
                obj_points, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                quat = tf_transformations.quaternion_from_matrix(
                    np.vstack([np.hstack([R, [[0],[0],[0]]]), [0,0,0,1]])
                )
                return tvec, rvec, quat
        return None, None, None

    def draw_aruco_poly(self, frame, pts4x2, marker_id):
        """
        Draws the 4 edges of the ArUco marker polygon.
        The edge whose midpoint has the lowest Y (lowest in image space)
        is drawn in a distinct color.
        """
        pts = pts4x2.reshape(-1, 2).astype(int)
        n = len(pts)
        if n != 4:
            return

        # Compute midpoints of edges and find the one with lowest Y
        midpoints = []
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            midpoints.append(mid)
        lowest_idx = int(np.argmax([m[1] for m in midpoints]))  # largest Y = lowest point on image

        # Draw each edge; highlight the lowest one
        for i in range(n):
            p1, p2 = tuple(pts[i]), tuple(pts[(i + 1) % n])
            color = (0, 200, 255)  # normal yellow-orange
            if i == lowest_idx:
                color = (0, 0, 255)  # red for lowest edge
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        # Draw the ID label near the center
        cxy = pts.mean(axis=0).astype(int)
        cv2.putText(frame, f"id:{int(marker_id)}",
                    (int(cxy[0]) + 6, int(cxy[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)

    def draw_aruco(self, frame, corners, ids, rvecs, tvecs):
        """Draw markers, IDs, and axes if calibration available."""
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if hasattr(self, "K") and self.K is not None and len(rvecs) == len(tvecs):
            for (rvec, tvec) in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, 0.05)

    def detect_and_publish_aruco(self, frame_unflipped, annotated_frame, M_forward, total_angle, ee_x, ee_y):
        """
        Detect ArUco on the R90/pre-flip snapshot, map corners to the processed view,
        draw overlays, and publish the SAME info format as the YOLO pipeline.
        Assumes a 7 cm (70 mm) square marker.
        """
        MARKER_SIDE_MM = 70.0

        corners_list, ids, _ = self.detect_aruco(frame_unflipped)
        if ids is None or len(ids) == 0:
            return

        # For drawing/scales
        frame_h, frame_w = annotated_frame.shape[:2]
        center_x, center_y = frame_w / 2.0, frame_h / 2.0

        batched_corners = []  # for aruco_corners
        all_infos = []        # for brick_top_infos (batched)

        for i, marker_id in enumerate(ids.flatten().tolist()):
            # 1) Corners in snapshot → map to processed (display) frame
            corners_un = corners_list[i].reshape(4, 2).astype(np.float32)
            corners_proc = self._apply_affine_points(corners_un, M_forward)

            # 2) Draw polygon + id
            self.draw_aruco_poly(annotated_frame, corners_proc, marker_id)

            # 4) Center in processed coords
            cx = float(corners_proc[:, 0].mean())
            cy = float(corners_proc[:, 1].mean())

            # 5) Yaw (processed coords) → compensate preprocessing
            angle_from_frame = self.calculate_brick_yaw(corners_proc)
            if angle_from_frame is None:
                continue
            angle_deg = self.compensate_angle(angle_from_frame, total_angle)

            # 6) Pixel → mm using known marker side (70 mm)
            cyc = np.vstack([corners_proc, corners_proc[0:1]])
            edge_px = [np.linalg.norm(cyc[k+1] - cyc[k]) for k in range(4)]
            # self.get_logger().info(
            #     f"[ArUco id {marker_id}] edges(px): "
            #     f"{edge_px[0]:.1f}, {edge_px[1]:.1f}, {edge_px[2]:.1f}, {edge_px[3]:.1f}"
            # )
            pixel_long_edge = max(edge_px)
            if pixel_long_edge <= 0:
                continue
            pixel_to_mm = MARKER_SIDE_MM / pixel_long_edge


            board_points = self.compute_board_end_points(
                corners_proc,
                mm_along_center_to_red=40.0,   # 4 cm away from marker center
                mm_half_board_length=150.0,    # 15 cm each side
                pixel_to_mm=pixel_to_mm
            )

            if board_points is not None:
                (xL, yL), (xR, yR) = board_points
                cv2.circle(annotated_frame, (int(xL), int(yL)), 6, (255, 255, 0), -1)
                cv2.circle(annotated_frame, (int(xR), int(yR)), 6, (255, 255, 0), -1)
                cv2.line(annotated_frame, (int(xL), int(yL)), (int(xR), int(yR)), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, "Board", (int((xL+xR)/2), int((yL+yR)/2)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                self.publish_board_end_points(board_points, pixel_to_mm, frame_w, frame_h)


            # 7) Offsets from image center (pixels → mm)
            dx_px = cx - center_x
            dy_px = cy - center_y
            pixel_dist = float(np.hypot(dx_px, dy_px))
            dx_mm = dx_px * pixel_to_mm
            dy_mm = dy_px * pixel_to_mm

            # 8) Draw the center line + distance/angle label (same as model path)
            cv2.line(
                annotated_frame,
                (int(center_x), int(center_y)),
                (int(cx), int(cy)),
                (255, 0, 0),  # same color as model
                1
            )
            mid_x = int((center_x + cx) / 2.0)
            mid_y = int((center_y + cy) / 2.0)
            dir_angle_deg = float(np.degrees(np.arctan2(dy_mm, dx_mm)))
            dist_mm = float(np.hypot(dx_mm, dy_mm))
            cv2.putText(
                annotated_frame,
                f"{dist_mm:.1f} mm @ {dir_angle_deg:.1f} deg",
                (mid_x - 50, mid_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
            )

            # 9) Camera-plane mm → base frame
            dx_base, dy_base = self.offset_cam_to_base(dx_mm, dy_mm, joint1_deg=0.0, joint6_deg=0.0)
            if dx_base is None or dy_base is None:
                continue
            dx_pub = -dx_base
            dy_pub = -dy_base

            # 10) Height estimate (same heuristic as model, with 70 mm side)
            focal_length_px = (frame_w / 2.0) / np.tan(np.radians(72.0 / 2.0))
            marker_half_px = pixel_long_edge / 2.0
            hypotenuse_px = float(np.hypot(marker_half_px, pixel_dist))
            est_height_mm = (focal_length_px * MARKER_SIDE_MM) / hypotenuse_px if hypotenuse_px > 0 else 0.0

            # 11) Absolute base XY (meters)
            cx_m = (ee_x if ee_x is not None else float('nan')) + (dx_pub / 1000.0)
            cy_m = (ee_y if ee_y is not None else float('nan')) + (dy_pub / 1000.0)

            # 12) Per-detection publish (same as model)
            info_msg = Float32MultiArray()
            info_msg.data = [dx_pub, dy_pub, float(angle_deg), float(est_height_mm)]
            self.info_publisher.publish(info_msg)

            # 13) Accumulate batched (same as model)
            all_infos.extend([dx_pub, dy_pub, float(angle_deg), float(est_height_mm), cx_m, cy_m])

            # 14) Also publish corners (processed coords) for consumers that use them
            flat = [float(marker_id)]
            flat.extend(corners_proc.reshape(-1).astype(float).tolist())
            batched_corners.extend(flat)

        # Batched corners
        if batched_corners:
            n = len(batched_corners) // 9
            msg = Float32MultiArray()
            msg.layout = MultiArrayLayout(
                dim=[
                    MultiArrayDimension(label='markers', size=n, stride=n*9),
                    MultiArrayDimension(label='fields',  size=9, stride=9),
                ],
                data_offset=0,
            )
            msg.data = batched_corners
            self.aruco_corners_pub.publish(msg)

        # Batched model-compatible infos
        if len(all_infos) > 0:
            n = len(all_infos) // 6
            batched = Float32MultiArray()
            batched.layout = MultiArrayLayout(
                dim=[
                    MultiArrayDimension(label='objects', size=n, stride=n*6),
                    MultiArrayDimension(label='fields',  size=6, stride=6),
                ],
                data_offset=0,
            )
            batched.data = all_infos
            self.infos_publisher.publish(batched)


    def get_aruco_red_edge_midpoint(self, pts4x2: np.ndarray):
        """
        Given a 4x2 array of ArUco corners in the SAME coordinate system used for drawing
        (i.e., the processed/annotated frame), return the midpoint (mx, my) of the 'red' edge,
        which is defined as the edge whose midpoint has the lowest (largest) Y.

        Args:
            pts4x2 (np.ndarray): shape (4, 2), float or int.

        Returns:
            (mx, my): tuple[float, float] midpoint of the red (lowest) edge in image pixels.
            Also returns the endpoints ((x1,y1), (x2,y2)) of that edge for convenience.
        """
        pts = np.asarray(pts4x2, dtype=float).reshape(-1, 2)
        if pts.shape != (4, 2):
            return None, None, None  # invalid

        # Compute midpoints of each consecutive edge (0-1,1-2,2-3,3-0)
        mids = []
        edges = []
        for i in range(4):
            p1, p2 = pts[i], pts[(i + 1) % 4]
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            mids.append(mid)
            edges.append((p1, p2))

        # "Red" edge is the one with the largest Y midpoint (lowest on the image)
        lowest_idx = int(np.argmax([m[1] for m in mids]))
        (mx, my) = mids[lowest_idx]
        (e1, e2) = edges[lowest_idx]
        return (mx, my), e1, e2

    def publish_board_end_points(self, board_points, pixel_to_mm, frame_w, frame_h):
        """
        Publish dx/dy (mm) of the two board endpoints relative to image center.

        Args:
            board_points: ((x_left, y_left), (x_right, y_right)) in pixels
            pixel_to_mm:  mm per pixel
            frame_w, frame_h: frame size
        """
        if board_points is None or pixel_to_mm is None:
            return

        (xL, yL), (xR, yR) = board_points
        cx, cy = frame_w / 2.0, frame_h / 2.0

        dxL_mm = (xL - cx) * pixel_to_mm
        dyL_mm = (yL - cy) * pixel_to_mm
        dxR_mm = (xR - cx) * pixel_to_mm
        dyR_mm = (yR - cy) * pixel_to_mm

        msg = Float32MultiArray()
        msg.data = [dxL_mm, dyL_mm, dxR_mm, dyR_mm]
        self.board_end_points_pub.publish(msg)



    def compute_board_end_points(self,
                                pts4x2: np.ndarray,
                                mm_along_center_to_red: float = 40.0,
                                mm_half_board_length: float = 150.0,   # <-- accept this name
                                pixel_to_mm: float = None,
                                marker_side_mm: float = 70.0):
        """
        Returns two pixel coords for the board endpoints, located ±mm_half_board_length
        along the perpendicular to the center→red-midpoint direction, at a point
        mm_along_center_to_red away from the marker center.
        """
        if pts4x2 is None or np.asarray(pts4x2).shape != (4, 2) or np.isnan(pts4x2).any():
            return None

        pts = np.asarray(pts4x2, dtype=float)
        center = pts.mean(axis=0)

        red_mid, _, _ = self.get_aruco_red_edge_midpoint(pts)
        if red_mid is None:
            return None

        v = np.array(red_mid) - center
        v_norm = np.hypot(v[0], v[1])
        if v_norm < 1e-6:
            return None
        u = v / v_norm

        if pixel_to_mm is None:
            cyc = np.vstack([pts, pts[0:1]])
            edges_px = [np.linalg.norm(cyc[k+1] - cyc[k]) for k in range(4)]
            px_long = max(edges_px)
            if px_long <= 0:
                return None
            pixel_to_mm = marker_side_mm / px_long

        px_along = mm_along_center_to_red / pixel_to_mm
        p_origin = center + u * px_along

        u_perp = np.array([-u[1], u[0]], dtype=float)
        px_perp = mm_half_board_length / pixel_to_mm

        p_left  = p_origin - u_perp * px_perp
        p_right = p_origin + u_perp * px_perp
        return (float(p_left[0]), float(p_left[1])), (float(p_right[0]), float(p_right[1]))


    def merge_obb_results(self, r1, r2):
        """
        Merge two YOLO OBB results objects into a single synthetic result object.
        Polygons, classes, and confidences are concatenated safely.
        """

        # --- 1) Extract polygons ---
        polys1 = r1.obb.xyxyxyxy if hasattr(r1.obb, "xyxyxyxy") else []
        polys2 = r2.obb.xyxyxyxy if hasattr(r2.obb, "xyxyxyxy") else []

        # convert to list of numpy arrays (4x2)
        polys1 = [p.cpu().numpy().reshape(4, 2) for p in polys1]
        polys2 = [p.cpu().numpy().reshape(4, 2) for p in polys2]

        merged_polys = polys1 + polys2

        # --- 2) Extract classes ---
        cls1 = r1.obb.cls if hasattr(r1.obb, "cls") else None
        cls2 = r2.obb.cls if hasattr(r2.obb, "cls") else None

        if cls1 is None or len(cls1) == 0:
            cls1 = torch.tensor([], dtype=torch.float32)
        if cls2 is None or len(cls2) == 0:
            cls2 = torch.tensor([], dtype=torch.float32)

        merged_cls = torch.cat([cls1.cpu(), cls2.cpu()]) if len(cls1) + len(cls2) > 0 else torch.tensor([])

        # --- 3) Extract confidences ---
        conf1 = r1.obb.conf if hasattr(r1.obb, "conf") else None
        conf2 = r2.obb.conf if hasattr(r2.obb, "conf") else None

        if conf1 is None or len(conf1) == 0:
            conf1 = torch.tensor([], dtype=torch.float32)
        if conf2 is None or len(conf2) == 0:
            conf2 = torch.tensor([], dtype=torch.float32)

        merged_conf = torch.cat([conf1.cpu(), conf2.cpu()]) if len(conf1) + len(conf2) > 0 else torch.tensor([])

        # --- 4) Create a new synthetic “results-like” object ---
        class SyntheticOBB:
            pass

        merged = SyntheticOBB()
        merged.obb = SyntheticOBB()

        # Convert merged polys to tensors shaped like YOLO expects
        if len(merged_polys) > 0:
            polys_tensor = torch.tensor(np.array(merged_polys)).reshape(-1, 4, 2)
        else:
            polys_tensor = torch.zeros((0, 4, 2), dtype=torch.float32)

        merged.obb.xyxyxyxy = polys_tensor
        merged.obb.cls = merged_cls
        merged.obb.conf = merged_conf

        return merged





    def timer_callback(self):

        if self.latest_frame is None:
            self.get_logger().warn("No frame received yet")
            return

        if not hasattr(self, "map1") or not hasattr(self, "map2"):
            self.get_logger().warn("Calibration maps not ready yet")
            return

        # Grab frame
        text_color = (255, 0, 0)

        if self.latest_frame is None:
            self.get_logger().warn("No frame received yet")
            return

        frame = self.latest_frame.copy()
        # Undistort/rectify
        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Align camera orientation (keep as before)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_unflipped = frame.copy()

        frame = cv2.flip(frame, 1)

        # --- Read joint angles ---
        j1 = self.get_joint_angle('joint_1')
        j2 = self.get_joint_angle('joint_2')
        j3 = self.get_joint_angle('joint_3')
        j4 = self.get_joint_angle('joint_4')
        j5 = self.get_joint_angle('joint_5')
        j6 = self.get_joint_angle('joint_6')

        if j1 is None or j6 is None:
            self.get_logger().warn("Joint angles not available, skipping frame processing")
            j1 = j2 = j3 = j4 = j5 = j6 = 0.0
            # return

        # Total rotation for the image (use J1 + J6)
        total_angle = j1 + j6

        # Rotate the frame about its center
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M_total = cv2.getRotationMatrix2D(center, total_angle, 1.0)
        frame = cv2.warpAffine(frame, M_total, (w, h))


        # Build forward transform from UNFLIPPED -> PROCESSED (annotated_frame) coordinates
        cx    = w / 2.0
        Mflip = np.array([[-1.0, 0.0, 2.0*cx],
                        [ 0.0, 1.0, 0.0     ]], dtype=np.float32)

        # Compose: first apply Mflip, then M_total
        M_forward = self._compose_affine(M_total, Mflip)   # M_total ∘ Mflip

        # >>> CHANGED: use a single stable base for display to avoid “jumps”
        annotated_frame = frame.copy()          # base for visualization
        ee_x, ee_y = self.get_ee_xy()   # meters; may be (None, None)

        # --- Inference (normal + fallback) ---
        try:
            # results = self.model.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]
            results_chip  = self.model_chip.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]
            # results_board = self.model_board.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]

            results_board = self.model_board(
                frame,
                imgsz=640,
                conf=0.5,
                task="obb",
                verbose=False
            )[0]
            cv2.imshow("Frame", frame)
            # MERGE BOTH MODELS
            results = self.merge_obb_results(results_chip, results_board)


        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # >>> NEW: containers used later; always defined to avoid branching differences
        boxes = []          # list[np.ndarray (4,2)] in current-frame coords
        confs = np.array([])

        if self._has_obb(results):
            # Normal path: take polygons as-is
            boxes = [poly.cpu().numpy().reshape(4, 2) for poly in results.obb.xyxyxyxy]
            confs = results.obb.conf.cpu().numpy()
        else:
            # >>> NEW: Fallback—rotate by +20°, detect, back-rotate polygons; keep display steady
            # print(f"{self.counter} No OBBs found, applying fallback rotation")
            self.counter += 1
            fallback_angle = 20.0
            frame_rot, M_fwd, M_inv = self._rotate_image(frame, fallback_angle)
            # orig_frame_rot, M_fwd_o, M_inv_o = self._rotate_image(orig_frame, fallback_angle)

            try:


                # results = self.model.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]
                results_chip  = self.model_chip.predict(frame_rot, imgsz=640, conf=0.8, verbose=False)[0]
                # results_board = self.model_board.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]

                results_board = self.model_board(
                    frame_rot,
                    imgsz=640,
                    conf=0.8,
                    task="obb",
                    verbose=False
                )[0]

                # MERGE BOTH MODELS
                results_rot = self.merge_obb_results(results_chip, results_board)

            except Exception as e:
                self.get_logger().error(f"Fallback inference failed: {e}")
                return

            if self._has_obb(results_rot):
                # >>> NEW: expose results so downstream consumers can still inspect them
                results = results_rot

                # Back-transform polygons to non-rotated display frame
                confs = results_rot.obb.conf.cpu().numpy()

                for poly in results_rot.obb.xyxyxyxy:
                    pts = poly.cpu().numpy().reshape(4, 2)
                    pts_back = self._apply_affine_points(pts, M_inv)
                    boxes.append(pts_back)

                
            # else: nothing found → boxes remain empty; we still show the steady base

        # >>> CHANGED: Always draw OBBs ourselves onto annotated_frame (no results.plot())
        # Overlay joint angles (same as before, but on our annotated_frame)
        y_text = 30
        for label, angle in (("J1", j1), ("J2", j2), ("J3", j3),("J4", j4), ("J5", j5), ("J6", j6)):
            if angle is not None:
                cv2.putText(annotated_frame, f"{label}: {angle:.1f}", (10, int(y_text)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                y_text += 30

        sum_joints = 0.0
        for v in (j2, j3, j5):
            if v is not None:
                sum_joints += v
              
                
        # --- add the sum of J2+J3+J5 ---
        cv2.putText(
            annotated_frame,
            f"J2,3,5: {sum_joints:.1f}",
            (10, int(y_text)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,   # green for visibility
            2
        )
        y_text += 30
        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w / 2, frame_h / 2
        try:
            if len(boxes)  > 0:
                # If counts mismatch (shouldn't, but be robust)
                n = len(boxes)

                conf_thr = 0.7
                all_infos = []  # NEW: will hold [dx_base, dy_base, angle_deg, est_height_mm] per detection

                for i in range(n):
                    if i < len(confs) and confs[i] is not None and confs[i] < conf_thr:
                        continue

                    points = boxes[i]

                    if points.shape != (4, 2) or np.isnan(points).any():
                        continue


                    # Estimate bounding box size from polygon
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    w_px = x_max - x_min
                    h_px = y_max - y_min
                    area_px = w_px * h_px

                    # Thresholds (tune these!)
                    if w_px > 400 or h_px > 400 or area_px > 120000:  
                        # self.get_logger().warn(f"Ignoring too-large OBB: {w_px:.1f}x{h_px:.1f}px (area={area_px:.0f})")
                        continue

                    # Draw polygon + corners
                    pts_i32 = points.astype(np.int32)
                    cv2.polylines(annotated_frame, [pts_i32], isClosed=True,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    for (px, py) in pts_i32:
                        cv2.circle(annotated_frame, (int(px), int(py)), 3, (0, 0, 255), -1)

                    # Center of the OBB (in current frame coords)
                    cx = float(points[:, 0].mean())
                    cy = float(points[:, 1].mean())
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)

                    # Angle from orig_points (your logic)
                    # angle_deg = self.calculate_brick_yaw(orig_points)
                    # if angle_deg is None:
                    #     continue

                    angle_deg_from_frame = self.calculate_brick_yaw(points)
                    angle_deg_from_frame =  self.compensate_angle(angle_deg_from_frame, total_angle)# compensate for J1 + J6

                    angle_deg = angle_deg_from_frame

                    cv2.putText(annotated_frame, f"angle:{angle_deg:.1f}", (10, int(y_text)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    y_text += 30

                    cv2.putText(annotated_frame, f"frame angle:{angle_deg_from_frame:.1f}", (10, int(y_text)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    y_text += 30


                    cv2.putText(
                        annotated_frame, f"{angle_deg:.1f}°",
                        (int(cx) + 30, int(cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )

                    # Vector from image center to brick center (visual)
                    cv2.line(annotated_frame, (int(center_x), int(center_y)),
                            (int(cx), int(cy)), (255, 0, 0), 1)

                    # Distances in px
                    dx_px = cx - center_x
                    dy_px = cy - center_y
                    pixel_dist = float(np.hypot(dx_px, dy_px))

                    # Estimate pixel→mm using long edge of OBB (assume 23mm long side)
                    edge1 = np.linalg.norm(points[1] - points[0])
                    edge2 = np.linalg.norm(points[2] - points[1])
                    pixel_long_edge = max(edge1, edge2)
                    if pixel_long_edge <= 0:
                        continue

                    pixel_to_mm = 23.0 / pixel_long_edge
                    dx_mm = dx_px * pixel_to_mm
                    dy_mm = dy_px * pixel_to_mm

                    # Convert camera-plane offsets to base frame using J1 & J6
                    dx_base, dy_base = self.offset_cam_to_base(
                        dx_mm, dy_mm,
                        j1 if j1 is not None else 0.0,
                        j6 if j6 is not None else 0.0
                    )
                    if dx_base is None or dy_base is None:
                        continue
                    

                    dx_pub = -dx_base            # mm
                    dy_pub = -dy_base            # mm

                    # Convert the published offsets to meters for absolute center:
                    dx_m = dx_pub / 1000.0       # m
                    dy_m = dy_pub / 1000.0       # m

                    # Absolute center in base frame (meters). If EE is unknown, publish NaN to keep shape.
                    cx_m = (ee_x if ee_x is not None else float('nan')) + dx_m
                    cy_m = (ee_y if ee_y is not None else float('nan')) + dy_m

                    # Height estimate
                    focal_length_px = (frame_w / 2) / np.tan(np.radians(72 / 2))
                    brick_half = pixel_long_edge / 2
                    hypotenuse_px = float(np.hypot(brick_half, pixel_dist))
                    est_height_mm = (focal_length_px * 23.0) / hypotenuse_px if hypotenuse_px > 0 else 0.0

                    # Publish info
                    info_msg = Float32MultiArray()
                    info_msg.data = [-dx_base, -dy_base, float(angle_deg), float(est_height_mm)]
                    self.info_publisher.publish(info_msg)
                    all_infos.extend([dx_pub, dy_pub, float(angle_deg), float(est_height_mm), cx_m, cy_m])


                    # Overlays for distance & base-frame dx/dy
                    mid_x = int((center_x + cx) / 2)
                    mid_y = int((center_y + cy) / 2)
                    dist_mm = float(np.hypot(dx_mm, dy_mm))
                    dir_angle_deg = float(np.degrees(np.arctan2(dy_mm, dx_mm)))

                    cv2.putText(
                        annotated_frame,
                        f"{dist_mm:.1f} mm @ {dir_angle_deg:.1f} deg",
                        (int(mid_x) - 50, int(mid_y) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                    )
                # NEW: publish all detections in one shot (shape: N x 4)
                if len(all_infos) > 0:
                    n = len(all_infos) // 6  # not 4 anymore
                    batched = Float32MultiArray()
                    batched.layout = MultiArrayLayout(
                        dim=[
                            MultiArrayDimension(label='objects', size=n,  stride=n*6),
                            MultiArrayDimension(label='fields',  size=6,  stride=6),
                        ],
                        data_offset=0,
                    )
                    batched.data = all_infos
                    self.infos_publisher.publish(batched)
            # else: nothing to draw this frame; annotated_frame is still steady base
        except Exception as e:
            self.get_logger().error(f"Error processing OBB: {e}")

        self.detect_and_publish_aruco(
            frame_unflipped,
            annotated_frame,
            M_forward,
            total_angle,
            ee_x,
            ee_y
        )

        h, w = annotated_frame.shape[:2]
        cv2.drawMarker(
            annotated_frame,
            (w // 2, h // 2),
            (0, 255, 255),               # color (B,G,R)
            markerType=cv2.MARKER_CROSS, # small cross
            markerSize=10,               # length in px
            thickness=2,
            line_type=cv2.LINE_AA
        )

        # Show window + keys
        cv2.imshow('YOLOv8 OBB Inference', annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested, shutting down node.")
            self.destroy_node()
        elif key == ord('c'):
            self.capture_count += 1
            filename = f"frame_{self.capture_count:05d}.jpg"
            filepath = os.path.join(self.capture_folder, filename)
            try:
                cv2.imwrite(filepath, frame)   # save current processed frame
                self.get_logger().info(f"Saved: {filepath}")
            except Exception as e:
                self.get_logger().error(f"Failed to save frame: {e}")


    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    # print("Debugger connected.")


    rclpy.init(args=args)
    node = OV5640Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
