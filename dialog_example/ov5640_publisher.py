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


class OV5640Publisher(Node):
    def __init__(self):
        super().__init__('ov5640_publisher')
        self.counter = 0
        self.joint_states = None  # Initialize joint states to None
        # ROS image publisher
        # self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        # self.pose_pub = self.create_publisher(Pose2D, 'brick_info', 10)
        self.info_publisher = self.create_publisher(Float32MultiArray, 'brick_top_info', 10)

        self.joint_positions = [None, None, None, None, None, None]  # store last positions
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(2)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 5)

        if not self.cap.isOpened():
            self.get_logger().warn('Camera not available — using black frames')
            self.use_dummy_frame = True
        else:
            self.use_dummy_frame = False

        self.capture_folder = "captured_images"
        os.makedirs(self.capture_folder, exist_ok=True)
        self.capture_count = 0

        # Load YOLOv8 OBB model
        model_path = "/home/alon/Documents/Projects/top_view_train/runs/obb/train8/weights/best.pt"
        self.model = YOLO(model_path)
        self.handle_camera_calibration()

    def handle_camera_calibration(self):
        # === Load calibration data ===
        calib_path = "/home/alon/ros_ws/src/dialog_example/dialog_example/camera_calibration.npz"
        if not os.path.exists(calib_path):
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            rclpy.shutdown()

        data = np.load(calib_path)
        self.K = data["K"]
        self.dist = data["dist"]

        # Precompute undistortion map
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("No initial frame — using black frame for calibration")
            height, width = 480, 640
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)
        self.optical_center = (self.new_K[0, 2], self.new_K[1, 2])  # (cx, cy)

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

    def _apply_affine_points(self, pts4x2, M2x3):
        """Apply 2x3 affine transform to a (4,2) polygon (numpy)."""
        pts = np.hstack([pts4x2, np.ones((pts4x2.shape[0], 1))])  # (4,3)
        out = pts @ M2x3.T  # (4,2)
        return out

    def timer_callback(self):
        # Grab frame
        ret, frame = self.cap.read()
        text_color = (255,0,0)
        if not ret or frame is None:
            # Create a black frame matching the calibration map size
            height, width = self.map1.shape[:2]
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Undistort/rectify
        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Shift so optical center becomes image center
        cx, cy = self.optical_center
        h, w = frame.shape[:2]
        target_cx, target_cy = w / 2, h / 2
        dx, dy = target_cx - cx, target_cy - cy
        M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M_translate, (w, h))

        orig_frame = frame.copy()

        # Align camera orientation (keep as before)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
            return

        # Total rotation for the image (use J1 + J6)
        total_angle = j1 + j6

        # Rotate the frame about its center
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, total_angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))

        # >>> CHANGED: use a single stable base for display to avoid “jumps”
        display_frame = frame.copy()          # base for visualization
        annotated_frame = display_frame.copy()  # we will draw everything ourselves

        # --- Inference (normal + fallback) ---
        try:
            results = self.model.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]
            orig_results = self.model.predict(orig_frame, imgsz=640, conf=0.8, verbose=False)[0]
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # >>> NEW: containers used later; always defined to avoid branching differences
        boxes = []          # list[np.ndarray (4,2)] in current-frame coords
        orig_boxes = []     # list[np.ndarray (4,2)] in orig_frame coords
        confs = np.array([])

        # >>> CHANGED: consistent OBB extraction with fallback, no results.plot()
        if self._has_obb(results) and self._has_obb(orig_results):
            # Normal path: take polygons as-is
            boxes = [poly.cpu().numpy().reshape(4, 2) for poly in results.obb.xyxyxyxy]
            orig_boxes = [poly.cpu().numpy().reshape(4, 2) for poly in orig_results.obb.xyxyxyxy]
            confs = results.obb.conf.cpu().numpy()
        else:
            # >>> NEW: Fallback—rotate by +20°, detect, back-rotate polygons; keep display steady
            # print(f"{self.counter} No OBBs found, applying fallback rotation")
            self.counter += 1
            fallback_angle = 20.0
            frame_rot, M_fwd, M_inv = self._rotate_image(frame, fallback_angle)
            orig_frame_rot, M_fwd_o, M_inv_o = self._rotate_image(orig_frame, fallback_angle)

            try:
                results_rot = self.model.predict(frame_rot, imgsz=640, conf=0.8, verbose=False)[0]
                orig_results_rot = self.model.predict(orig_frame_rot, imgsz=640, conf=0.8, verbose=False)[0]
            except Exception as e:
                self.get_logger().error(f"Fallback inference failed: {e}")
                return

            if self._has_obb(results_rot) and self._has_obb(orig_results_rot):
                # >>> NEW: expose results so downstream consumers can still inspect them
                results = results_rot
                orig_results = orig_results_rot

                # Back-transform polygons to non-rotated display frame
                confs = results_rot.obb.conf.cpu().numpy()

                for poly in results_rot.obb.xyxyxyxy:
                    pts = poly.cpu().numpy().reshape(4, 2)
                    pts_back = self._apply_affine_points(pts, M_inv)
                    boxes.append(pts_back)

                for poly in orig_results_rot.obb.xyxyxyxy:
                    pts = poly.cpu().numpy().reshape(4, 2)
                    pts_back = self._apply_affine_points(pts, M_inv_o)
                    orig_boxes.append(pts_back)
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

        # >>> CHANGED: Draw polygons consistently; then process/publish like before
        try:
            if len(boxes) > 0 and len(orig_boxes) > 0:
                # If counts mismatch (shouldn't, but be robust)
                n = min(len(boxes), len(orig_boxes))
                conf_thr = 0.7

                for i in range(n):
                    if i < len(confs) and confs[i] is not None and confs[i] < conf_thr:
                        continue

                    points = boxes[i]
                    orig_points = orig_boxes[i]

                    if points.shape != (4, 2) or np.isnan(points).any():
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
                    angle_deg = self.calculate_brick_yaw(orig_points)
                    if angle_deg is None:
                        continue

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

                    # Height estimate
                    focal_length_px = (frame_w / 2) / np.tan(np.radians(72 / 2))
                    brick_half = pixel_long_edge / 2
                    hypotenuse_px = float(np.hypot(brick_half, pixel_dist))
                    est_height_mm = (focal_length_px * 23.0) / hypotenuse_px if hypotenuse_px > 0 else 0.0

                    # Publish info
                    info_msg = Float32MultiArray()
                    info_msg.data = [-dx_base, -dy_base, float(angle_deg), float(est_height_mm)]
                    self.info_publisher.publish(info_msg)

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
            # else: nothing to draw this frame; annotated_frame is still steady base
        except Exception as e:
            self.get_logger().error(f"Error processing OBB: {e}")

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




    def run_batch_inference(self):
        input_dir = "/home/alon/Documents/Projects/top_view_train/images/val/"
        output_dir = "/home/alon/Documents/Projects/top_view_train/images/val_annotated/"
        os.makedirs(output_dir, exist_ok=True)

        image_paths = sorted(glob(os.path.join(input_dir, "*.jpg")))
        if not image_paths:
            self.get_logger().warn(f"No images found in {input_dir}")
            return

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                self.get_logger().warn(f"Could not read {path}")
                continue

            results = self.model.predict(img, imgsz=640, conf=0.25)[0]
            annotated = results.plot()

            if results.obb is not None and results.obb.xywh is not None:
                for obb in results.obb.xywh:
                    x_center, y_center, w, h, angle_deg = obb.tolist()
                    if h > w:
                        angle_deg += 90
                        w, h = h, w
                    self.get_logger().info(f"[{os.path.basename(path)}] Center: ({x_center:.1f}, {y_center:.1f}), Angle: {angle_deg:.1f}°")

            save_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(save_path, annotated)
            self.get_logger().info(f"Saved annotated: {save_path}")

    def destroy_node(self):
        self.cap.release()
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
