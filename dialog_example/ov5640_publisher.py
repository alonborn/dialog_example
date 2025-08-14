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
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")

class OV5640Publisher(Node):
    def __init__(self):
        super().__init__('ov5640_publisher')
        self.joint_states = None  # Initialize joint states to None
        self.last_rotated_points = None
        self.last_center = None
        self.last_angle = None
        self.last_prediction_time = None
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
        #model_path = "/home/alon/Documents/Projects/top_view_train/runs/obb/train2/weights/best.pt"
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

    def normalize_angle(self, angle_deg):

        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        return angle_deg

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
            (optical_x + 10, optical_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )



    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Failed to read frame")
            return


        # Rotate and flip if needed
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)


        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w / 2, frame_h / 2

        annotated_frame = frame.copy()

        # Get joint angles
        j1 = self.get_joint_angle("joint_1")
        j6 = self.get_joint_angle("joint_6")
        if j1 is None or j6 is None:
            self.get_logger().warn("Joint angles not available")
            return

        # Run YOLO OBB detection
        results = self.model(frame)
        found_prediction = False

        for r in results:
            if not hasattr(r, "obb") or r.obb is None:
                continue
            for obb in r.obb.xyxyxyxy:
                points = obb.cpu().numpy().reshape(4, 2)
                if points.shape != (4, 2) or np.isnan(points).any():
                    continue

                # --- Black area filter ---
                cx_orig = int(points[:, 0].mean())
                cy_orig = int(points[:, 1].mean())
                if 1 <= cx_orig < frame_w - 1 and 1 <= cy_orig < frame_h - 1:
                    region = frame[cy_orig-1:cy_orig+2, cx_orig-1:cx_orig+2]
                    if region.size > 0 and region.mean() < 10:
                        continue

                # --- Aspect ratio filter ---
                edge_lengths = [
                    np.linalg.norm(points[1] - points[0]),
                    np.linalg.norm(points[2] - points[1]),
                    np.linalg.norm(points[3] - points[2]),
                    np.linalg.norm(points[0] - points[3])
                ]
                length_px = max(edge_lengths)
                width_px = min(edge_lengths)
                if width_px == 0:
                    continue
                ratio = length_px / width_px
                expected_ratio = 24.6 / 14.6
                if abs(ratio - expected_ratio) > 0.15:  # tolerance
                    continue

                # --- Yaw angle ---
                angle_deg = self.calculate_brick_yaw(points)
                rotate_angle = -angle_deg
                
                angle_deg = self.normalize_angle(angle_deg)
                if angle_deg is None:
                    continue
                

                # Rotate bbox points
                M_rot = cv2.getRotationMatrix2D((center_x, center_y), rotate_angle, 1.0)
                rotated_points = np.hstack([points, np.ones((4, 1))]) @ M_rot.T
                rotated_points = rotated_points.astype(int)

                # Rotated bbox center
                cx_rot = float(rotated_points[:, 0].mean())
                cy_rot = float(rotated_points[:, 1].mean())

                # dx, dy in px
                dx_px = cx_rot - center_x
                dy_px = cy_rot - center_y

                # Convert px → mm
                pixel_to_mm = 23.0 / length_px
                dx_mm = dx_px * pixel_to_mm
                dy_mm = dy_px * pixel_to_mm

                # Convert to base frame
                dx_base, dy_base = self.offset_cam_to_base(dx_mm, dy_mm, j1, j6)
                if dx_base is None or dy_base is None:
                    continue

                # Distance in mm
                dist_mm = np.sqrt(dx_base**2 + dy_base**2)

                # Draw center mark & line
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (int(cx_rot), int(cy_rot)), 4, (0, 255, 255), -1)
                cv2.line(annotated_frame, (int(center_x), int(center_y)), (int(cx_rot), int(cy_rot)), (255, 0, 0), 2)
                cv2.polylines(annotated_frame, [rotated_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Text overlay
                text_color = (255,0,0)
                cv2.putText(annotated_frame, f"J1: {j1:.1f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(annotated_frame, f"J6: {j6:.1f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(annotated_frame, f"dx: {dx_base:.1f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(annotated_frame, f"dy: {dy_base:.1f} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(annotated_frame, f"Dist: {dist_mm:.1f} mm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(annotated_frame, f"Angle: {angle_deg:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

                # Publish info
                info_msg = Float32MultiArray()
                info_msg.data = [dx_base, dy_base, float(angle_deg), float(dist_mm)]
                self.info_publisher.publish(info_msg)

                # Save last prediction
                self.last_rotated_points = rotated_points
                self.last_bbox_center = (cx_rot, cy_rot)
                self.last_dx_base = dx_base
                self.last_dy_base = dy_base
                self.last_angle_deg = angle_deg
                self.last_dist_mm = dist_mm

                found_prediction = True
                break  # only one brick

        # If no prediction, use last
        if not found_prediction and self.last_rotated_points is not None:
            cx_rot, cy_rot = self.last_bbox_center
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (int(cx_rot), int(cy_rot)), 4, (0, 255, 255), -1)
            cv2.line(annotated_frame, (int(center_x), int(center_y)), (int(cx_rot), int(cy_rot)), (255, 0, 0), 2)
            cv2.polylines(annotated_frame, [self.last_rotated_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Text
            cv2.putText(annotated_frame, f"J1: {j1:.1f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"J6: {j6:.1f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"dx: {self.last_dx_base:.1f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"dy: {self.last_dy_base:.1f} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Dist: {self.last_dist_mm:.1f} mm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Angle: {self.last_angle_deg:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Publish last known values
            info_msg = Float32MultiArray()
            info_msg.data = [self.last_dx_base, self.last_dy_base, float(self.last_angle_deg), float(self.last_dist_mm)]
            self.info_publisher.publish(info_msg)

        # Show annotated frame
        cv2.imshow("Rotated BBox View", annotated_frame)
        cv2.waitKey(1)

        
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

            results = self.model.predict(img, imgsz=640, conf=0.25,verbose=False)[0]
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
    debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    print("Debugger connected.")


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
