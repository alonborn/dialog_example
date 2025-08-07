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
            self.get_logger().error('Failed to open video device')
            rclpy.shutdown()

        self.capture_folder = "captured_images"
        os.makedirs(self.capture_folder, exist_ok=True)
        self.capture_count = 0

        # Load YOLOv8 OBB model
        model_path = "/home/alon/Documents/Projects/top_view_train/runs/obb/train8/weights/best.pt"
        self.model = YOLO(model_path)


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


    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn('Failed to read frame')
            return
        
        orig_frame = frame.copy()


        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)

        # --- Get joint 1 and joint 6 angles ---
        joint1_angle = self.get_joint_angle('joint_1')
        joint6_angle = self.get_joint_angle('joint_6')

        if joint1_angle is None or joint6_angle is None:
            self.get_logger().warn("Joint angles not available, skipping frame processing")
            return

        # Total rotation angle
        total_angle = joint1_angle + joint6_angle  # degrees

        # --- Rotate the frame around its center ---
        (h, w) = frame.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, total_angle, 1.0)

        # Perform the rotation
        frame = cv2.warpAffine(frame, M, (w, h))


        try:
            # Run inference (disable verbose)
            results = self.model.predict(frame, imgsz=640, conf=0.8, verbose=False)[0]
            orig_results = self.model.predict(orig_frame, imgsz=640, conf=0.8, verbose=False)[0]
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        annotated_frame = results.plot()



        if joint1_angle is not None and joint6_angle is not None:
            cv2.putText(annotated_frame, f"J1: {joint1_angle:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(annotated_frame, f"J6: {joint6_angle:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w / 2, frame_h / 2

        # Process oriented bounding boxes (OBB) safely
        if hasattr(results, "obb") and results.obb is not None \
            and getattr(results.obb, "xyxyxyxy", None) is not None:
            try:
                boxes = results.obb.xyxyxyxy
                orig_boxes = orig_results.obb.xyxyxyxy

                confs = results.obb.conf.cpu().numpy()

                for obb, conf,obb_orig in zip(boxes, confs,orig_boxes):
                    if conf is None or conf < 0.7:
                        continue

                    points = obb.cpu().numpy().reshape(4, 2)
                    orig_points = obb_orig.cpu().numpy().reshape(4, 2)
                    
                    if points.shape != (4, 2) or np.isnan(points).any():
                        continue

                    cx = float(points[:, 0].mean())
                    cy = float(points[:, 1].mean())

                    # Draw center point
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)

                    angle_deg = self.calculate_brick_yaw(orig_points)
 
                    if angle_deg is None:
                        continue
 
                    cv2.putText(annotated_frame,
                                f"{angle_deg:.1f}°",
                                (int(cx) + 30, int(cy) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    # Line from center to brick
                    cv2.line(annotated_frame, (int(center_x), int(center_y)), (int(cx), int(cy)), (255, 0, 0), 1)
                    dx_px = cx - center_x
                    dy_px = cy - center_y
                    pixel_dist = np.sqrt(dx_px**2 + dy_px**2)


                    edge1 = np.linalg.norm(points[1] - points[0])
                    edge2 = np.linalg.norm(points[2] - points[1])

                    pixel_long_edge = max(edge1, edge2)
                    if pixel_long_edge > 0:
                        pixel_to_mm = 23.0 / pixel_long_edge
                        dx_mm = dx_px * pixel_to_mm
                        dy_mm = dy_px * pixel_to_mm

                        dx_base, dy_base = self.offset_cam_to_base(
                            dx_mm, dy_mm,
                            joint1_angle if joint1_angle is not None else 0.0,
                            joint6_angle if joint6_angle is not None else 0.0
                        )

                        if dx_base is None or dy_base is None:
                            continue  # Skip if offset conversion failed

                        # Height estimate
                        focal_length_px = (frame_w / 2) / np.tan(np.radians(72 / 2))
                        brick_half = pixel_long_edge / 2
                        hypotenuse_px = np.sqrt(brick_half**2 + pixel_dist**2)
                        if hypotenuse_px > 0:
                            est_height_mm = (focal_length_px * 23.0) / hypotenuse_px
                        else:
                            est_height_mm = 0.0

                        height_text = f"H: {est_height_mm:.1f} mm"
                        cv2.putText(annotated_frame,
                                    height_text,
                                    (int(cx) + 30, int(cy) + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        # Publish info
                        info_msg = Float32MultiArray()
                        info_msg.data = [dx_base, dy_base, angle_deg, est_height_mm]
                        self.info_publisher.publish(info_msg)

                        # Distance and angle from center → brick
                        mid_x = int((center_x + cx) / 2)
                        mid_y = int((center_y + cy) / 2)
                        dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)
                        dir_angle_deg = np.degrees(np.arctan2(dy_mm, dx_mm))

                        dist_angle_text = f"{dist_mm:.1f} mm @ {dir_angle_deg:.1f}\n deg"
                        cv2.putText(annotated_frame, dist_angle_text, (mid_x-50, mid_y-50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        
                        cv2.putText(annotated_frame, f"dx_base: {dx_base:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(annotated_frame, f"dy_base: {dy_base:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            except Exception as e:
                self.get_logger().error(f"Error processing OBB: {e}")

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
                cv2.imwrite(filepath, frame)
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
