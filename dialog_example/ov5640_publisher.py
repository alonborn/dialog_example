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


class OV5640Publisher(Node):
    def __init__(self):
        super().__init__('ov5640_publisher')

        # ROS image publisher
        # self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        # self.pose_pub = self.create_publisher(Pose2D, 'brick_info', 10)
        self.info_publisher = self.create_publisher(Float32MultiArray, 'brick_info', 10)


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

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        # Run inference (disable verbose)
        results = self.model.predict(frame, imgsz=640, conf=0.7, verbose=False)[0]
        annotated_frame = results.plot()

        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w / 2, frame_h / 2

        # Process oriented bounding boxes (OBB)
        if results.obb is not None and results.obb.xyxyxyxy is not None:
            boxes = results.obb.xyxyxyxy
            confs = results.obb.conf.cpu().numpy()

            for obb, conf in zip(boxes, confs):
                if conf < 0.7:
                    continue  # Skip low confidence detections

                points = obb.cpu().numpy().reshape(4, 2)
                cx = float(points[:, 0].mean())
                cy = float(points[:, 1].mean())

                # Draw center point
                cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)

                # Compute angle based on longer edge
                edge1 = np.linalg.norm(points[1] - points[0])
                edge2 = np.linalg.norm(points[2] - points[1])
                if edge1 >= edge2:
                    dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
                else:
                    dx, dy = points[2][0] - points[1][0], points[2][1] - points[1][1]
                angle_rad = np.arctan2(dy, dx)
                angle_deg = float(np.degrees(angle_rad))

                # Normalize angle to [-90, 90]
                if angle_deg > 90:
                    angle_deg -= 180
                elif angle_deg < -90:
                    angle_deg += 180

                # Draw angle text
                cv2.putText(annotated_frame,
                            f"{angle_deg:.1f}°",
                            (int(cx) + 30, int(cy) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Draw line and distance from image center
                cv2.line(annotated_frame, (int(center_x), int(center_y)), (int(cx), int(cy)), (255, 0, 0), 1)
                dx_px = cx - center_x
                dy_px = cy - center_y
                pixel_dist = np.sqrt(dx_px**2 + dy_px**2)

                # Use the known long edge (23 mm) and measured long edge in pixels for scaling
                pixel_long_edge = max(edge1, edge2)
                if pixel_long_edge > 0:
                    pixel_to_mm = 23.0 / pixel_long_edge
                    dx_mm = dx_px * pixel_to_mm
                    dy_mm = dy_px * pixel_to_mm

                    # Adjust height estimate based on real 3D distance from center
                    focal_length_px = (frame_w / 2) / np.tan(np.radians(72 / 2))  # 72 deg HFOV
                    brick_half = pixel_long_edge / 2
                    hypotenuse_px = np.sqrt(brick_half**2 + pixel_dist**2)
                    est_height_mm = (focal_length_px * 23.0) / hypotenuse_px
                    height_text = f"H: {est_height_mm:.1f} mm"

                    # Display estimated height near the brick
                    cv2.putText(annotated_frame,
                                height_text,
                                (int(cx) + 30, int(cy) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Publish detection info
                    
                    info_msg = Float32MultiArray()
                    info_msg.data = [dx_mm, dy_mm, angle_deg, est_height_mm]
                    self.info_publisher.publish(info_msg)

                    # Show distance text at midpoint
                    mid_x = int((center_x + cx) / 2)
                    mid_y = int((center_y + cy) / 2)
                    dist_text = f"{np.sqrt(dx_mm**2 + dy_mm**2):.1f} mm"
                    cv2.putText(annotated_frame, dist_text, (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Publish original frame (not the annotated one)
        # msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        # self.publisher_.publish(msg)

        # Show annotated frame
        cv2.imshow('YOLOv8 OBB Inference', annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info("Quit requested, shutting down node.")
            self.destroy_node()
        elif key == ord('c'):
            self.capture_count += 1
            filename = f"frame_{self.capture_count:05d}.jpg"
            filepath = os.path.join(self.capture_folder, filename)
            cv2.imwrite(filepath, frame)
            self.get_logger().info(f"Saved: {filepath}")






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
