#!/home/alon/venv/bin/python3

import sys
print(f"[DEBUG] Running with Python interpreter: {sys.executable}")

import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

CAMERA_INDEX = 2  # adjust if needed
CALIB_FILE = "/home/alon/ros_ws/src/dialog_example/dialog_example/camera_calibration.npz"

class OV5640ImagePublisher(Node):
    def __init__(self):
        super().__init__('ov5640_image_publisher')

        # ROS publishers
        self.image_pub = self.create_publisher(Image, '/ov5640/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/ov5640/camera_info', 10)
        self.bridge = CvBridge()

        # Open the camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open OV5640 camera")
            self.use_dummy = True
        else:
            self.use_dummy = False
            self.get_logger().info("OV5640 camera opened successfully")

        # Defaults
        self.map1 = self.map2 = None
        self.K = None
        self.dist = None
        self.new_K = None

        # Load calibration
        if os.path.exists(CALIB_FILE):
            data = np.load(CALIB_FILE)
            self.K = data["K"]
            self.dist = data["dist"]

            # get an initial frame to compute undistortion map
            ret, frame = self.cap.read()
            if not ret or frame is None:
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                h, w = frame.shape[:2]

            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2
            )
            self.get_logger().info("Loaded camera calibration")
        else:
            self.get_logger().warn(f"No calibration file found: {CALIB_FILE}")

        # Run at ~10Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if self.use_dummy:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to capture frame")
                return

        # Undistort if calibration is available
        if self.map1 is not None and self.map2 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Publish image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(msg)

        # Publish CameraInfo
        cam_info = CameraInfo()
        cam_info.width = frame.shape[1]
        cam_info.height = frame.shape[0]
        cam_info.header = msg.header  # sync timestamps/frame_id

        if self.K is not None and self.dist is not None:
            cam_info.distortion_model = "plumb_bob"
            cam_info.d = self.dist.flatten().tolist()
            cam_info.k = self.K.flatten().tolist()
            cam_info.r = [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0]
            if self.new_K is not None:
                fx, fy = self.new_K[0, 0], self.new_K[1, 1]
                cx, cy = self.new_K[0, 2], self.new_K[1, 2]
            else:
                fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            cam_info.p = [fx, 0.0, cx, 0.0,
                          0.0, fy, cy, 0.0,
                          0.0, 0.0, 1.0, 0.0]
        else:
            # Fallback dummy intrinsics
            fx, fy = 600.0, 600.0
            cx, cy = cam_info.width / 2, cam_info.height / 2
            cam_info.distortion_model = "plumb_bob"
            cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            cam_info.k = [fx, 0.0, cx,
                          0.0, fy, cy,
                          0.0, 0.0, 1.0]
            cam_info.r = [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0]
            cam_info.p = [fx, 0.0, cx, 0.0,
                          0.0, fy, cy, 0.0,
                          0.0, 0.0, 1.0, 0.0]

        self.info_pub.publish(cam_info)

        # Show image locally
        # cv2.imshow("OV5640 Undistorted", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     self.get_logger().info("Quit requested, shutting down node.")
        #     rclpy.shutdown()

    def destroy_node(self):
        if not self.use_dummy:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OV5640ImagePublisher()
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
