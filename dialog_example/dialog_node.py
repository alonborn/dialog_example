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
from rclpy.node import Node
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
from tf2_ros import Buffer, TransformListener
import easy_handeye2_msgs as ehm
import easy_handeye2 as hec
from easy_handeye2_msgs.srv import TakeSample, SaveCalibration,ComputeCalibration
import queue

import time


import debugpy

class TkinterROS(Node):
    counter = 0

    arm_is_available = False


    def __init__(self):
        super().__init__('tkinter_ros_node')


        self.init_cal_poses()

        # Set up ROS publisher
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        self.timer = self.create_timer(1.0, self.publish_message)

        #Timer: callback every 2.0 seconds
        self.test_timer_callback_group = ReentrantCallbackGroup()
        #self.test_timer = self.create_timer(2.0, self.test_timer_callback,callback_group=self.test_timer_callback_group)



        # Create a ROS service client to request homing
        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')

        #Wait for homing service to be available
        # while not self.homing_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().warn('Waiting for homing service to become available...')

        # Set up Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")
        self.root.geometry("600x600")  # Increased dialog size

        #Create a label
        self.label = tk.Label(self.root, text="Waiting for messages...", font=("Arial", 16))
        self.label.pack(pady=10)

        # Create buttons
        self.homing_button = tk.Button(self.root, text="Homing", command=self.on_homing_button_click, font=("Arial", 14), width=20, height=2)
        self.homing_button.pack(pady=5)

        self.calibrate_button = tk.Button(self.root, text="Calibrate", command=self.on_calibrate_button_click, font=("Arial", 14), width=20, height=2)
        self.calibrate_button.pack(pady=5)

        self.validate_button = tk.Button(self.root, text="Validate", command=self.on_validate_button_click, font=("Arial", 14), width=20, height=2)
        self.validate_button.pack(pady=5)
        
        self.init_button = tk.Button(self.root, text="Init", command=self.on_init_button_click, font=("Arial", 14), width=20, height=2)
        self.init_button.pack(pady=5)
        
        # Create text field for 'Joints to calibrate'
        self.joints_entry = tk.Entry(self.root, font=("Arial", 14), width=30)
        self.joints_entry.pack(pady=10)
        self.joints_entry.insert(0, "000001")

        self.pos_text = tk.Text(self.root, height=3, font=("Arial", 16), wrap="word")
        self.pos_text.pack(pady=10)
        self.pos_text.configure(state='disabled')  # make it look like a label

        self.update_position_label()

        # Use after to periodically update Tkinter UI in the main thread
        self.root.after(100, self.tk_mainloop)

        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]

        self.get_logger().info('registering ar_move_to_pose service')
        self.move_client = self.create_client(MoveToPose,'ar_move_to_pose')
        # self.move_client.wait_for_service()
        self.get_logger().info('ar_move_to_pose service registered')

        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)

        #self.take_sample_client.wait_for_service()
        self.get_logger().info('take_sample service registered')
        self.arm_is_available = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
    def test_timer_callback(self):
        self.get_logger().info('ROS loop is alive!')

    def init_cal_poses(self):
        self.cal_poses =    [[(0.03, -0.38, 0.39), (0.31, -0.56, 0.64, -0.43)],
                            [(0.03, -0.39, 0.39), (0.26, -0.48, 0.70, -0.46)],
                            [(0.03, -0.38, 0.39), (0.34, -0.43, 0.77, -0.34)],
                            [(0.03, -0.38, 0.39), (0.36, -0.48, 0.73, -0.31)],
                            [(0.03, -0.38, 0.39), (0.39, -0.47, 0.72, -0.33)],
                            [(0.03, -0.38, 0.39), (0.31, -0.52, 0.66, -0.44)]]


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

            new_text = (f"pos = ({x:.2f}, {y:.2f}, {z:.2f})\n"
                        f"ori = ({ox:.2f}, {oy:.2f}, {oz:.2f}, {ow:.2f})")

            current_text = self.pos_text.get("1.0", tk.END).strip()

            if new_text != current_text:
                self.pos_text.configure(state='normal')
                self.pos_text.delete(1.0, tk.END)
                self.pos_text.insert(tk.END, new_text)
                self.pos_text.configure(state='disabled')

        self.root.after(100, self.update_position_label)


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

    def send_move_request(self, pose):
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

    def publish_message(self):
        msg = String()
        msg.data = self.joints_entry.get()  # Send the value from the text field
        self.publisher.publish(msg)
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

 
    def on_calibrate_button_click(self):

        # position = Pose()
        
        # pose = self.create_pose((0.04, -0.31, 0.4), (0.044, -0.702, 0.71, -0.03))
        # print(pose)
        # print ("sending move request to:" + str(pose)) 
        # self.send_move_request(pose)
        # time.sleep(2)

        for i, pose in enumerate(self.cal_poses):
            #print("sending move request")
            pose_msg = self.create_pose(pose[0], pose[1])
            #print(pose_msg)
            self.send_move_request(pose_msg)
            time.sleep(3)
            self.take_sample()
            print(f"sample no. {i} taken")   
           

        self.compute_calibration()
        print("calibration computed")
        self.save_calibration()
        print("calibration saved")
        print("---------------done calibrating poses-------------")


        # self.take_sample()
        # self.get_logger().info("Sample taken - in dialog_node")
        # pose = self.create_pose((0.04, -0.31, 0.275), (0.1, -0.702, 0.71, -0.03))
        # print(pose)
        # print ("sending move request number 222222") 
        # self.send_move_request(pose)


        # pose = self.create_pose((0.04, -0.31, 0.375), (0.1, -0.702, 0.71, -0.03))
        # print(pose)
        # print ("sending move request number 333333") 
        # self.send_move_request(pose)

        

    def on_validate_button_click(self):
        self.trigger_service("Validation in progress...", "Validation successful!", "Validation failed")

    def on_init_button_click(self):
        self.initAR4()

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

def main():


    # debugpy.listen(("localhost", 5678))  # Port for debugger to connect
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Ensures the debugger connects before continuing
    # print("Debugger connected.")


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
