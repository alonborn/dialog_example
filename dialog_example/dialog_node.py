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

import debugpy

class TkinterROS(Node):
    counter = 0

    arm_is_available = False


    def __init__(self):
        super().__init__('tkinter_ros_node')
        self.init_cal_poses()
        self.mover = Mover.Mover(self)

        # Set up ROS publisher
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        self.timer = self.create_timer(1.0, self.publish_message)
        self.create_subscription(Point, '/aruco_pose', self.aruco_pose_callback, 10)
        self.test_timer_callback_group = ReentrantCallbackGroup()

        self.homing_client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')

        # Subscribe to joint states
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)

        # Set up Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")
        self.root.geometry("1000x600")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        self.right_frame = tk.Frame(self.main_frame, bg='lightgray')
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        self.label = tk.Label(self.left_frame, text="Waiting for messages...", font=("Arial", 16))
        self.label.pack(pady=10)

        self.homing_button = tk.Button(self.left_frame, text="Homing", command=self.on_homing_button_click, font=("Arial", 14), width=20, height=2)
        self.homing_button.pack(pady=5)

        self.calibrate_button = tk.Button(self.left_frame, text="Calibrate", command=self.on_calibrate_button_click, font=("Arial", 14), width=20, height=2)
        self.calibrate_button.pack(pady=5)

        self.validate_button = tk.Button(self.left_frame, text="Go Home", command=self.on_go_home_button_click, font=("Arial", 14), width=20, height=2)
        self.validate_button.pack(pady=5)

        self.init_button = tk.Button(self.left_frame, text="Send Pos", command=self.on_send_pos_button_click, font=("Arial", 14), width=20, height=2)
        self.init_button.pack(pady=5)

        self.joints_entry = tk.Entry(self.left_frame, font=("Arial", 14), width=30)
        self.joints_entry.pack(pady=10)
        self.joints_entry.insert(0, "000001")

        self.aruco_pose_entry = tk.Text(self.right_frame, font=("Courier", 10), width=30, height=3)
        self.aruco_pose_entry.pack(pady=10)

        self.copy_aruco_button = tk.Button(
            self.right_frame, 
            text="Copy Aruco Pose", 
            font=("Arial", 10), 
            command=self.copy_aruco_pose_to_clipboard
        )
        self.copy_aruco_button.pack(pady=5)

        self.pos_text = tk.Text(self.left_frame, height=3, font=("Arial", 12), wrap="word")
        self.pos_text.pack(pady=10)
        self.pos_text.configure(state='disabled')

        self.pose_frame = tk.Frame(self.left_frame)
        self.pose_frame.pack(pady=(20, 5))

        self.translation_label = tk.Label(self.pose_frame, text="Translation:", font=("Arial", 10))
        self.translation_label.grid(row=0, column=0, padx=5)
        self.translation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=12)
        self.translation_entry.grid(row=0, column=1, padx=5)
        self.translation_entry.insert(0, "(0.03, -0.38, 0.20)")

        self.rotation_label = tk.Label(self.pose_frame, text="Rotation:", font=("Arial", 10))
        self.rotation_label.grid(row=0, column=2, padx=5)
        self.rotation_entry = tk.Entry(self.pose_frame, font=("Arial", 10), width=18)
        self.rotation_entry.grid(row=0, column=3, padx=5)
        self.rotation_entry.insert(0, "(0.0, 0.7071, 0.0, 0.7071)")

        self.joint_states_var = tk.StringVar()
        self.joint_states_entry = tk.Text(self.right_frame, font=("Courier", 10), width=30, height=10)
        self.joint_states_entry.pack(pady=10)

        self.update_position_label()
        self.root.after(100, self.tk_mainloop)

        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        self.get_logger().info('registering ar_move_to_pose service')
        #self.move_client = self.create_client(MoveToPose, 'ar_move_to_pose')

        #self.action_client = ActionClient(self,MoveToPoseAc, 'ar_move_to_pose')
        
        #self.get_logger().info('ar_move_to_pose service registered')

        self.save_sample_calibration_client = self.create_client(SaveCalibration, hec.SAVE_CALIBRATION_TOPIC)
        self.take_sample_client = self.create_client(TakeSample, hec.TAKE_SAMPLE_TOPIC)
        self.compute_calibration_client = self.create_client(ComputeCalibration, hec.COMPUTE_CALIBRATION_TOPIC)

        self.get_logger().info('take_sample service registered')
        self.arm_is_available = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
    def copy_aruco_pose_to_clipboard(self):
        try:
            pose_text = self.aruco_pose_entry.get("1.0", tk.END).strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(pose_text)
            self.root.update()  # now it stays on the clipboard after the window is closed
            print("Aruco pose copied to clipboard:", pose_text)
        except Exception as e:
            self.get_logger().error(f"Error copying Aruco pose to clipboard: {e}")

    def joint_states_callback(self, msg):
        try:
            joint_info = "\n".join([f"{pos:.4f}," for pos in msg.position])
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
        self.cal_poses = [
            [(0.03, -0.38, 0.39), (0.0, 0.7071, 0.0, 0.7071)],  # Facing left (camera side)
            [(0.03, -0.38, 0.39), (0.3827, 0.0, 0.0, 0.9239)],  # Roll +45°
            [(0.03, -0.38, 0.39), (0.2706, 0.0, 0.0, 0.9627)],  # Roll +30° 
            [(0.03, -0.38, 0.39), (0.0, 0.3827, 0.0, 0.9239)],  # Pitch +45°
            [(0.03, -0.38, 0.39), (0.0, 0.2588, 0.0, 0.9659)],  # Pitch +30°
            [(0.03, -0.38, 0.39), (0.0, 0.0, -0.2588, 0.9659)], # Yaw -30°   XXXXXXXXx
            [(0.03, -0.38, 0.39), (0.2706, 0.2706, 0.2706, 0.8820)],  # All axes +30° ?????
            [(0.03, -0.38, 0.39), (0.191, 0.191, -0.191, 0.9511)],  # Mixed small rotation 
            [(0.03, -0.38, 0.39), (0.5, 0.5, 0.5, 0.5)],  # 90° on all axes
            [(0.03, -0.38, 0.39), (-0.5, 0.5, -0.5, 0.5)],  # 90° mixed
            [(0.03, -0.38, 0.35), (0.0, 0.7071, 0.0, 0.7071)],  # Facing left, lower Z
            [(0.03, -0.38, 0.43), (0.0, 0.7071, 0.0, 0.7071)],  # Facing left, higher Z
            [(0.06, -0.38, 0.41), (0.0, 0.7071, 0.0, 0.7071)],  # Slight X/Z offset, still facing camera
            [(0.03, -0.38, 0.41), (0.191, 0.0, 0.0, 0.9815)], # Opposite lean
            [(0.03, -0.38, 0.37), (0.191, -0.191, 0.191, 0.9511)], # Mixed small rotation, lower Z
        ]


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
        time.sleep(0.5) #for some reason the arm is not available immediately after the service call
        self.mover.MoveArm(pose.position, pose.orientation)
        #time.sleep(1)
        return
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

    def move_pos_action(self, pose):
        # Create a goal message
        goal_msg = MoveToPoseAc.Goal(pose = pose)
        

        # Send the goal to the action server
        self.action_client.wait_for_server()
        future = self.action_client.send_goal_async(goal_msg)
        self.spin_until_future(future)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        self.spin_until_future(result_future)
        result = result_future.result().result
        print(f"Result: {result.success}, Code: {result.error_code}, Msg: {result.message}")

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
            time.sleep(3)
            if i > 2:
                self.compute_calibration()
            print(f"sample no. {i} taken")   
           

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
