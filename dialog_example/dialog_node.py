import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
from std_srvs.srv import Trigger  # Standard service type for triggering actions

class TkinterROS(Node):
    counter = 0
    def __init__(self):
        super().__init__('tkinter_ros_node')

        # Set up ROS publisher
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)
        self.timer = self.create_timer(1.0, self.publish_message)

        # Create a ROS service client to request homing
        self.client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for homing service to become available...')

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

        # Use after to periodically update Tkinter UI in the main thread
        self.root.after(100, self.tk_mainloop)

    def publish_message(self):
        msg = String()
        msg.data = self.joints_entry.get()  # Send the value from the text field
        self.publisher.publish(msg)
        self.get_logger().warn('Publishing message: ' + msg.data)

    def on_homing_button_click(self):
        self.trigger_service("Homing in progress...", "Homing successful!", "Homing failed")

    def on_calibrate_button_click(self):
        self.trigger_service("Calibration in progress...", "Calibration successful!", "Calibration failed")

    def on_validate_button_click(self):
        self.trigger_service("Validation in progress...", "Validation successful!", "Validation failed")

    def on_init_button_click(self):
        self.initAR4()

    def trigger_service(self, start_msg, success_msg, failure_msg):
        self.label.config(text=start_msg)
        request = Trigger.Request()
        future = self.client.call_async(request)
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


def ros_spin(tkinter_ros):
    rclpy.spin(tkinter_ros)

def main():
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
