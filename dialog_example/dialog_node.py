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
        self.publisher = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.publish_message)
        
         # Create a ROS service client to request homing
        self.client = self.create_client(Trigger, '/ar4_hardware_interface_node/homing')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for homing service to become available...')

        # Set up ROS publisher
        self.publisher = self.create_publisher(String, '/ar4_hardware_interface_node/homing_string', 10)

        # Set up Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Tkinter and ROS 2")

        # Create a label
        self.label = tk.Label(self.root, text="Waiting for messages...")
        self.label.pack()

        # Create a button that will trigger the publish_message function when clicked
        self.button = tk.Button(self.root, text="Publish Message", command=self.on_button_click)
        self.button.pack()

        # Use after to periodically update Tkinter UI in the main thread
        self.root.after(100, self.tk_mainloop)

    def publish_message(self):
        msg = String()
        msg.data = (str(self.counter))
        self.publisher.publish(msg)
        self.counter += 1
        self.get_logger().warn('publishing message' + msg.data)

    def on_button_click(self):
        """Called when the button is clicked - triggers homing service."""
        self.label.config(text="Homing in progress...")

        # Create a service request
        request = Trigger.Request()

        # Call the homing service asynchronously
        future = self.client.call_async(request)
        future.add_done_callback(self.homing_response_callback)

    def homing_response_callback(self, future):
        """Handles the response from the homing service."""
        try:
            response = future.result()
            if response.success:
                self.update_gui("Homing successful!")
            else:
                self.update_gui(f"Homing failed: {response.message}")
        except Exception as e:
            self.update_gui(f"Service call failed: {str(e)}")
            
    def update_gui(self, message):
        # Update the Tkinter label in the main thread using `after`
        self.label.config(text=message)

    def tk_mainloop(self):
        # Run Tkinter's event loop
        self.root.update_idletasks()
        self.root.update()

    def on_shutdown(self):
        self.root.quit()  # Gracefully shutdown Tkinter when the node is shutdown

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
