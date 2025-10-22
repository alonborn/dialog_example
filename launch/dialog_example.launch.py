from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import ThisLaunchFileDir
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_handeye_publisher_arg = DeclareLaunchArgument(
        "use_handeye_publisher",
        default_value="false",
        description="Whether to launch the handeye_publisher node"
    )
    use_handeye_publisher = LaunchConfiguration("use_handeye_publisher")

    aruco_params = os.path.join(
        get_package_share_directory("ros2_aruco"),
        "config",
        "aruco_parameters_ov5640.yaml",
    )

    dialog_node = Node(
        package='dialog_example',
        executable='dialog_node',
        name='dialog_node_gui',
        output='screen'
    )

    # Run ov5640_publisher with venv Python
    arm_cam_publisher = ExecuteProcess(
        cmd=[
            '/home/alon/venv/bin/python',
            '/home/alon/ros_ws/src/dialog_example/dialog_example/ov5640_publisher.py'
        ],
        name='ov5640_publisher_gui',
        output='screen'
    )

    move_ar_node = Node(
        package='ar_utils',
        executable='move_ar',
        name='move_ar_node',
        output='screen'
    )

    aruco_node = Node(
        package="ros2_aruco", executable="aruco_node", parameters=[aruco_params]
    )



    arm_cam_image_publisher = ExecuteProcess(
        cmd=[
            '/home/alon/venv/bin/python',
            '/home/alon/ros_ws/src/dialog_example/dialog_example/ov5640_image_publisher.py'
        ],
        name='ov5640_image_publisher_gui',
        output='screen'
    )

    hand_eye_tf_publisher = Node(
        package="ar4_hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{"calibration_name": "ar4_calibration"}],
        condition=IfCondition(use_handeye_publisher),
    )

    inferences_node = ExecuteProcess(
        cmd=[
            '/home/alon/venv/bin/python',
            '-m', 'detector.inferences'
        ],
        name='brick_inference_node',
        output='screen'
    )

    return LaunchDescription([
        use_handeye_publisher_arg,
        dialog_node,
        arm_cam_publisher,
        move_ar_node,
        aruco_node,
        hand_eye_tf_publisher,
        arm_cam_image_publisher
        # inferences_node
    ])
