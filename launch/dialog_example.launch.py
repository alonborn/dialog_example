from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dialog_example',
            executable='dialog_node',
            name='dialog_node_gui',
            output='screen'
        ),
        # Node(
        #     package='ar_utils',
        #     executable='move_ar2',
        #     name='move_ar2_node',
        #     output='screen'
        # ),
        # Node(
        #     package='ar_utils',
        #     executable='move_ar2',
        #     name='move_ar2_node',
        #     output='screen'
        # ),
        
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('ar4_hand_eye_calibration'),
                    'launch',
                    'visualize.launch.py'
                )
            )
        )

    ])
