from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dialog_example',
            executable='dialog_node',
            name='dialog_node',
            output='screen'
        ),
        Node(
            package='ar_utils',
            executable='move_ar2',
            name='move_ar2_node',
            output='screen'
        )
    ])
