import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('cone_detection'),
        'config',
        'cone_detection_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='cone_detection',
            executable='cone_detection_node',
            name='cone_detection',
            parameters=[config_file]
        )
    ])
