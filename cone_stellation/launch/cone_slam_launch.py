#!/usr/bin/env python3
"""
Launch file for ConeSTELLATION SLAM
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('cone_stellation')
    config_file = os.path.join(pkg_dir, 'config', 'slam_config.yaml')
    
    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to SLAM configuration file'
    )
    
    # SLAM node
    slam_node = Node(
        package='cone_stellation',
        executable='cone_slam_node',
        name='cone_slam',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/fused_sorted_cones_ukf', '/fused_sorted_cones_ukf'),
            ('/odom', '/odom')
        ]
    )
    
    # RViz for visualization (optional)
    rviz_config = os.path.join(pkg_dir, 'config', 'slam_visualization.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=None  # Add condition to make optional
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        slam_node,
        # rviz_node  # Uncomment to auto-launch RViz
    ])