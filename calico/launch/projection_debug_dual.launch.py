#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    package_dir = get_package_share_directory('calico')
    default_config = os.path.join(package_dir, 'config', 'multi_hungarian_config.yaml')
    
    # Declare launch arguments
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to the configuration file'
    )
    
    declare_sync_tolerance_cmd = DeclareLaunchArgument(
        'sync_tolerance',
        default_value='0.1',
        description='Time synchronization tolerance in seconds'
    )
    
    declare_circle_radius_cmd = DeclareLaunchArgument(
        'circle_radius',
        default_value='5',
        description='Radius of projected points in pixels'
    )
    
    # Create projection debug node for camera_1
    projection_debug_node_1 = Node(
        package='calico',
        executable='projection_debug_node',
        name='projection_debug_node_camera_1',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'camera_id': 'camera_1',
            'sync_tolerance': LaunchConfiguration('sync_tolerance'),
            'circle_radius': LaunchConfiguration('circle_radius'),
        }],
        remappings=[
            ('/debug/projection_overlay', '/debug/camera_1/projection_overlay'),
        ]
    )
    
    # Create projection debug node for camera_2
    projection_debug_node_2 = Node(
        package='calico',
        executable='projection_debug_node',
        name='projection_debug_node_camera_2',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'camera_id': 'camera_2',
            'sync_tolerance': LaunchConfiguration('sync_tolerance'),
            'circle_radius': LaunchConfiguration('circle_radius'),
        }],
        remappings=[
            ('/debug/projection_overlay', '/debug/camera_2/projection_overlay'),
        ]
    )
    
    # Create and return launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_sync_tolerance_cmd)
    ld.add_action(declare_circle_radius_cmd)
    
    # Add nodes
    ld.add_action(projection_debug_node_1)
    ld.add_action(projection_debug_node_2)
    
    return ld