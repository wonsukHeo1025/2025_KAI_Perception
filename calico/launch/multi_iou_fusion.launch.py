#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Launch multi-camera IoU fusion node."""
    
    # Get package directories
    calico_dir = get_package_share_directory('calico')
    # hungarian_dir = get_package_share_directory('hungarian_association')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(calico_dir, 'config', 'multi_hungarian_config.yaml'),
        description='Path to multi-camera configuration file'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.1',
        description='Minimum IoU threshold for valid matches'
    )
    
    enable_debug_viz_arg = DeclareLaunchArgument(
        'enable_debug_viz',
        default_value='true',
        description='Enable debug visualization overlay for each camera'
    )
    
    # Multi-camera IoU Fusion Node
    multi_iou_fusion_node = Node(
        package='calico',
        executable='multi_iou_fusion_node',
        name='multi_iou_fusion_node',
        output='screen',
        parameters=[
            {
                'config_file': LaunchConfiguration('config_file'),
                'iou_threshold': LaunchConfiguration('iou_threshold'),
                'enable_debug_viz': LaunchConfiguration('enable_debug_viz'),
            }
        ],
        remappings=[
            # The node will read topic names from config file
            # These are just defaults that can be overridden
            ('/cone/lidar/box', '/cone/lidar/box'),
            ('/fused_sorted_cones', '/fused_sorted_cones'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        iou_threshold_arg,
        enable_debug_viz_arg,
        multi_iou_fusion_node
    ])