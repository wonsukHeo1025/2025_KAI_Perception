#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Perception Vision Pipeline Launch File
    
    Includes:
    - YOLO dual camera node
    - LiDAR point cloud interpolation (prism)
    - LiDAR cone detection
    - Camera-LiDAR fusion (calico)
    """
    
    # YOLO dual camera node
    yolo_dual_camera_node = Node(
        package='yolo_ros',
        executable='yolo_dual_camera_node',
        name='yolo_dual_camera_node',
        output='screen',
        parameters=[],
    )
    
    # LiDAR point cloud interpolation (prism)
    prism_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('prism'),
                'launch',
                'prism.launch.py'
            ])
        ])
    )
    
    # LiDAR cone detection
    cone_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('cone_detection'),
                'launch',
                'cone_detection_launch.py'
            ])
        ])
    )
    
    # Camera-LiDAR fusion (calico)
    calico_full_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('calico'),
                'launch',
                'calico_full.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        yolo_dual_camera_node,
        prism_launch,
        cone_detection_launch,
        calico_full_launch,
    ])
