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
    - LiDAR driver (ouster_ros)
    - Camera drivers (usb_cam_1, usb_cam_2)
    - YOLO dual camera node
    - LiDAR point cloud interpolation (prism)
    - LiDAR cone detection
    - Camera-LiDAR fusion (calico)
    """
    
    # LiDAR driver (ouster_ros)
    lidar_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ouster_ros'),
                'launch',
                'driver.launch.py'
            ])
        ]),
        launch_arguments={
            'params_file': '/home/kai/KAI_ws/src/ouster-ros/ouster-ros/config/driver_params.yaml'
        }.items()
    )
    
    # Camera driver 1 (usb_cam_1)
    usbcam1_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node_exe',
        namespace='usb_cam_1',
        output='screen',
        parameters=['/home/kai/KAI_ws/src/Perception/usb_cam/config/params_1.yaml']
    )
    
    # Camera driver 2 (usb_cam_2)
    usbcam2_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node_exe',
        namespace='usb_cam_2',
        output='screen',
        parameters=['/home/kai/KAI_ws/src/Perception/usb_cam/config/params_2.yaml']
    )
    
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
        lidar_driver_launch,
        usbcam1_node,
        usbcam2_node,
        yolo_dual_camera_node,
        prism_launch,
        cone_detection_launch,
        calico_full_launch,
    ])
