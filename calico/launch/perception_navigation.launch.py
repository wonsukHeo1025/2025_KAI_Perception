#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Perception Navigation Pipeline Launch File
    
    Includes:
    - GPS node (ublox_gps)
    - fix2nmea converter node
    - IMU driver (myahrs_ros2_driver)
    - tf_static
    - Velocity magnitude node
    """
    
    # GPS node (ublox_gps)
    ublox_gps_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ublox_gps'),
                'launch',
                'ublox_gps_node-launch.py'
            ])
        ])
    )
    
    # fix2nmea converter node
    fix2nmea_node = Node(
        package='fix2nmea',
        executable='fix2nmea',
        name='fix2nmea',
        output='screen',
    )
    
    # IMU driver (myahrs_ros2_driver)
    myahrs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('myahrs_ros2_driver'),
                'launch',
                'myahrs_ros2_driver.launch.py'
            ])
        ])
    )
    
    # tf_static
    tf_static_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gps_imu_fusion'),
                'launch',
                'tf_static.launch.py'
            ])
        ])
    )
    
    # Velocity magnitude node
    velocity_magnitude_node = Node(
        package='gps_imu_fusion',
        executable='velocity_magnitude_node.py',
        name='velocity_magnitude_node',
        output='screen',
    )
    
    return LaunchDescription([
        ublox_gps_launch,
        fix2nmea_node,
        myahrs_launch,
        tf_static_launch,
        velocity_magnitude_node,
    ])
