#!/usr/bin/env python3
"""
Launch file for cone_stellation SLAM only (assumes EKF is already running)
This is useful for testing SLAM independently with EKF data
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Find packages
    cone_stellation_pkg = FindPackageShare('cone_stellation')
    gps_imu_fusion_pkg = FindPackageShare('gps_imu_fusion')
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time for bag file playback'
    )
    
    slam_config_arg = DeclareLaunchArgument(
        'slam_config',
        default_value=PathJoinSubstitution([
            cone_stellation_pkg,
            'config',
            'slam_config.yaml'
        ]),
        description='Path to SLAM configuration file'
    )
    
    # Launch SLAM node directly
    slam_node = Node(
        package='cone_stellation',
        executable='cone_slam_node',
        name='cone_slam',
        output='screen',
        parameters=[
            LaunchConfiguration('slam_config'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        slam_config_arg,
        slam_node
    ])