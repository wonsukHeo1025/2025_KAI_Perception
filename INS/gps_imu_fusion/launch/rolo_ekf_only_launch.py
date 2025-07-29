#!/usr/bin/env python3
"""
Launch file for robot_localization EKF node with GPS converter.
Tests IMU+GPS fusion against GPS ground truth.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('gps_imu_fusion')
    imu_preprocess_pkg = FindPackageShare('imu_preprocess')
    
    # Configuration files - 자작 EKF와 동일한 설정 사용
    ekf_config = PathJoinSubstitution([
        pkg_share,
        'config',
        'rolo_ekf_config.yaml'
    ])
    
    # IMU calibration file path
    imu_calib_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'improved_imu_calibration.json'
    ])
    
    # IMU Preprocess Node
    imu_preprocess_node = Node(
        package='imu_preprocess',
        executable='imu_preprocess_node',
        name='imu_preprocess_node',
        output='screen',
        parameters=[{
            'calibration_file': imu_calib_file,
            'use_json_bias': True,
            'use_adaptive_filter': True
        }],
        remappings=[
            ('/imu/data', '/imu/data'),
            ('/imu/processed', '/imu/processed')
        ]
    )
    
    # GPS to Cartesian converter node - scripts 디렉토리의 버전 사용
    gps_converter_node = Node(
        package='gps_imu_fusion',
        executable='gps_to_cartesian.py',
        name='gps_to_cartesian_converter',
        output='screen',
        parameters=[{
            'reference_latitude': 37.540091,  # Konkuk University Ilgamho
            'reference_longitude': 127.076555,
            'reference_altitude': 39.5,
            'publish_tf': True,  # GPS TF 발행
            'world_frame': 'map',  # map 프레임 기준 (자작 EKF와 동일)
            'child_frame': 'gps'
        }]
    )
    
    # Robot localization EKF node
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config],
        remappings=[
            # 입력 토픽 (자작 EKF와 동일)
            ('imu0', '/imu/processed'),
            ('pose0', '/gps/pose'),
            ('twist0', '/ublox_gps_node/fix_velocity'),
            # 출력 토픽
            ('odometry/filtered', '/odometry/filtered'),
            ('accel/filtered', '/accel/filtered'),
        ]
    )
    
    # Static transform publishers for robot's physical structure
    
    # base_link -> imu_link
    base_to_imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu_tf',
        output='screen',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'imu_link']
    )
    
    # base_link -> gps
    base_to_gps_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_gps_tf',
        output='screen',
        arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'gps']
    )
    
    # map -> odom (identity transform for now, SLAM will update this later)
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    
    return LaunchDescription([
        # IMU preprocess node
        imu_preprocess_node,
        
        # GPS converter
        gps_converter_node,
        
        # EKF node
        ekf_node,
        
        # Static TF nodes
        map_to_odom_tf,
        base_to_imu_tf,
        base_to_gps_tf,
    ])