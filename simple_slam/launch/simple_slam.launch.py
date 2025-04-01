#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    """Launch IMU odometry node and cone mapper node with ordered execution."""
    
    # Launch arguments
    publish_fallback_tf_arg = DeclareLaunchArgument(
        'publish_fallback_tf',
        default_value='false',
        description='Whether to publish a fallback TF if needed'
    )
    
    # IMU Odometry Node - Start first
    imu_odometry_node = Node(
        package='simple_slam',
        executable='imu_odometry',
        name='imu_odometry_node',
        output='screen',
        parameters=[
            {
                'base_frame': 'os_sensor',  # 베이스 프레임을 os_sensor로 설정
                'imu_topic': '/ouster/imu',
                'use_orientation_for_gravity': False,  # orientation 기반 보정 비활성화
                'apply_simple_gravity_compensation': True,  # 단순 Z축 보정 활성화
                'gravity': 9.80665  # 중력 가속도 값 (필요에 따라 조정)
            }
        ]
    )
    
    # Cone Mapper Node - Start after IMU odometry with a delay
    cone_mapper_node = Node(
        package='simple_slam',
        executable='cone_mapper',
        name='cone_mapper_node',
        output='screen',
        parameters=[
            {
                'publish_fallback_tf': LaunchConfiguration('publish_fallback_tf'),
                'input_topic': '/fused_sorted_cones_ukf',
                'marker_topic': '/mapped_cones_markers'
            }
        ]
    )
    
    # Delay cone_mapper to ensure IMU odometry is running
    delayed_cone_mapper = TimerAction(
        period=1.0,  # 1초 지연
        actions=[cone_mapper_node]
    )
    
    return LaunchDescription([
        publish_fallback_tf_arg,
        imu_odometry_node,
        delayed_cone_mapper
    ]) 