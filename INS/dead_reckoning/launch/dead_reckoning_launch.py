#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # 패키지 경로
    pkg_share = FindPackageShare('dead_reckoning')
    
    # RViz 설정 파일 경로
    rviz_config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'dead_reckoning.rviz'
    ])
    
    # Launch arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='RViz를 실행할지 여부'
    )
    
    processed_topic_arg = DeclareLaunchArgument(
        'processed_topic',
        default_value='',
        description='전처리된 IMU 토픽 이름 (비어있으면 캘리브레이션 파일 사용)'
    )
    
    rotation_only_mode_arg = DeclareLaunchArgument(
        'rotation_only_mode',
        default_value='false',
        description='위치 변화를 무시하고 회전 드리프트만 시각화'
    )
    
    # Dead reckoning 노드
    dead_reckoning_node = Node(
        package='dead_reckoning',
        executable='dead_reckoning_node',
        name='dead_reckoning_node',
        output='screen',
        parameters=[{
            'processed_topic': LaunchConfiguration('processed_topic'),
            'rotation_only_mode': LaunchConfiguration('rotation_only_mode')
        }],
        remappings=[]
    )
    
    # RViz 노드
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        use_rviz_arg,
        processed_topic_arg,
        rotation_only_mode_arg,
        dead_reckoning_node,
        rviz_node
    ]) 