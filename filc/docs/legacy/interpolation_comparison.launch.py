import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 파라미터 선언
    enable_test_arg = DeclareLaunchArgument(
        'enable_test',
        default_value='true',
        description='Enable test interpolation node'
    )
    
    enable_image_arg = DeclareLaunchArgument(
        'enable_image',
        default_value='true',
        description='Enable image-based interpolation node'
    )
    
    enable_spherical_arg = DeclareLaunchArgument(
        'enable_spherical',
        default_value='true',
        description='Enable spherical interpolation node'
    )
    
    scale_factor_arg = DeclareLaunchArgument(
        'scale_factor',
        default_value='4.0',
        description='Interpolation scale factor'
    )
    
    # 테스트 보간 노드 (간단한 선형)
    test_interpolation_node = Node(
        package='filc',
        executable='test_interpolation_node',
        name='test_interpolation_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_test')),
        parameters=[{
            'input_topic': '/ouster/points',
            'output_topic': '/ouster/test_interpolated_points',
            'scale_factor': LaunchConfiguration('scale_factor'),
        }],
        prefix=['nice -n 10']  # 낮은 우선순위
    )
    
    # 이미지 기반 보간 노드
    image_interpolation_node = Node(
        package='filc',
        executable='image_interpolation_node',
        name='image_interpolation_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_image')),
        parameters=[{
            'scale_factor': LaunchConfiguration('scale_factor'),
            'interpolation_method': 'bicubic',
            'process_range': True,
            'process_intensity': True,
            'process_pointcloud': True,
        }]
    )
    
    # 구면 좌표계 기반 정밀 보간 노드
    spherical_interpolation_node = Node(
        package='filc',
        executable='spherical_interpolation_node',
        name='spherical_interpolation_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_spherical')),
        parameters=[{
            'input_topic': '/ouster/points',
            'output_topic': '/ouster/spherical_interpolated_points',
            'scale_factor': LaunchConfiguration('scale_factor'),
            'use_adaptive_interpolation': True,
            'validate_interpolation': True,
            'variance_threshold': 0.25,
            'num_threads': 0,  # auto
        }]
    )
    
    # 시각화 노드
    visualizer_node = Node(
        package='filc',
        executable='visualize_interpolation.py',
        name='interpolation_visualizer',
        output='screen',
        parameters=[{
            'original_topic': '/ouster/points',
            'interpolated_topic': '/ouster/spherical_interpolated_points',
        }]
    )
    
    # RViz2 (선택적)
    rviz_config_file = os.path.join(
        get_package_share_directory('filc'), 'rviz', 'interpolation_comparison.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('launch_rviz'))
    )
    
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz2'
    )
    
    return LaunchDescription([
        # 파라미터 선언
        enable_test_arg,
        enable_image_arg,
        enable_spherical_arg,
        scale_factor_arg,
        launch_rviz_arg,
        
        # 노드 그룹
        GroupAction([
            test_interpolation_node,
            image_interpolation_node,
            spherical_interpolation_node,
            visualizer_node,
            rviz_node,
        ])
    ])