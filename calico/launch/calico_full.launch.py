from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    calico_share_dir = get_package_share_directory('calico')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            calico_share_dir, 'config',
            'multi_hungarian_config.yaml'
        ),
        description='Path to the configuration file'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.01',
        description='Minimum IoU threshold for valid matches'
    )
    
    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='true',
        description='Whether to use IMU data for tracking'
    )
    
    show_track_ids_arg = DeclareLaunchArgument(
        'show_track_ids',
        default_value='true',
        description='Whether to show track IDs in RViz'
    )
    
    enable_debug_viz_arg = DeclareLaunchArgument(
        'enable_debug_viz',
        default_value='true',
        description='Enable projection debug visualization'
    )
    
    debug_camera_id_arg = DeclareLaunchArgument(
        'debug_camera_id',
        default_value='camera_1',
        description='Camera ID for debug visualization'
    )
    
    # Get configuration values
    config_file = LaunchConfiguration('config_file')
    iou_threshold = LaunchConfiguration('iou_threshold')
    use_imu = LaunchConfiguration('use_imu')
    show_track_ids = LaunchConfiguration('show_track_ids')
    enable_debug_viz = LaunchConfiguration('enable_debug_viz')
    debug_camera_id = LaunchConfiguration('debug_camera_id')
    
    # Multi-camera IoU fusion node (boundingbox branch)
    multi_iou_fusion_node = Node(
        package='calico',
        executable='multi_iou_fusion_node',
        name='calico_multi_iou_fusion',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'iou_threshold': iou_threshold,
            'enable_debug_viz': enable_debug_viz
        }]
    )
    
    # UKF tracking node
    ukf_tracking_node = Node(
        package='calico',
        executable='ukf_tracking_node',
        name='calico_ukf_tracking',
        output='screen',
        parameters=[{
            'use_imu': use_imu,
            'q_pos': 0.1,
            'q_vel': 0.1,
            'r_pos': 0.1,
            'max_age_before_deletion': 4,
            'min_hits_before_confirmation': 3,
            'max_association_distance': 0.7
        }]
    )
    
    # Visualization node
    visualization_node = Node(
        package='calico',
        executable='visualization_node',
        name='calico_visualization',
        output='screen',
        parameters=[{
            'show_track_ids': show_track_ids,
            'show_color_labels': False,
            'cone_height': 0.5,
            'cone_radius': 0.15,
            'frame_id': 'ouster_lidar'
        }]
    )
    
    # Projection debug nodes (optional) - one for each camera
    projection_debug_node_1 = Node(
        package='calico',
        executable='projection_debug_node',
        name='calico_projection_debug_camera_1',
        output='screen',
        condition=IfCondition(enable_debug_viz),
        parameters=[{
            'config_file': config_file,
            'camera_id': 'camera_1',
            'sync_tolerance': 0.1,
            'circle_radius': 5
        }],
        remappings=[
            ('/debug/projection_overlay', '/debug/camera_1/projection_overlay'),
        ]
    )
    
    projection_debug_node_2 = Node(
        package='calico',
        executable='projection_debug_node',
        name='calico_projection_debug_camera_2',
        output='screen',
        condition=IfCondition(enable_debug_viz),
        parameters=[{
            'config_file': config_file,
            'camera_id': 'camera_2',
            'sync_tolerance': 0.1,
            'circle_radius': 5
        }],
        remappings=[
            ('/debug/projection_overlay', '/debug/camera_2/projection_overlay'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        iou_threshold_arg,
        use_imu_arg,
        show_track_ids_arg,
        enable_debug_viz_arg,
        debug_camera_id_arg,
        multi_iou_fusion_node,
        ukf_tracking_node,
        visualization_node,
        projection_debug_node_1,
        projection_debug_node_2
    ])