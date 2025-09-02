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
    
    # Time sync and clock control
    time_sync_mode_arg = DeclareLaunchArgument(
        'time_sync_mode',
        default_value='arrival_ros',
        description='Time sync mode: header | arrival_ros | arrival_wall'
    )
    arrival_slop_arg = DeclareLaunchArgument(
        'arrival_slop',
        default_value='0.2',
        description='Arrival-time synchronization slop in seconds'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time from /clock'
    )
    override_fused_stamp_now_arg = DeclareLaunchArgument(
        'override_fused_stamp_now',
        default_value='true',
        description='Override fused output header.stamp with now()'
    )
    override_tracked_stamp_now_arg = DeclareLaunchArgument(
        'override_tracked_stamp_now',
        default_value='true',
        description='Override tracked output header.stamp with now()'
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
    time_sync_mode = LaunchConfiguration('time_sync_mode')
    arrival_slop = LaunchConfiguration('arrival_slop')
    use_sim_time = LaunchConfiguration('use_sim_time')
    override_fused_stamp_now = LaunchConfiguration('override_fused_stamp_now')
    override_tracked_stamp_now = LaunchConfiguration('override_tracked_stamp_now')
    
    # Multi-camera IoU fusion node (boundingbox branch)
    multi_iou_fusion_node = Node(
        package='calico',
        executable='multi_iou_fusion_node',
        name='calico_multi_iou_fusion',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'iou_threshold': iou_threshold,
            'enable_debug_viz': enable_debug_viz,
            'time_sync_mode': time_sync_mode,
            'arrival_slop': arrival_slop,
            'override_fused_stamp_now': override_fused_stamp_now,
            'use_sim_time': use_sim_time
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
            'max_association_distance': 0.7,
            'time_sync_mode': time_sync_mode,
            'arrival_slop': arrival_slop,
            'override_tracked_stamp_now': override_tracked_stamp_now,
            'use_sim_time': use_sim_time
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
            'frame_id': 'ouster_lidar',
            'use_sim_time': use_sim_time
        }]
    )
    
    # # Projection debug nodes (optional) - one for each camera
    # projection_debug_node_1 = Node(
    #     package='calico',
    #     executable='projection_debug_node',
    #     name='calico_projection_debug_camera_1',
    #     output='screen',
    #     condition=IfCondition(enable_debug_viz),
    #     parameters=[{
    #         'config_file': config_file,
    #         'camera_id': 'camera_1',
    #         'sync_tolerance': 0.1,
    #         'circle_radius': 5
    #     }],
    #     remappings=[
    #         ('/debug/projection_overlay', '/debug/camera_1/projection_overlay'),
    #     ]
    # )
    
    # projection_debug_node_2 = Node(
    #     package='calico',
    #     executable='projection_debug_node',
    #     name='calico_projection_debug_camera_2',
    #     output='screen',
    #     condition=IfCondition(enable_debug_viz),
    #     parameters=[{
    #         'config_file': config_file,
    #         'camera_id': 'camera_2',
    #         'sync_tolerance': 0.1,
    #         'circle_radius': 5
    #     }],
    #     remappings=[
    #         ('/debug/projection_overlay', '/debug/camera_2/projection_overlay'),
    #     ]
    # )
    
    return LaunchDescription([
        config_file_arg,
        iou_threshold_arg,
        use_imu_arg,
        show_track_ids_arg,
        enable_debug_viz_arg,
        debug_camera_id_arg,
        time_sync_mode_arg,
        arrival_slop_arg,
        use_sim_time_arg,
        override_fused_stamp_now_arg,
        override_tracked_stamp_now_arg,
        multi_iou_fusion_node,
        ukf_tracking_node,
        visualization_node,
        #projection_debug_node_1,
        #projection_debug_node_2
    ])
