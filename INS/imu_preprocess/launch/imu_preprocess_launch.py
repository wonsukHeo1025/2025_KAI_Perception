from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 패키지 경로 찾기
    pkg_dir = get_package_share_directory('imu_preprocess')
    
    # Launch 인자 선언
    use_json_bias_arg = DeclareLaunchArgument(
        'use_json_bias',
        default_value='true',
        description='Use bias from JSON calibration file'
    )
    
    use_adaptive_filter_arg = DeclareLaunchArgument(
        'use_adaptive_filter',
        default_value='true',
        description='Use Allan variance based adaptive filtering'
    )
    
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=os.path.join(pkg_dir, 'config', 'improved_imu_calibration.json'),
        description='Path to IMU calibration JSON file'
    )
    
    # IMU preprocess 노드
    imu_preprocess_node = Node(
        package='imu_preprocess',
        executable='imu_preprocess_node',
        name='imu_preprocess',
        output='screen',
        parameters=[{
            'calib_duration': 20.0,
            'lpf_cutoff': 15.0,
            'use_json_bias': LaunchConfiguration('use_json_bias'),
            'use_adaptive_filter': LaunchConfiguration('use_adaptive_filter'),
            'calibration_file': LaunchConfiguration('calibration_file'),
            'bias_window_size': 100
        }],
        remappings=[
            ('/imu/data', '/imu/data'),
            ('/imu/processed', '/imu/processed')
        ]
    )
    
    return LaunchDescription([
        use_json_bias_arg,
        use_adaptive_filter_arg,
        calibration_file_arg,
        imu_preprocess_node
    ])