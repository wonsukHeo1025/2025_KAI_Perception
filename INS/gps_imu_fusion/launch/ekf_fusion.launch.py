from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    param_dir = LaunchConfiguration(
        'param_dir',
        default=os.path.join(
            get_package_share_directory('gps_imu_fusion'),
            'config',
            'fusion_params.yaml'))
            
    declare_param_dir = DeclareLaunchArgument(
        'param_dir',
        default_value=param_dir,
        description='Full path to parameter file')
    
    ekf_fusion_node = Node(
        package='gps_imu_fusion',
        executable='ekf_fusion_node',
        name='ekf_fusion_node',
        parameters=[param_dir],
        output='screen'
    )
    
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_imu_to_base',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link']
    )
    
    static_tf_node2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_gps_to_base',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'gps']
    )
    
    return LaunchDescription([
        declare_param_dir,
        ekf_fusion_node,
        static_tf_node,
        static_tf_node2
    ])