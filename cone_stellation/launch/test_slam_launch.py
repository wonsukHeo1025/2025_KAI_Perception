from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def launch_setup(context, *args, **kwargs):
    """Setup function that reads config to determine what to launch"""
    package_dir = get_package_share_directory('cone_stellation')
    
    # Configuration files
    slam_config = os.path.join(package_dir, 'config', 'slam_config.yaml')
    dummy_config = os.path.join(package_dir, 'config', 'dummy_publisher_config.yaml')
    
    # Read dummy publisher config to check use_slam setting
    use_slam = True  # default
    with open(dummy_config, 'r') as f:
        config = yaml.safe_load(f)
        use_slam = config.get('dummy_publisher', {}).get('ros__parameters', {}).get('use_slam', True)
    
    nodes = []
    
    # Always launch dummy publisher
    dummy_publisher = Node(
        package='cone_stellation',
        executable='dummy_publisher_node.py',
        name='dummy_publisher',
        parameters=[dummy_config],
        output='screen'
    )
    nodes.append(dummy_publisher)
    
    # Launch SLAM node only if use_slam is true in config
    if use_slam:
        slam_node = Node(
            package='cone_stellation',
            executable='cone_slam_node',
            name='cone_slam',
            parameters=[slam_config],
            output='screen'
        )
        nodes.append(slam_node)
    
    return nodes

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])