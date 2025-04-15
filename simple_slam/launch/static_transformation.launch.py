from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # os_sensor -> os_imu
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_sensor_imu',
            arguments=[
                '--x', '0.006253', '--y', '-0.011775', '--z', '0.007645',
                '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                '--frame-id', 'os_sensor', '--child-frame-id', 'os_imu'
            ],
            output='screen'
        ),

        # os_sensor -> os_lidar
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_sensor_lidar',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '0.038195',
                '--qx', '0', '--qy', '0', '--qz', '1', '--qw', '0',
                '--frame-id', 'os_sensor', '--child-frame-id', 'os_lidar'
            ],
            output='screen'
        ),
    ])