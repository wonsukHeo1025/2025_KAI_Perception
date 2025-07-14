# robot_localization_launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # 사용할 설정 파일의 경로를 지정합니다.
    # your_robot_package를 실제 패키지 이름으로, my_robot_localization.yaml을 실제 파일 이름으로 변경하세요.
    config_file_path = os.path.join(
        get_package_share_directory('gps_imu_fusion'), # <<-- 실제 패키지 이름으로 변경!
        'config',
        'my_robot_localization.yaml' # <<-- 실제 YAML 파일 이름으로 변경!
    )

    # Launch argument로 파라미터 파일 경로를 받을 수도 있습니다.
    # params_file_arg = DeclareLaunchArgument(
    #     'params_file',
    #     default_value=config_file_path,
    #     description='Path to the robot_localization parameters file'
    # )
    # params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        # params_file_arg, # Launch argument를 사용할 경우 주석 해제

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config_file_path], # 또는 [params_file]
            remappings=[] # 필요한 경우 여기에 토픽 리매핑 추가
                          # 예: [('odometry/filtered', 'my_odometry/filtered')]
        ),

        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[config_file_path], # 또는 [params_file]
            remappings=[ # navsat_transform_node가 구독/발행하는 토픽 이름 변경
                        ('imu', '/imu/data_raw'),             # <<-- 실제 IMU 토픽 이름으로 변경!
                        ('gps/fix', '/ublox_gps_node/fix'),     # <<-- 실제 GPS fix 토픽 이름으로 변경!
                        # ('odometry/gps', 'odometry/gps'), # 기본 출력 토픽 (ekf_node의 odom0 입력)
                        # ('gps/filtered', 'gps/filtered_fix') # 기본 출력 토픽
                       ]
        ),

        # base_link to imu_link static transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_base_to_imu',
            arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'imu_link']
            # 위 숫자는 실제 로봇에서 base_link 기준 imu_link의 x, y, z, roll, pitch, yaw 오프셋입니다.
        ),
        # base_link to gps static transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_base_to_gps',
            arguments=['-0.2', '0', '0.3', '0', '0', '0', 'base_link', 'gps']
            # 위 숫자는 실제 로봇에서 base_link 기준 gps 프레임의 x, y, z, roll, pitch, yaw 오프셋입니다.
        ),

    ])