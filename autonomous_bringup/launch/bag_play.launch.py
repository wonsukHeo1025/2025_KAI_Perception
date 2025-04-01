import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    ld = LaunchDescription()

    # 각 패키지의 공유 디렉토리 경로 가져오기
    cone_detection_share_dir = get_package_share_directory('cone_detection')
    yolo_ros_share_dir = get_package_share_directory('yolo_ros')
    hungarian_association_share_dir = get_package_share_directory('hungarian_association')
    autonomous_bringup_share_dir = get_package_share_directory('autonomous_bringup')
    
    # 1. Bag 파일 재생은 사용자가 별도로 실행
    # 예: ros2 bag play /path/to/your/bagfile
    
    # 2. 웹캠 압축 이미지 데이터 압축해제 노드 런치
    republish_node = Node(
        package='image_transport',
        executable='republish',
        name='image_republish',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', 'image_raw/compressed'),
            ('out', 'image_raw/uncompressed')
        ]
    )

    # 3. 라이다 포인트 클라우드 전처리 및 클러스터링 패키지 런치
    cone_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(cone_detection_share_dir, 'launch'),
            '/cone_detection_launch.py'
        ])
    )

    # 4. YOLO 패키지 런치
    yolo_node = Node(
        package='yolo_ros',
        executable='yolo_debug_node',
        name='yolo_detector'
    )

    # 5. 헝가리안 매칭 퓨전 노드 런치
    hungarian_association_node = Node(
        package='hungarian_association',
        executable='hungarian_association_node',
        name='hungarian_association_node'
    )

    # 6. 각 디텍션 칼만 필터링 노드 런치
    kalman_filtering_node = Node(
        package='hungarian_association',
        executable='kalman_filtering_node',
        name='kalman_filtering_node'
    )

    # 7. 필터링된 콘 위치 rviz 시각화 노드 런치
    visualize_fused_cones_node = Node(
        package='hungarian_association',
        executable='visualize_fused_cones_rviz_marker_node',
        name='visualize_fused_cones_node'
    )

    # 8. RViz2 실행
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz'
    )
    
    # 9. 이후 슬램용 노드 추가 (TBD)
    # slam_node_1 = Node(
    #     package='simple_slam',
    #     executable='slam_node_1',
    #     name='slam_node_1'
    # )
    
    # slam_node_2 = Node(
    #     package='simple_slam',
    #     executable='slam_node_2',
    #     name='slam_node_2'
    # )

    # LaunchDescription에 모든 노드와 포함된 launch 파일 추가
    ld.add_action(republish_node)
    ld.add_action(cone_detection_launch)
    ld.add_action(yolo_node)
    ld.add_action(hungarian_association_node)
    ld.add_action(kalman_filtering_node)
    ld.add_action(visualize_fused_cones_node)
    ld.add_action(rviz_node)
    # ld.add_action(slam_node_1)
    # ld.add_action(slam_node_2)

    return ld
