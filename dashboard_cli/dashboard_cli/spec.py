from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TopicSpec:
    name: str
    topic: str
    type: str
    min_hz: float = 0.0
    checks: Dict[str, float] = None


# Perception topics
TOPICS: List[TopicSpec] = [
    # Cameras
    TopicSpec(name='cam1', topic='/usb_cam_1/image_raw', type='sensor_msgs/msg/Image', min_hz=20.0),
    TopicSpec(name='cam2', topic='/usb_cam_2/image_raw', type='sensor_msgs/msg/Image', min_hz=20.0),
    # LiDAR
    TopicSpec(name='lidar', topic='/ouster/points', type='sensor_msgs/msg/PointCloud2', min_hz=10.0, checks={'min_points': 1000}),
    # Cones
    TopicSpec(name='cones_lidar', topic='/cone/lidar', type='custom_interface/msg/TrackedConeArray', min_hz=10.0),
    TopicSpec(name='cones_fused', topic='/cone/fused', type='custom_interface/msg/TrackedConeArray', min_hz=10.0),
    TopicSpec(name='cones_ukf', topic='/cone/fused/ukf', type='custom_interface/msg/TrackedConeArray', min_hz=10.0),
    # Localization
    TopicSpec(name='odom', topic='/odometry/filtered', type='nav_msgs/msg/Odometry', min_hz=20.0),

    # Planning
    TopicSpec(name='local_path', topic='/local_planned_path', type='nav_msgs/msg/Path', min_hz=10.0),
    TopicSpec(name='speed_profile', topic='/desired_speed_profile', type='std_msgs/msg/Float32MultiArray', min_hz=10.0),

    # Control
    TopicSpec(name='steer_cmd', topic='/cmd/steer', type='std_msgs/msg/Float32', min_hz=20.0),
    TopicSpec(name='steer_fb', topic='/ctrl/steer', type='std_msgs/msg/Float32', min_hz=10.0),
    TopicSpec(name='rpm_cmd', topic='/cmd/rpm', type='std_msgs/msg/Float32', min_hz=20.0),
    TopicSpec(name='rpm_target', topic='/target_rpm', type='std_msgs/msg/Int32', min_hz=50.0),
    TopicSpec(name='speed_fb', topic='/ctrl/speed', type='std_msgs/msg/Float32', min_hz=10.0),
    TopicSpec(name='throttle', topic='/throttle_data', type='throttle_msgs/msg/ThrottleData', min_hz=50.0),

    # Safety
    TopicSpec(name='aeb', topic='/aeb', type='std_msgs/msg/UInt8', min_hz=0.0),
]


# Expected nodes for liveness check (names only)
EXPECTED_NODES: List[str] = [
    # Perception
    'calico_ukf_tracking',
    'calico_multi_iou_fusion',
    'outlier_filter',
    'cone_detection_visualization_node',
    # Localization
    'ekf_fusion_node',
    # Planning / Control
    'pure_pursuit_static',
    'simple_speed_planner',
    'steering_control',
    'steering_feedback',
    'rpm_mux',
    'can_speed_sender',
    'can_speed_receiver',
    'throttle_serial',
]


# UI settings
UI_REFRESH_HZ: float = 10.0
SHOW_NODES: bool = True

