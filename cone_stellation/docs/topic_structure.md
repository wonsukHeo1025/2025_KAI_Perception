# ConeSTELLATION Topic Structure

## Overview
This document describes all ROS2 topics used by the cone_stellation SLAM system for input/output communication.

## Current System Status
- ✅ **Fully Operational**: SLAM + EKF fusion working with all topics below
- ✅ **TF Tree Complete**: map → odom → base_link with proper transforms
- ✅ **100Hz Odometry**: EKF fusion provides high-rate odometry
- ✅ **Rosbag Compatible**: Works with recorded sensor data

## Input Topics

### 1. `/cones/fused/ukf` [custom_interface/msg/TrackedConeArray] ✅ 권장
- **Publisher**: Cone detection/fusion system or simulation
- **Subscriber**: cone_slam_node
- **Description**: Tracked cone detections in ORIGINAL sensor frame
- **Frame**: os_sensor (또는 실제 센서 프레임)
- **Rate**: 10-20 Hz
- **Content**:
  - Cone positions (x, y) in sensor frame (원본)
  - Cone colors (Yellow, Blue, Red, Orange, Unknown)
  - Track IDs for temporal association

참고: SLAM 노드는 수신 관측을 TF로 `base_link` 기준 상대좌표로 변환한 뒤 요인 그래프에 사용합니다. 코드 근거:
```208:279:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/src/cone_stellation/ros/cone_slam_node.cpp
os_to_base_tf = tf_buffer_.lookupTransform("base_link", msg->header.frame_id, tf2::TimePointZero);
// ... T_base_sensor * cone_sensor → base_link 상대좌표 obs.position
```

### 1-ALT. `/cones/fused/ukf/map` [custom_interface/msg/TrackedConeArray] (비권장/주의)
- **Publisher**: 변환 노드(로컬 카르테시안 → 사용자 기준점)
- **Subscriber**: cone_slam_node (기본 로직과 충돌 가능)
- **Description**: Map 또는 로컬 카르테시안 좌표로 변환된 콘
- **Frame**: map (또는 사용자 정의)
- **Rate**: 10-20 Hz
- **주의**:
  - 현재 SLAM 로직은 수신 관측을 차량 기준 상대좌표로 사용합니다. `map` 프레임으로 들어오면 다시 `base_link`로 변환되어 의미가 왜곡됩니다(월드 좌표를 센서 좌표로 오인).
  - 이 토픽을 사용하려면, 코드에서 “관측을 차량 상대좌표로 사용”하는 경로를 비활성/분기해야 합니다(예: 이미 월드 좌표인 경우 별도 월드-관측 팩터 사용).

### 2. `/odometry/filtered` [nav_msgs/msg/Odometry] ✅
- **Publisher**: EKF fusion node (robot_localization)
- **Subscriber**: cone_slam_node
- **Description**: Vehicle odometry for motion prediction
- **Frame**: odom -> base_link
- **Rate**: 50-100 Hz
- **Content**:
  - Pose with covariance
  - Twist with covariance

### 3. `/ouster/imu` [sensor_msgs/msg/Imu] ✅
- **Publisher**: IMU sensor / imu_gps_publishers.py
- **Subscriber**: EKF fusion node (robot_localization)
- **Description**: IMU measurements for motion model
- **Frame**: imu_link
- **Rate**: 100-400 Hz

### 4. `/ublox_gps_node/fix` [sensor_msgs/msg/NavSatFix] ✅
- **Publisher**: GPS receiver / imu_gps_publishers.py
- **Subscriber**: EKF fusion node (robot_localization)
- **Description**: Global position for loop closure
- **Rate**: 1-10 Hz

## Output Topics

### 1. `/slam/landmarks` [visualization_msgs/msg/MarkerArray]
- **Publisher**: cone_slam_node
- **Subscriber**: RViz, other visualization tools
- **Description**: Optimized cone landmarks in map frame
- **Frame**: map
- **Rate**: 10 Hz
- **Content**:
  - Cone positions as cylinders
  - Cone IDs as text labels
  - Colors matching cone types

### 2. `/slam/factor_graph` [visualization_msgs/msg/MarkerArray]
- **Publisher**: cone_slam_node
- **Subscriber**: RViz, other visualization tools
- **Description**: Factor graph edges for debugging
- **Frame**: map
- **Rate**: 10 Hz
- **Content**:
  - Green lines: Odometry factors (pose-to-pose)
  - Blue lines: Observation factors (pose-to-landmark)
  - Red lines: Inter-landmark factors (landmark-to-landmark)

### 3. `/slam/pose` [geometry_msgs/msg/PoseStamped]
- **Publisher**: cone_slam_node
- **Subscriber**: Navigation stack, controller
- **Description**: Current vehicle pose estimate
- **Frame**: map
- **Rate**: 10 Hz

### 4. `/slam/path` [nav_msgs/msg/Path]
- **Publisher**: cone_slam_node
- **Subscriber**: RViz, path planning
- **Description**: Full trajectory history
- **Frame**: map
- **Rate**: 10 Hz

### 5. `/slam/keyframes` [visualization_msgs/msg/MarkerArray]
- **Publisher**: cone_slam_node
- **Subscriber**: RViz, debugging tools
- **Description**: Keyframe poses visualization
- **Frame**: map
- **Rate**: 10 Hz
- **Content**:
  - Cyan arrows: Keyframe poses with orientation
  - Text labels: Keyframe IDs (KF0, KF1, ...)

### 6. `/slam/map` [nav_msgs/msg/OccupancyGrid] (Future)
- **Publisher**: cone_slam_node
- **Description**: 2D occupancy grid for navigation

## TF Transforms

### Published by cone_slam_node: ✅
- `map` -> `odom`: SLAM correction transform (drift correction)

### Published by EKF fusion node: ✅
- `odom` -> `base_link`: Fused odometry at 100Hz

### Required by cone_slam_node:
- `odom` -> `base_link`: From odometry source
- `base_link` -> `sensor_frame`: Static transform (if sensor not at base_link)

## Configuration

Topic names can be remapped in the launch file if needed:
```xml
<node pkg="cone_stellation" exec="cone_slam_node" name="cone_slam">
  <remap from="/cones/for_sim" to="/your/cone_topic"/>
  <remap from="/odom" to="/your/odometry_topic"/>
</node>
```

## Notes

- All timestamps should be synchronized
- Cone detections should be provided in ORIGINAL sensor frame with valid TF to `base_link` (권장: `/cones/fused/ukf`)
- The system expects pre-tracked cones (with consistent IDs across frames)