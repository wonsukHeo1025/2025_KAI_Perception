# ConeSTELLATION Topic Structure

## Overview
This document describes all ROS2 topics used by the cone_stellation SLAM system for input/output communication.

## Input Topics

### 1. `/fused_sorted_cones_ukf_sim` [custom_interface/msg/TrackedConeArray]
- **Publisher**: Cone detection/fusion system or simulation
- **Subscriber**: cone_slam_node
- **Description**: Tracked cone detections in sensor frame
- **Frame**: base_link
- **Rate**: 10-20 Hz
- **Content**:
  - Cone positions (x, y) in sensor frame
  - Cone colors (Yellow, Blue, Red, Orange, Unknown)
  - Track IDs for temporal association

### 2. `/odom` [nav_msgs/msg/Odometry]
- **Publisher**: Odometry system (wheel encoders, VIO, etc.)
- **Subscriber**: cone_slam_node
- **Description**: Vehicle odometry for motion prediction
- **Frame**: odom -> base_link
- **Rate**: 50-100 Hz
- **Content**:
  - Pose with covariance
  - Twist with covariance

### 3. `/imu/data` [sensor_msgs/msg/Imu] (Future)
- **Publisher**: IMU sensor
- **Subscriber**: cone_slam_node (not implemented yet)
- **Description**: IMU measurements for motion model
- **Frame**: imu_link
- **Rate**: 100-400 Hz

### 4. `/gps/fix` [sensor_msgs/msg/NavSatFix] (Future)
- **Publisher**: GPS receiver
- **Subscriber**: cone_slam_node (not implemented yet)
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

### Published by cone_slam_node:
- `map` -> `odom`: SLAM correction transform
- `map` -> `base_link_slam`: Direct SLAM pose estimate

### Required by cone_slam_node:
- `odom` -> `base_link`: From odometry source
- `base_link` -> `sensor_frame`: Static transform (if sensor not at base_link)

## Configuration

Topic names can be remapped in the launch file if needed:
```xml
<node pkg="cone_stellation" exec="cone_slam_node" name="cone_slam">
  <remap from="/fused_sorted_cones_ukf_sim" to="/your/cone_topic"/>
  <remap from="/odom" to="/your/odometry_topic"/>
</node>
```

## Notes

- All timestamps should be synchronized
- Cone detections must be in base_link frame or have valid TF available
- The system expects pre-tracked cones (with consistent IDs across frames)