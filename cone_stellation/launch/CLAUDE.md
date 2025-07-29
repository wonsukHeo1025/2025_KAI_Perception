# launch/ Directory

This directory contains ROS2 launch files for the cone_stellation SLAM system.

## Launch Files

### cone_slam_launch.py
Main launch file for the SLAM system:
- Launches cone_slam_node
- Loads SLAM configuration
- Sets up parameter files

### dummy_publisher_launch.py
Launch file for testing with simulated data:
- Launches dummy cone publisher
- Configures simulation parameters
- Can run standalone for testing TF/visualization

### test_slam_launch.py
Combined launch for integrated testing:
- Launches both dummy publisher and SLAM
- Includes topic remapping (/fused_sorted_cones_ukf_sim -> /lidar/cone_detection_cones)
- Checks config to enable/disable SLAM

### ekf_only_launch.py
Minimal launch file for robot_localization EKF with real bag file data:
- **GPS Converter Node**: Converts GPS coordinates to Cartesian (ENU) coordinates
  - Reference location: Konkuk University Ilgamho (37.540091°N, 127.076555°E, 39.5m altitude)
  - Publishes /gps/cartesian topic with converted coordinates
  - Publishes tf2 transform from map → gps frame
- **EKF Filter Node**: Fuses IMU and GPS data for localization
  - Uses ekf_config_real.yaml configuration
  - Subscribes to /imu/data and /gps/cartesian
  - Publishes /odometry/filtered with fused pose estimate
- **Static Transforms**: Complete TF tree structure
  - map → odom (identity transform, placeholder for future SLAM integration)
  - base_link → imu_link (0, 0, 0.1m offset)
  - base_link → gps_link (0, 0, 0.2m offset)

## Usage

```bash
# Run full test
ros2 launch cone_stellation test_slam_launch.py

# Run only SLAM (with real data)
ros2 launch cone_stellation cone_slam_launch.py

# Run only dummy publisher
ros2 launch cone_stellation dummy_publisher_launch.py

# Run EKF-only localization with real bag file data
ros2 launch cone_stellation ekf_only_launch.py
```

## Current Status

- Basic launch files created
- Topic remapping added for testing
- EKF-only launch ready for real bag file data with:
  - GPS to Cartesian conversion (Konkuk University reference)
  - Complete TF tree (map → odom → base_link → sensors)
  - Proper coordinate system setup for sensor fusion
- Missing launch configurations for:
  - Different sensor setups
  - Hardware integration
  - Multi-robot scenarios