# config/ Directory

This directory contains configuration files for the cone_stellation SLAM system.

## Files

### slam_config.yaml
Main SLAM configuration including:
- Preprocessing parameters (max distance, confidence thresholds)
- Mapping parameters (optimization frequency, factor weights)
- Keyframe selection criteria
- Inter-landmark factor settings
- Data association thresholds
- Loop closure parameters

### dummy_publisher_config.yaml
Test data publisher configuration:
- Simulation parameters
- Noise models
- Track scenarios (circular, figure-8, complex)
- Sensor simulation settings
- IMU/GPS simulation parameters

### cone_slam.rviz
RViz configuration for visualizing:
- Cone landmarks (colored by type)
- Factor graph edges (observation, odometry, inter-landmark, loop closure)
- Vehicle trajectory
- TF tree (map → odom → base_link)
- Tentative landmarks
- Optimization visualization

## Usage

Configuration files are loaded at runtime via ROS2 parameters:
```bash
ros2 run cone_stellation cone_slam_node --ros-args --params-file config/slam_config.yaml
```

For testing with dummy publisher:
```bash
ros2 launch cone_stellation test_slam_launch.py
```

## Current Status

- ✅ SLAM parameters fully configured
- ✅ Visualization configuration optimized for performance
- ✅ Simulation parameters for various test scenarios
- ⚠️ EKF configuration files may be deprecated (moved to launch files)