# config/ Directory

This directory contains configuration files for the cone_stellation SLAM system.

## Files

### slam_config.yaml
Main SLAM configuration including:
- Preprocessing parameters (max distance, confidence thresholds)
- Mapping parameters (optimization frequency, factor weights)
- Keyframe selection criteria
- Inter-landmark factor settings

### dummy_publisher_config.yaml
Test data publisher configuration:
- Simulation parameters
- Noise models
- Track scenarios
- Sensor simulation settings

### cone_slam.rviz
RViz configuration for visualizing:
- Cone landmarks
- Factor graph edges
- Vehicle trajectory
- TF tree

## Usage

Configuration files are loaded at runtime via ROS2 parameters:
```bash
ros2 run cone_stellation cone_slam_node --ros-args --params-file config/slam_config.yaml
```

## Current Status

- Basic SLAM parameters defined
- Need to add odometry-specific configs
- Missing sub-mapping parameters
- No async execution configs yet