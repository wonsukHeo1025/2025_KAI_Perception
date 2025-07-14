# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS2 Python package for sensor fusion that combines YOLO object detection with LiDAR point cloud data using the Hungarian algorithm for data association. The system is designed for cone detection and tracking in autonomous vehicle applications.

## Common Development Commands

### Build Commands
```bash
# Build from workspace root
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select hungarian_association

# Build with symlinks (for development)
colcon build --packages-select hungarian_association --symlink-install

# Clean build
rm -rf build/ install/ log/
colcon build --packages-select hungarian_association
```

### Running the Nodes
```bash
# Source the workspace first
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

# Single camera fusion
ros2 run hungarian_association hungarian_association_node

# Multi-camera fusion
ros2 run hungarian_association yolo_lidar_multicam_fusion_node

# Kalman filter tracking
ros2 run hungarian_association kalman_filtering_node

# RViz visualization
ros2 run hungarian_association visualize_fused_cones_rviz_marker_node
```

### Testing and Linting
```bash
# Run tests
colcon test --packages-select hungarian_association
colcon test-result --verbose

# Linting
ament_flake8 hungarian_association/
ament_pep257 hungarian_association/
```

### Launch with Parameters
```bash
# Override configuration file
ros2 run hungarian_association hungarian_association_node --ros-args -p config_file:=path/to/custom_config.yaml
```

## High-Level Architecture

### Core Components

1. **Sensor Fusion Pipeline**
   - `yolo_lidar_fusion.py`: Single camera YOLO-LiDAR fusion using Hungarian algorithm
   - `yolo_lidar_multicam_fusion.py`: Multi-camera variant supporting multiple camera streams
   - Both nodes project 3D LiDAR points to 2D camera plane and match with YOLO detections

2. **Tracking System**
   - `kalman_filtering.py`: Implements Unscented Kalman Filter (UKF) for cone tracking
   - Integrates IMU data for motion compensation
   - Maintains track history with color confidence tracking
   - State vector: [x, y, z, vx, vy, vz]

3. **Visualization**
   - `visualize_fused_cones_rviz_marker.py`: Creates color-coded RViz markers
   - Dynamically updates markers based on tracked cone states

### Data Flow
1. LiDAR detects cones → produces 3D positions
2. YOLO detects objects in camera images → produces 2D bounding boxes
3. Fusion node projects LiDAR points to camera plane using calibration
4. Hungarian algorithm associates projected points with YOLO detections
5. Kalman filter tracks cones over time with IMU compensation
6. Visualization node displays results in RViz

### Configuration System
- YAML-based configuration files in `config/`
- Camera calibration files (intrinsic/extrinsic) in dedicated directories
- Runtime parameter loading via `config_utils.py`

### Message Synchronization
- Uses ROS2 `message_filters.ApproximateTimeSynchronizer`
- Configurable time tolerance for sensor fusion
- QoS settings optimized for real-time operation

### Key Design Patterns
- Modular node-based architecture
- Publisher-subscriber pattern for data flow
- Configuration-driven behavior
- Comprehensive error handling and logging
- Support for both single and multi-camera setups

## Important Notes

- The package uses custom message types (e.g., `ModifiedFloat32MultiArray`, `DetectionArray`)
- Kalman filter parameters can be tuned following guidelines in `파라미터 수정 전략.md`
- Camera calibration is critical for accurate fusion - ensure calibration files are properly formatted
- The system assumes cone detection scenario (racing/autonomous driving context)