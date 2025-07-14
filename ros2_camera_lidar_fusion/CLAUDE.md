# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS2 Humble package for camera-LiDAR sensor fusion, focusing on intrinsic and extrinsic calibration. The package is written in Python using `ament_python` build system.

## Build and Run Commands

### Using Docker (Recommended)
```bash
# Build Docker image
cd docker && sh build.sh

# Run container
cd docker && sh run.sh
```

### Manual Build
```bash
# Build the package
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select ros2_camera_lidar_fusion

# Source workspace
source install/setup.bash
```

### Testing
```bash
# Run unit tests (uses pytest)
colcon test --packages-select ros2_camera_lidar_fusion
colcon test-result --verbose
```

### Linting
```bash
# Python linting (package uses flake8 and pep257)
ament_flake8 ros2_camera_lidar_fusion/
ament_pep257 ros2_camera_lidar_fusion/
```

## Architecture

### Calibration Workflow
1. **Intrinsic Calibration** (`get_intrinsic_camera_calibration.py`): Calibrates camera using chessboard pattern
2. **Data Collection** (`save_sensor_data.py`): Records synchronized camera/LiDAR data
3. **Point Correspondence** (`extract_points.py`): Manual selection of matching points
4. **Extrinsic Calibration** (`get_extrinsic_camera_calibration.py`): Computes transformation matrix
5. **Projection** (`lidar_camera_projection.py`): Visualizes LiDAR points on camera images

### Key Components
- **Single Camera**: Standard calibration pipeline
- **Multi Camera**: Extended support via `project_dual_cameras_points.py` and multi-config files
- **Object Projection**: `project_boxes_cones_points.py` for specific object types
- **Configuration**: YAML-based configuration in `config/` directory
- **Visualization**: RViz2 integration via launch files

### Configuration Structure
- Single camera: `general_configuration.yaml`, `camera_intrinsic_calibration.yaml`, `camera_extrinsic_calibration.yaml`
- Multi camera: `multi_general_configuration.yaml`, `multi_camera_intrinsic_calibration.yaml`, `multi_camera_extrinsic_calibration.yaml`
- Chessboard: 8x9 pattern, 0.07m squares

### ROS2 Topics and Frames
- LiDAR topic: Configurable in YAML (e.g., `/point_cones_rec`)
- Camera topics: Configurable per camera
- Frame IDs: Must match TF tree (e.g., `os_sensor`, `camera_l`)

## Development Notes

- Pure Python package - no CMakeLists.txt
- Entry points defined in setup.py
- Uses cv_bridge for ROS2-OpenCV integration
- Message synchronization via message_filters
- Korean comments present in some files
- Test data included in `data/` directory (.pcd, .png files)