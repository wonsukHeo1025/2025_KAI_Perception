# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Standard build from workspace root
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select filc

# Release build with optimizations
colcon build --packages-select filc --cmake-args -DCMAKE_BUILD_TYPE=Release

# Debug build
colcon build --packages-select filc --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Clean build
rm -rf build/filc install/filc log/filc
colcon build --packages-select filc
```

## Test Commands

```bash
# Run linting tests
colcon test --packages-select filc --event-handlers console_direct+

# Run functional test (requires node to be running)
python3 /home/user1/ROS2_Workspace/ros2_ws/src/filc/scripts/test_functionality.py

# View test results
colcon test-result --verbose
```

## Run Commands

```bash
# Source workspace
source /home/user1/ROS2_Workspace/ros2_ws/install/setup.bash

# Run main fusion node
ros2 run filc main_fusion_node

# Run visualization node
ros2 run filc visualization_node

# Launch with parameters (launch file needs to be created)
ros2 launch filc interpolation_demo.launch.py
```

## Architecture Overview

**filc (Fusion of Interpolated LiDAR and Camera)** is a ROS2 package that enhances Ouster LiDAR point clouds through:
- High-resolution interpolation of range and intensity images
- Multi-camera color assignment to LiDAR points
- Real-time processing with performance optimizations

### Core Components

1. **main_fusion_node.cpp**: Main processing pipeline
   - Subscribes to LiDAR data (points, range/intensity images)
   - Subscribes to multiple camera feeds
   - Performs interpolation and color assignment
   - Publishes colored point clouds and interpolated images

2. **visualization_node.cpp**: System monitoring
   - Displays processing statistics
   - Monitors performance metrics
   - Provides real-time status updates

3. **Utility Libraries**:
   - `InterpolationUtils`: Implements multiple interpolation algorithms (LINEAR, CUBIC, BICUBIC, LANCZOS, ADAPTIVE)
   - `ProjectionUtils`: Handles camera-LiDAR coordinate transformations and color mapping
   - `ConfigLoader`: Manages YAML configuration files

### Key Features

- **Interpolation Methods**: Multiple algorithms with configurable scale factors
- **Color Assignment Strategies**: NEAREST_CAMERA, ALL_CAMERAS, WEIGHTED_AVERAGE
- **Hybrid Mode**: Interpolates front 270° while preserving rear sector
- **Performance Optimizations**: OpenMP multithreading, SIMD instructions (AVX2/SSE4.1), spatial culling
- **Robustness**: Exception handling for missing topics, graceful degradation, runtime parameter updates

### Configuration Files

- `config/interpolation_config.yaml`: Interpolation parameters and Ouster OS1-32 settings
- `config/multi_general_configuration.yaml`: Topic mappings and file paths
- `config/multi_camera_*_calibration.yaml`: Camera intrinsic/extrinsic parameters

### Runtime Parameters

```bash
# Change interpolation method
ros2 param set /filc_main_fusion_node interpolation_type BICUBIC

# Change color assignment strategy  
ros2 param set /filc_main_fusion_node color_assignment_strategy ALL_CAMERAS

# Adjust max projection distance
ros2 param set /filc_main_fusion_node max_projection_distance 20.0
```

### Dependencies

- ROS2 packages: rclcpp, sensor_msgs, cv_bridge, pcl_ros, tf2, visualization_msgs
- External: OpenCV, PCL, Eigen3, yaml-cpp, OpenMP

### Performance Considerations

- Build uses aggressive optimizations (-O3, -march=native)
- Supports parallel processing via OpenMP
- Memory-efficient point cloud handling
- Spatial optimization through frustum culling