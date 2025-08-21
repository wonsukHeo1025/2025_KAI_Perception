# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the CALICO package.

## Overview

CALICO (Cone Attribute Linking by Image and Cluster Output) is a high-performance C++ package for sensor fusion in autonomous racing applications, combining YOLO object detection with LiDAR point cloud data.

## Development Status (2025-07-01)

CALICO is currently in active development as a high-performance C++ implementation for sensor fusion optimization.

### ✅ Completed Components
- Basic package structure (CMakeLists.txt, package.xml)
- Configuration loader utility (100% Python YAML compatible)
- Message converter utility
- Multi-camera fusion node with Hungarian matching
- UKF tracking system (basic implementation)
- IMU compensator with EMA/Butterworth filters
- RViz visualization node
- Launch files for individual and full system

### 🚧 Components Needing Improvement
- **Hungarian Algorithm**: ✅ Now using kalman_filters library implementation
  - Removed dlib dependency completely
- **UKF Implementation**: ✅ Using kalman_filters library
  - Complete UKF with numerical stability improvements
- **Butterworth Filter**: Simplified 2nd order implementation
  - Need: Full filter design or DSP library

### 🐛 Current Issues
- Matching returns 0 in some cases - debugging needed
- Projection visualization needed for troubleshooting

## Build Commands

```bash
# Build from workspace root
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select calico

# Clean build
rm -rf build/calico install/calico log/
colcon build --packages-select calico

# Build with debug symbols
colcon build --packages-select calico --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

## Running the Nodes

```bash
# Source the workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run multi-camera fusion node
ros2 run calico multi_camera_fusion_node --ros-args \
  -p config_file:=/home/user1/ROS2_Workspace/ros2_ws/src/calico/config/multi_hungarian_config.yaml

# Using launch file
ros2 launch calico multi_camera_fusion.launch.py

# With custom config
ros2 launch calico multi_camera_fusion.launch.py \
  config_file:=/path/to/custom_config.yaml
```

## Configuration Compatibility

CALICO uses its own YAML configuration files:
- `multi_hungarian_config.yaml` - Main configuration
- `multi_camera_intrinsic_calibration.yaml` - Camera intrinsic parameters
- `multi_camera_extrinsic_calibration.yaml` - Camera extrinsic parameters

This ensures seamless transition from Python to C++ implementation without changing configuration.

## Development Guidelines

### Non-Invasive Development Approach
- Maintain compatibility with existing ROS2 interfaces
- Use the same topic names and message types as Python version
- Configuration files remain unchanged
- Allow easy fallback to Python implementation

### Performance Optimization Focus
- Primary goal: Computation speed improvement (target: 5x faster than Python)
- Use Eigen for matrix operations
- OpenCV for camera projections
- Consider parallel processing for multi-camera
- Implement efficient Hungarian algorithm (dlib or munkres-cpp)

### Code Organization
```
calico/
├── include/calico/     # Header files
│   ├── fusion/        # Sensor fusion algorithms
│   ├── utils/         # Utilities and IMU compensator
│   └── visualization/ # RViz markers
├── src/               # Implementation files
│   ├── fusion/
│   ├── utils/
│   ├── visualization/
│   └── nodes/         # ROS2 node executables
└── launch/            # Launch files
```

### Testing Strategy
- Unit tests for core algorithms
- Integration tests with bag files
- Performance benchmarks against Python implementation
- Validate same output as Python version (< 1% difference)

## Common Issues and Solutions

### Issue: Missing dependencies
```bash
# Install required packages
sudo apt update
sudo apt install libeigen3-dev libyaml-cpp-dev libopencv-dev
```

### Issue: Custom messages not found
```bash
# Build message packages first
colcon build --packages-select custom_interface yolo_msgs
```

### Issue: Configuration file not found
Ensure absolute paths are used or files are in the expected location.

## Next Steps for Implementation

1. **Hungarian Algorithm Integration**
   - Evaluate dlib vs munkres-cpp
   - Implement cost matrix computation
   - Add distance threshold filtering

2. **Fusion Logic Implementation**
   - Port LiDAR to camera projection
   - Implement multi-camera fusion logic
   - Handle camera conflicts

3. **UKF Tracking** ✅
   - Now using kalman_filters external library
   - IMU compensation integrated
   - Color voting mechanism implemented

4. **Performance Optimization**
   - Profile and identify bottlenecks
   - Implement parallel processing
   - Optimize memory allocation

## Projection Debug Node ✅

A projection visualization node has been implemented to help debug sensor fusion issues:

### Features
- Subscribes to `/sorted_cones_time` (LiDAR cones) and camera image topics
- Projects LiDAR points onto camera image plane using calibration parameters
- Draws green circles at projected points with cone index and color labels
- Publishes overlay image to `/debug/projection_overlay`
- Shows statistics: total cones, projected points, and valid projections

### Usage
```bash
# Run standalone debug node
ros2 launch calico projection_debug.launch.py camera_id:=camera_1

# Enable debug visualization in full launch
ros2 launch calico calico_full.launch.py enable_debug_viz:=true debug_camera_id:=camera_1

# View debug image
ros2 run rqt_image_view rqt_image_view /debug/projection_overlay
```

## Next Steps

### 2. Library Integration Priority
1. **Hungarian Algorithm**: 
   - Option A: dlib (already in Ubuntu repos)
   - Option B: munkres-cpp (header-only)
   - Option C: Implement full algorithm with Eigen

2. **UKF Library**:
   - Option A: Port filterpy logic directly
   - Option B: Use existing C++ Kalman library
   - Option C: Complete current implementation

3. **DSP Library** for filters:
   - Option A: Use existing DSP library
   - Option B: Port scipy.signal algorithms

### 3. Key Differences from Python
- Python uses lowercase color names: "blue cone", "yellow cone", "red cone", "unknown"
- Python doesn't filter projected points by image bounds
- Python UKF parameters: R=0.1, max_age=4, distance_threshold=0.7
- Python uses filterpy's UKF which handles sigma points differently

## Important Notes

- The package name "CALICO" stands for "Cone Attribute Linking by Image and Cluster Output"
- This is a performance-focused port, not a feature enhancement
- Maintain backward compatibility with Python package
- Focus on multi-camera support (single camera is not needed)
- Color output format: Python uses "Blue Cone" (capitalized) for tracked output