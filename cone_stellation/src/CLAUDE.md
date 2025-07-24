# src/ Directory

This directory contains implementation files for the cone_stellation SLAM system.

## Structure

- **cone_stellation/**: Main source directory
  - **ros/**: ROS2 specific implementations
    - `cone_slam_node.cpp`: Main ROS2 node that orchestrates SLAM
  - **common/**: Common utilities implementation (if needed)
  - **factors/**: Factor implementations (currently header-only)
    - `inter_landmark_factors.cpp`: Inter-landmark factor implementations
  - **preprocessing/**: Preprocessing implementations
    - `cone_preprocessor.cpp`: Cone preprocessing logic
  - **mapping/**: Mapping module implementations (currently header-only)
  - **util/**: Utility implementations (if needed)

## Implementation Notes

- Most logic is in headers (header-only design)
- Source files mainly for:
  - ROS2 node implementations
  - Complex algorithms that benefit from separate compilation
  - Reducing compilation time for frequently used headers

## Current Status (2025-07-20)

### Recent Updates
- **cone_slam_node.cpp**: 
  - ✅ Full SLAM pipeline working with inter-landmark factors
  - ✅ Integrated DriftCorrectionManager for map->odom transform
  - ✅ Uses cone-based odometry (AsyncConeOdometry)
  - ✅ Real-time factor graph optimization
  - ✅ Comprehensive visualization via SLAMVisualizer
  - ✅ Track ID properly utilized in data association
- **inter_landmark_factors.cpp**: 
  - ✅ ConeDistanceFactor properly creates constraints between landmarks
  - ✅ Co-observation tracking bug fixed (was returning binary values)

### Current System Flow
1. Cone detections received → Preprocessing
2. Odometry estimation from cone matching
3. Data association with track ID support
4. Mapping with inter-landmark factors
5. ISAM2 optimization
6. Drift correction updates map->odom transform

### Working Features
- ✅ Real-time SLAM with cone landmarks
- ✅ Inter-landmark distance constraints
- ✅ Track ID based data association
- ✅ Drift correction between odometry and SLAM
- ✅ Comprehensive visualization (landmarks, factors, path)

### Next Development Phase
- Pattern detection for line/curve factors
- IMU/GPS integration for high-rate control
- Loop closure detection
- Performance optimization for large-scale maps