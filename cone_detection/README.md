# Cone Detection

A ROS2 package for cone detection using LiDAR point cloud data, designed for autonomous racing applications.

## Overview

The `cone_detection` package processes LiDAR point cloud data to detect traffic cones in real-time. It uses advanced filtering, clustering, and tracking algorithms to provide robust cone detection even in challenging conditions.

## Features

- **Point Cloud Filtering**: ROI-based filtering, ground plane removal, and outlier rejection
- **Cone Clustering**: Euclidean clustering with adaptive parameters
- **Multi-Stage Validation**: Two-stage validation process for high accuracy
- **Object Tracking**: Integrated UKF-based tracking using the kalman_filters library
- **Real-time Performance**: Optimized for low latency processing
- **Visualization Support**: RViz markers for debugging and monitoring

## Dependencies

- ROS2 Humble
- PCL (Point Cloud Library)
- Eigen3
- kalman_filters library (for UKF tracking)
- custom_interface (for message types)

## Installation

1. Install dependencies:
```bash
sudo apt update
sudo apt install ros-humble-pcl-* libeigen3-dev
```

2. Build the kalman_filters library (see kalman_filters README)

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select cone_detection
```

## Configuration

The main configuration file is located at `config/cone_detection_config.yaml`. Key parameters include:

### Filtering Parameters
- `x/y/z_threshold_*`: Spatial filtering bounds
- `min_distance`, `max_distance`: Distance-based filtering
- `intensity_threshold`: Minimum intensity for valid points
- `roi_angle_min/max`: Region of interest angles

### Clustering Parameters
- `ec_cluster_tolerance`: Euclidean clustering distance threshold
- `ec_min/max_cluster_size`: Cluster size limits
- `min/max_cone_height`: Valid cone height range

### Tracking Parameters (when enabled)
- `enable_tracking`: Enable/disable UKF tracking
- `max_association_distance`: Maximum distance for track association
- `min_hits_before_confirmation`: Hits required for track confirmation
- `max_age_before_deletion`: Frames before unmatched track deletion

## Usage

### Launch the Detection Node
```bash
ros2 launch cone_detection cone_detection_launch.py
```

### Launch with Custom Config
```bash
ros2 launch cone_detection cone_detection_launch.py \
    config_file:=/path/to/custom_config.yaml
```

## Topics

### Subscribed Topics
- `/ouster/points` (sensor_msgs/PointCloud2): Input LiDAR point cloud

### Published Topics
- `/sorted_cones_time` (custom_interface/ModifiedFloat32MultiArray): Detected cone positions
- `/sorted_cones_time_ukf` (custom_interface/TrackedConeArray): Tracked cones with velocities
- `/point_cones` (sensor_msgs/PointCloud2): Clustered cone points for visualization
- `/ouster/points_fixed` (sensor_msgs/PointCloud2): Filtered point cloud
- `/vis/cone/lidar` (visualization_msgs/MarkerArray): Cone markers for RViz

## Visualization

A separate visualization node is provided for RViz display:
```bash
ros2 run cone_detection cone_detection_visualization_node
```

This node converts the detected/tracked cones into visualization markers.

## Algorithm Overview

1. **Preprocessing**: 
   - Transform LiDAR points to sensor frame
   - Apply ROI and distance filtering
   - Voxelize for performance

2. **Ground Removal**:
   - RANSAC-based plane segmentation
   - Remove points close to ground plane

3. **Cone Detection**:
   - Euclidean clustering of remaining points
   - Height-based validation
   - Optional second-stage validation

4. **Tracking** (if enabled):
   - UKF-based motion prediction
   - Hungarian algorithm for data association
   - Track management (creation/deletion)

## Performance Tuning

- Reduce `voxel_leaf_size` for higher accuracy (at cost of speed)
- Adjust `ec_cluster_tolerance` based on cone density
- Enable `enable_stage2_validation` for higher accuracy
- Disable tracking if not needed for better performance

## Recent Updates (2025-07-23)

- Integrated kalman_filters library for UKF tracking
- Removed dlib dependency (now using kalman_filters' Hungarian implementation)
- Improved build configuration
- Added debug logging

## License

TODO: Add license information

## Authors

TODO: Add author information