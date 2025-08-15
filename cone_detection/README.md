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

#### Cone Detection Output (Dual Publishing Strategy)
- `/sorted_cones_time` (custom_interface/ModifiedFloat32MultiArray): **Legacy format** - for backward compatibility
- `/cones/lidar` (custom_interface/TrackedConeArray): **New format** - recommended for new implementations
- `/cones/lidar/ukf` (custom_interface/TrackedConeArray): Tracked cones with velocities (UKF tracking)

#### Visualization and Debug Topics
- `/ouster/points/preprocessed` (sensor_msgs/PointCloud2): Clustered cone points for visualization
- `/ouster/points_fixed` (sensor_msgs/PointCloud2): Filtered point cloud
- `/vis/cone/lidar` (visualization_msgs/MarkerArray): Cone markers for RViz

## Message Formats

### Legacy Format: ModifiedFloat32MultiArray (`/sorted_cones_time`)
The legacy topic uses a custom extension of Float32MultiArray with class names:

```
Header header                    # Standard ROS header with timestamp and frame_id
MultiArrayLayout layout          # Data organization (rows=cones, cols=3 for x,y,z)
float32[] data                   # Flattened array of cone positions [x1,y1,z1,x2,y2,z2,...]
string[] class_names             # Cone classifications (always "Unknown" for LiDAR)
```

**Data Structure**:
- Each cone represented as 3 consecutive floats: `[x, y, z]`
- Sorted by x-coordinate (front to back)
- Enhanced NaN/Inf validation (NaN/Inf values replaced with 0.0)

### New Format: TrackedConeArray (`/cones/lidar`)
The recommended new format provides better structure and extensibility:

```
Header header                    # Standard ROS header with timestamp and frame_id
TrackedCone[] cones             # Array of detected/tracked cones

TrackedCone:
  geometry_msgs/Point position   # 3D position (x, y, z) with NaN/Inf validation
  geometry_msgs/Vector3 velocity # Velocity vector (zero for untracked detections)
  string color                   # Cone color ("unknown" for LiDAR-only detections)
  int32 track_id                # Sequential ID (1-based) for untracked, persistent for UKF
  float64 confidence            # Detection confidence (1.0 for validated cones)
```

**Advantages of TrackedConeArray**:
- **Type Safety**: Structured data vs. raw float arrays
- **Frame Support**: Proper header with timestamp and frame_id
- **Extensibility**: Ready for color classification and velocity estimation
- **Consistency**: Same format as `/cones/lidar/ukf` topic
- **Clarity**: Self-documenting field names
- **Robust Validation**: Enhanced NaN/Inf protection for all position fields

## Migration Guide: Legacy to New Topic Format

### Why Dual Publishing?
The cone_detection package implements a dual publishing strategy to support smooth migration from the legacy `ModifiedFloat32MultiArray` format to the new `TrackedConeArray` format:

- **Backward Compatibility**: Existing systems can continue using `/sorted_cones_time` without immediate changes
- **Future-Proofing**: New implementations should use `/cones/lidar` for better structure and features
- **Gradual Migration**: Teams can migrate at their own pace without breaking existing functionality

### Migration Steps

#### 1. For Consumer Nodes (Immediate - Low Risk)
Update your subscribers to use the new topic and message type:

**Old Code:**
```cpp
// Legacy subscription
auto subscription = create_subscription<custom_interface::msg::ModifiedFloat32MultiArray>(
    "/sorted_cones_time", 10, 
    std::bind(&YourNode::cone_callback_old, this, std::placeholders::_1));

void cone_callback_old(const custom_interface::msg::ModifiedFloat32MultiArray::SharedPtr msg) {
    // Parse flattened array manually
    for (size_t i = 0; i < msg->data.size(); i += 3) {
        float x = msg->data[i];
        float y = msg->data[i + 1]; 
        float z = msg->data[i + 2];
        // Process cone position...
    }
}
```

**New Code:**
```cpp
// New subscription - recommended
auto subscription = create_subscription<custom_interface::msg::TrackedConeArray>(
    "/cones/lidar", 10,
    std::bind(&YourNode::cone_callback_new, this, std::placeholders::_1));

void cone_callback_new(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
    // Direct access to structured data
    for (const auto& cone : msg->cones) {
        float x = cone.position.x;
        float y = cone.position.y;
        float z = cone.position.z;
        int track_id = cone.track_id;  // Additional tracking info available
        // Process cone position...
    }
}
```

#### 2. Benefits of Migration
- **Cleaner Code**: No manual array parsing required
- **Type Safety**: Compile-time checks for field access
- **Additional Data**: Access to track_id, confidence, and future fields
- **Better Debugging**: Self-documenting message structure
- **Frame Information**: Proper timestamp and frame_id handling

#### 3. Testing Migration
Both topics publish identical cone detection data:
```bash
# Compare outputs during migration
ros2 topic echo /sorted_cones_time
ros2 topic echo /cones/lidar
```

### Deprecation Timeline
- **Current**: Both topics published simultaneously
- **Phase 1** (Next Release): `/cones/lidar` becomes primary recommendation
- **Phase 2** (Future Release): Legacy topic marked deprecated with warnings
- **Phase 3** (Final Release): Legacy topic removed

**Recommendation**: Migrate to `/cones/lidar` as soon as possible to take advantage of improved data validation and future features.

## Data Validation and Reliability

### Enhanced NaN/Inf Protection
The cone_detection package implements comprehensive validation to ensure data reliability:

#### Input Validation
- **Point Cloud Filtering**: NaN points automatically filtered during preprocessing
- **Minimum Point Requirements**: Ensures sufficient data for reliable clustering
- **Range Validation**: Distance and intensity thresholds prevent invalid detections

#### Output Validation
Both publishing formats include robust NaN/Inf validation:

**Legacy Format (`/sorted_cones_time`)**:
```cpp
// NaN/Inf values replaced with 0.0 in the data array
if (std::isnan(val) || std::isinf(val)) {
    msg.data.push_back(0.0);  
} else {
    msg.data.push_back(val);
}
```

**New Format (`/cones/lidar`)**:
```cpp
// Individual field validation for x, y, z coordinates
if (std::isnan(cone.mean.x) || std::isinf(cone.mean.x)) {
    tracked_cone.position.x = 0.0;
} else {
    tracked_cone.position.x = cone.mean.x;
}
// Similar validation for y and z coordinates
```

#### Benefits of Enhanced Validation
- **System Stability**: Prevents downstream nodes from receiving invalid floating-point values
- **Consistent Behavior**: Predictable output even with noisy sensor data
- **Debugging Support**: Clear identification of problematic data points
- **Graceful Degradation**: System continues operating with best-effort data when sensors malfunction

### Error Handling Strategy
- **Exception Safety**: All major operations wrapped in try-catch blocks
- **Logging**: Comprehensive error logging for troubleshooting
- **Graceful Recovery**: System continues processing even if individual frames fail
- **Resource Management**: Proper cleanup of PCL objects and memory

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

5. **Data Publishing**:
   - Raw detections published via `publishTrackedConeArray(ConeDescriptor)`
   - Tracked objects published via `publishTrackedConeArray(TrackedObject)`
   - Both methods convert to standardized `TrackedConeArray` format

## Performance Tuning

- Reduce `voxel_leaf_size` for higher accuracy (at cost of speed)
- Adjust `ec_cluster_tolerance` based on cone density
- Enable `enable_stage2_validation` for higher accuracy
- Disable tracking if not needed for better performance

## Recent Updates

### 2025-07-31: Dual Publishing Strategy Implementation
- **BACKWARD COMPATIBLE**: Implemented dual publishing strategy for smooth migration
- **New Topic**: Added `/cones/lidar` with `TrackedConeArray` format (recommended)
- **Legacy Topic**: Maintained `/sorted_cones_time` with `ModifiedFloat32MultiArray` format
- **Enhanced Validation**: Implemented robust NaN/Inf validation for both publishing formats
- **Dual Publishing Methods**: Added overloaded `publishTrackedConeArray()` methods:
  - One for raw cone detections (`ConeDescriptor` → `TrackedConeArray`)
  - One for tracked objects (`TrackedObject` → `TrackedConeArray`)
- **Data Consistency**: Both topics publish identical cone detection results with enhanced validation
- **Migration Support**: Comprehensive migration guide for consumer nodes

### 2025-07-23: Tracking Integration  
- Integrated kalman_filters library for UKF tracking
- Removed dlib dependency (now using kalman_filters' Hungarian implementation)
- Improved build configuration
- Added debug logging

## License

TODO: Add license information

## Authors

TODO: Add author information