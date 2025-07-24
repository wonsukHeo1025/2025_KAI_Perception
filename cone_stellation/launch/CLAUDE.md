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

## Usage

```bash
# Run full test
ros2 launch cone_stellation test_slam_launch.py

# Run only SLAM (with real data)
ros2 launch cone_stellation cone_slam_launch.py

# Run only dummy publisher
ros2 launch cone_stellation dummy_publisher_launch.py
```

## Current Status

- Basic launch files created
- Topic remapping added for testing
- Missing launch configurations for:
  - Different sensor setups
  - Hardware integration
  - Multi-robot scenarios