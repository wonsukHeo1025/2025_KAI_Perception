# launch/ Directory

This directory contains ROS2 launch files for the cone_stellation SLAM system.

## Launch Files

### slam_only_launch.py
Main SLAM launch file:
- Launches cone_slam_node with SLAM configuration
- Sets up visualization and TF publishing
- For use with real sensor data or rosbags

### dummy_publisher_launch.py  
Launch file for testing with simulated data:
- Launches dummy cone publisher with configurable tracks
- Generates cone detections, IMU, GPS data
- Can run standalone for testing visualization

### test_slam_launch.py
Combined launch for integrated testing:
- Launches both dummy publisher and SLAM nodes
- Includes topic remapping for simulation
- Complete testing environment

## Usage

```bash
# Run full SLAM test with simulation
ros2 launch cone_stellation test_slam_launch.py

# Run SLAM only (for real data/rosbags)
ros2 launch cone_stellation slam_only_launch.py

# Run simulation only (for testing)
ros2 launch cone_stellation dummy_publisher_launch.py
```

## Current Status

- ✅ Core launch files functional
- ✅ Topic remapping configured for simulation
- ✅ Visualization and TF tree properly configured
- ⚠️ Note: IMU/GPS EKF launch files mentioned in old docs may be deprecated
- 📝 TODO: Launch configurations for different sensor setups and hardware integration