# Path: /home/user1/ROS2_Workspace/Symforce_ws/src/cone_stellation/scripts/CLAUDE.md

# Scripts Directory

This directory contains simulation scripts adapted from cc_slam_sym for testing ConeSTELLATION without real sensor data.

## Components

### Core Simulation
- **dummy_publisher_node.py** - Main simulation node with cone detections, IMU, GPS
- **sensor_simulator.py** - Basic sensor noise models (Allan variance for IMU, RTK modes for GPS)
- **sensor_simulator_enhanced.py** - Enhanced simulators with temperature drift, axis misalignment, WGS84/UTM conversion
- **motion_controller.py** - Vehicle motion with smooth spline trajectories
- **cone_definitions.py** - Formula Student cone types and colors

### Test Scripts
- **test_dummy_publisher.py** - Basic ROS2 diagnostic checks
- **test_imu_gps_fusion.py** - IMU-GPS fusion testing with motion profiles
- **test_loop_closure.py** - Loop closure scenario testing
- **test_slam_only.py** - Automated SLAM testing with monitoring

## Enhanced Sensor Features (2025-07-24)

### IMU Simulator
- Allan variance noise modeling (gyro/accel)
- Temperature-dependent bias drift
- Scale factor errors (ppm)
- Axis misalignment simulation
- G-sensitivity effects

### GPS Simulator  
- Full WGS84 ↔ UTM coordinate transformation
- RTK status transitions (Fix/Float/Single)
- Realistic covariance (2cm for RTK Fix)
- DOP effects and multipath simulation
- Seoul origin: 37.5665°N, 126.9780°E (UTM 52S)

## Usage

```bash
# Basic simulation
ros2 run cone_stellation dummy_publisher_node.py

# With enhanced sensors (enable in config)
# Set simulation.use_gps: true in dummy_publisher_config.yaml

# Test specific motion profiles
ros2 run cone_stellation test_imu_gps_fusion.py --ros-args -p motion_profile:=circular
```

## Status
✅ Enhanced IMU-GPS simulators integrated
✅ Backward compatible with basic simulators
✅ Test scripts cleaned up (removed redundant shell script)
✅ Ready for robot_localization integration
