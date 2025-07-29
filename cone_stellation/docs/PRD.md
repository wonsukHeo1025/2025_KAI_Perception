# ConeSTELLATION Product Requirements Document (PRD)

## Product Vision

ConeSTELLATION is a specialized cone-based Graph SLAM system designed for Formula Student autonomous racing vehicles. It addresses the unique challenge of extremely sparse landmark observations (only 2-10 cones visible per frame) through novel inter-landmark geometric constraints and a multi-rate hybrid architecture that separates high-rate vehicle control from mapping responsibilities.

## Problem Statement

Formula Student autonomous vehicles face unique SLAM challenges:
- **Sparse Observations**: Only 2-10 cones visible at any time
- **High-Speed Racing**: Vehicles travel up to 100 km/h requiring stable control
- **Dynamic Environment**: Cones may be knocked over or misplaced
- **Limited Compute**: Embedded systems with power constraints
- **Real-time Requirements**: Control loop must run at 100Hz minimum

Traditional SLAM systems designed for dense point clouds fail in this sparse, high-speed environment.

## Target Users

- Formula Student Driverless teams
- Autonomous racing researchers
- Robotics developers working with sparse landmarks
- Academic institutions teaching autonomous navigation

## Core Requirements

### Functional Requirements

1. **Multi-Sensor Fusion**
   - Fuse IMU data at 100-400Hz
   - Integrate RTK GPS at 10Hz with Fix/Float/Single awareness
   - Process cone detections from LiDAR clustering
   - Incorporate YOLO-based color classification

2. **SLAM Capabilities**
   - Maintain consistent global map of cone positions
   - Provide drift correction (map→odom transform)
   - Support loop closure in sparse environments
   - Handle dynamic landmarks (moved/knocked cones)

3. **Real-time Performance**
   - External odometry at 100Hz for vehicle control
   - SLAM updates at 10-30Hz for mapping
   - Latency < 50ms for drift correction
   - CPU-only operation (no GPU required)

4. **Robustness**
   - Handle 50% cone occlusions
   - Recover from temporary sensor failures
   - Maintain operation with degraded GPS (Single mode)
   - Reject false positive cone detections

### Non-Functional Requirements

1. **Accuracy**
   - Position drift < 0.5m over 1km track
   - Cone position accuracy < 0.3m
   - Orientation accuracy < 2 degrees

2. **Scalability**
   - Support tracks up to 500m circumference
   - Handle up to 200 cones in global map
   - Maintain performance with 10+ laps

3. **Modularity**
   - Plugin architecture for different algorithms
   - Configurable via YAML/JSON files
   - ROS2 compatible interfaces
   - Extensible factor types

4. **Maintainability**
   - Comprehensive documentation
   - Unit tests for critical components
   - Continuous integration support
   - Clear separation of concerns

## Key Features

### 1. Inter-landmark Factors (Innovation)
- Geometric constraints between co-observed cones
- Handles sparse observations effectively
- Reduces drift in feature-poor environments

### 2. Multi-rate Hybrid Architecture
- Separates control (100Hz) from mapping (10-30Hz)
- External EKF for sensor fusion
- SLAM for global consistency only

### 3. Enhanced Data Association
- Color-based matching with confidence scores
- Track ID support for temporal consistency
- Tentative landmark system to filter noise

### 4. Adaptive Loop Closure
- Constellation-based recognition (3+ cones)
- Path curvature profile matching
- Feature detection (turns, chicanes)

### 5. Comprehensive Visualization
- Real-time RViz display
- Factor graph visualization
- Performance metrics overlay
- Debug modes for development

## Success Metrics

1. **Performance KPIs**
   - Achieve < 0.5m drift over standard FSG tracks
   - Maintain 100Hz odometry output
   - Process cone detections within 50ms

2. **Reliability KPIs**
   - 99% uptime during 20-minute endurance runs
   - Recover from GPS outages within 5 seconds
   - Zero critical failures in 100 test runs

3. **Adoption KPIs**
   - Used by 5+ Formula Student teams
   - 50+ GitHub stars within first year
   - Active community with 10+ contributors

## Technical Architecture

Built on proven technologies:
- **GTSAM**: Factor graph optimization
- **Eigen**: Linear algebra
- **ROS2**: Middleware and communication
- **robot_localization**: EKF sensor fusion
- **OpenCV**: Visualization (optional)

Inspired by GLIM's successful architecture while adapted for discrete landmarks rather than continuous point clouds.

## Development Roadmap

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for detailed phases and timeline.

## Constraints and Assumptions

### Constraints
- Must run on embedded ARM processors
- Power consumption < 50W total system
- No reliance on external infrastructure
- Compatible with standard FS sensor suite

### Assumptions
- Cones are static during each run
- Track layout follows FS regulations
- Sensors are properly calibrated
- Vehicle dynamics model available

## Risks and Mitigation

1. **Sparse Data Risk**: Too few cones visible
   - Mitigation: Inter-landmark factors, IMU integration

2. **Computational Risk**: Cannot meet real-time requirements
   - Mitigation: Multi-rate architecture, optimization

3. **Environmental Risk**: Weather affects sensors
   - Mitigation: Robust outlier rejection, multi-sensor fusion

4. **Integration Risk**: Incompatible with team's stack
   - Mitigation: Standard ROS2 interfaces, documentation

## Appendices

- Sensor specifications: [input_topic_form.md](input_topic_form.md)
- Architecture details: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- Debug history: [debug_log.md](debug_log.md)