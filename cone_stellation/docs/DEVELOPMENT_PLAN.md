# ConeSTELLATION Development Plan

## 1. Introduction

ConeSTELLATION (Cone-based STructural ELement Layout for Autonomous NavigaTION) is a cone-based Graph SLAM system designed for Formula Student autonomous racing. It processes cone detection instances from LiDAR clustering with YOLO-based color classification, inspired by GLIM's modular architecture.

**Key Resources:**
- Sensor data formats: [input_topic_form.md](input_topic_form.md)
- ROS2 topic structure: [topic_structure.md](topic_structure.md)
- Debugging archive: [debug_log.md](debug_log.md)

## 2. Current Status (2025-07-24)

### ✅ Working Components
- **Core SLAM**: GTSAM-based factor graph with ISAM2 optimization
- **Data Association**: Robust with color constraints and track ID support
- **Inter-landmark Factors**: Distance constraints between co-observed cones (working)
- **Loop Closure**: Enhanced for sparse environments with constellation-based recognition
- **Drift Correction**: Dynamic map→odom transform calculation
- **Visualization**: Comprehensive RViz display with performance optimization
- **Sensor Simulators**: Enhanced IMU/GPS with realistic noise models

### 🚧 In Progress
- **IMU-GPS Integration**: GTSAM IMU preintegration and GPS factors
- **EKF Configuration**: External 100Hz odometry for vehicle control

### ❌ Not Yet Implemented
- GTSAM IMU preintegration factors
- RTK GPS position factors with adaptive weighting
- Robot localization EKF configuration
- Multi-threaded architecture
- Production-ready loop closure

## 3. System Architecture

### 3.1 Overall Design
```
┌─────────────────────────────────────────────┐
│         External Sensors (100Hz)             │
│    IMU + RTK GPS → robot_localization       │
└────────────────┬────────────────────────────┘
                 │ Fused Odometry (100Hz)
                 ↓
┌─────────────────────────────────────────────┐
│         ConeSTELLATION SLAM (10-30Hz)       │
│                                             │
│  Cone Detection → Data Association →        │
│  Factor Graph → Optimization → Map         │
│                                             │
│  Output: map→odom drift correction         │
└─────────────────────────────────────────────┘
```

### 3.2 Architectural Decisions

**Multi-rate Hybrid Architecture:**
- **Control Layer** (100Hz): IMU+GPS fusion via robot_localization for stable vehicle control
- **SLAM Layer** (10-30Hz): Mapping and drift correction
- **Rationale**: Control stability matters more than global accuracy during racing

**Why External Odometry (like GLIM)?**
- Fixed-lag smoother inappropriate for landmark SLAM (requires continuous features)
- External EKF handles high-rate sensor fusion efficiently
- SLAM focuses on global consistency and drift correction
- Proven approach in GLIM for similar reasons

### 3.3 Module Structure
```
cone_stellation/
├── include/cone_stellation/
│   ├── common/              # Core data structures
│   ├── preprocessing/       # Cone data preprocessing
│   ├── mapping/            # SLAM mapping with inter-landmark
│   ├── factors/            # Custom GTSAM factors
│   └── util/               # ROS2 utilities
├── src/                    # Implementation files
├── config/                 # YAML configuration
├── scripts/                # Python simulators
└── launch/                 # ROS2 launch files
```

## 4. Implementation Details

### 4.1 Inter-landmark Factors

**Innovation**: Pairwise distance constraints between co-observed cones to handle sparse observations (2-10 cones/frame).

**Key Features:**
- Co-visibility tracking with configurable thresholds
- Clustering algorithm to avoid over-constraining
- Adaptive noise model based on distance
- Visualization as red lines in RViz

**Parameters:**
```yaml
inter_landmark:
  enabled: true
  min_covisibility_count: 3
  min_distance: 1.5
  max_distance: 15.0
  noise_model:
    base_stddev: 0.05
    distance_factor: 0.01
```

### 4.2 Loop Closure Implementation

**Enhanced for Sparse Environments** (2025-07-22):
- **Constellation-based**: Geometric patterns with 3+ cones
- **Path-based**: Curvature profiles and traveled distance
- **Feature Detection**: Turns, straights, chicanes
- **Combined Scoring**: 30% cones, 30% path, 40% features

**Visualization**: Purple lines for loop closure factors

### 4.3 IMU-GPS Integration (Current Focus)

**Enhanced Simulators**:
- **IMU**: Allan variance noise, temperature drift, g-sensitivity
- **GPS**: RTK Fix/Float/Single modes, multipath effects

**Integration Plan**:
1. GTSAM IMU preintegration between keyframes
2. GPS position factors with RTK-aware covariance
3. Coordinate frame setup (map → odom → base_link → imu_link)
4. Robot_localization EKF configuration

## 5. Feature Integration Roadmap

### 5.1 GLIM Features to Integrate

**High Priority:**
- ✅ ISAM2 incremental optimization
- ✅ ROS2 parameter system
- ✅ Drift correction (map→odom)
- 🚧 Multi-threading architecture
- ⏳ Memory management
- ⏳ Serialization/recovery

**Medium Priority:**
- ⏳ Robust kernels (Huber, Cauchy)
- ⏳ Interactive viewer
- ⏳ Global registration
- ⏳ GPU acceleration

### 5.2 ConeSTELLATION-Specific Features

**Completed:**
- ✅ Inter-landmark factors
- ✅ Tentative landmark system
- ✅ Color-based data association
- ✅ Track ID utilization

**Planned:**
- ⏳ Sparse rigid body constraints
- ⏳ Pattern-based loop detection
- ⏳ Cone constellation matching

## 6. Development Phases

### Phase 1: Core Infrastructure ✅
- Basic data structures
- GTSAM integration
- ROS2 node setup
- Visualization

### Phase 2: Basic SLAM ✅
- Data association
- Factor graph construction
- ISAM2 optimization
- Drift correction

### Phase 3: Advanced Features (Current)
- ✅ Inter-landmark factors
- ✅ Enhanced loop closure
- 🚧 IMU-GPS integration
- ⏳ Multi-threading

### Phase 4: Production Ready
- ⏳ Robust optimization
- ⏳ Error recovery
- ⏳ Performance optimization
- ⏳ Comprehensive testing

## 7. Testing Strategy

### Simulation Testing
- Enhanced sensor simulators with realistic noise
- Multiple motion profiles (straight, circular, figure-8)
- Ground truth comparison

### Real Data Testing
- Rosbag playback from actual races
- Performance metrics (accuracy, timing)
- Failure mode analysis

### Integration Testing
- End-to-end system validation
- Multi-sensor synchronization
- Real-time performance

## 8. Performance Targets

- **Odometry Rate**: 100Hz (external EKF)
- **SLAM Rate**: 10-30Hz
- **Accuracy**: < 0.5m drift over 1km track
- **Robustness**: Handle 50% cone occlusions
- **Latency**: < 50ms for drift correction

## 9. Dependencies

**Required:**
- Eigen3
- GTSAM 4.0+
- gtsam_points
- spdlog
- Boost
- robot_localization

**Optional:**
- OpenCV (visualization)
- CUDA (future GPU acceleration)

## 10. References

- GLIM architecture: `/home/user1/ROS2_Workspace/GLIM_ws/src/glim/`
- Original development inspired by: GLIM paper and codebase
- Formula Student rules and track specifications