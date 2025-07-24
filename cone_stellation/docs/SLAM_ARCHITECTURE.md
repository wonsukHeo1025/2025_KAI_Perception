# ConeSTELLATION SLAM Architecture

## Overview

ConeSTELLATION is a cone-based Graph SLAM system that extends GLIM's architecture with novel inter-landmark factors. The key innovation is adding geometric constraints between cones observed from the same pose, extracting maximum information from sparse cone observations.

## Current Status (2025-07-20)

### Working Features ✅
- **Data Association**: Excellent performance with minimal overlapping landmark markers
- **Noise Filtering**: Successfully filters noise cones, wrong colored cones, false positives/negatives
- **Factor Graph Construction**: Properly builds graph with pose-landmark observation edges (KF** format)
- **Graph Optimization**: Real-time backend optimization correctly adjusts poses and map
- **Visualization**: Clean factor graph visualization without orphan nodes

### Not Yet Implemented ❌
- **Inter-Landmark Factors**: Implemented but not yet enabled in configuration
- **Loop Closure**: Planned after IMU Preintegration + RTK GPS + inter-landmark factors
- **Drift Correction**: map->odom transform remains fixed at identity (no tf adjustment)
- **High-rate Odometry**: Currently running at SLAM rate, not sensor rate

## Core Innovation: Inter-Landmark Factors

Traditional SLAM only creates factors between:
- Pose → Landmark (observation factors)
- Pose → Pose (odometry factors)

ConeSTELLATION adds:
- **Landmark → Landmark factors** for co-observed cones
- **Geometric pattern factors** (lines, curves, grids)
- **Relative distance/angle constraints** between cone pairs

### Why This Matters

In Formula Student:
- Only 2-10 cones visible per frame (very sparse)
- Cones form structured patterns (track boundaries)
- Relative positions are highly constrained
- Co-visibility implies geometric relationships

## System Architecture

### 1. Data Flow Pipeline

```
Cone Detections → Preprocessing → Odometry → Mapping → Optimization
     ↓                ↓              ↓          ↓           ↓
TrackedCones    AssociatedCones  LocalMap  GlobalMap  Optimized Map
```

### 2. Key Components

#### 2.1 Preprocessing Module
- Cone tracking and ID management
- Outlier rejection
- Uncertainty estimation
- Pattern detection (lines, curves)

#### 2.2 Odometry Module
- Fast local cone association
- Motion estimation from cone observations
- IMU integration (if available)
- Keyframe selection

#### 2.3 Mapping Module
- Global cone map management
- Inter-landmark factor creation
- Loop closure detection
- Map optimization triggers

#### 2.4 Factor Graph Structure

**Symbols:**
- `X(i)`: Vehicle pose at keyframe i
- `L(j)`: Cone landmark j position
- `V(i)`: Vehicle velocity (optional)
- `B(i)`: IMU bias (if IMU available)

**Factor Types:**

1. **Odometry Factors**
   - Between consecutive poses X(i) → X(i+1)
   - From wheel encoders or IMU

2. **Cone Observation Factors**
   - Pose to landmark X(i) → L(j)
   - Range-bearing or Cartesian

3. **Inter-Landmark Factors** (Novel)
   - Between co-observed cones L(j) → L(k)
   - Distance constraints
   - Angle constraints
   - Line/curve alignment

4. **Pattern Factors** (Novel)
   - Multiple cones forming geometric patterns
   - Straight line constraints
   - Circular arc constraints
   - Parallel line constraints

### 3. Inter-Landmark Factor Design

#### 3.1 Pairwise Distance Factor
For cones j,k observed from pose i:
```
residual = ||L(j) - L(k)|| - d_observed
```

#### 3.2 Relative Angle Factor
For cone triple (j,k,m):
```
residual = angle(L(j), L(k), L(m)) - θ_observed
```

#### 3.3 Line Alignment Factor
For cones on a detected line:
```
residual = distance_to_line(L(j), line_params)
```

#### 3.4 Track Boundary Factor
Enforces parallel track edges:
```
residual = parallelism(left_cones, right_cones)
```

## Implementation Plan

### Phase 1: Basic Framework ✅ COMPLETED
- [x] Core data structures (Cone, ConeObservation)
- [x] Basic factor graph with pose-landmark factors
- [x] Simple data association with track ID support
- [x] Visualization infrastructure (factor graph, landmarks, paths)

### Phase 2: Inter-Landmark Factors ⏳ IN PROGRESS
- [x] Pairwise distance factors (implemented)
- [x] Line alignment factors (implemented)
- [ ] Enable in configuration and test
- [ ] Co-visibility tracking integration
- [ ] Factor weight tuning

### Phase 3: Drift Correction & High-Rate Odometry 🎯 PRIORITY
- [ ] Implement map->odom transform calculation
- [ ] Separate odometry module for sensor-rate tracking
- [ ] Drift visualization and monitoring
- [ ] Smooth correction interpolation

### Phase 4: Advanced Features 🔮 FUTURE
- [ ] IMU Preintegration for robust motion prediction
- [ ] RTK GPS integration for global consistency
- [ ] Loop closure with constellation descriptors
- [ ] Pattern-based track structure enforcement

## Technical Details

### Coordinate Frames
- `map`: Global fixed frame
- `odom`: Drifting odometry frame
- `base_link`: Vehicle center
- `imu_link`: IMU sensor (optional)

### Cone Representation
```cpp
struct Cone {
    int id;                    // Unique landmark ID
    Eigen::Vector2d position;  // 2D position in map frame
    ConeColor color;          // YELLOW, BLUE, RED, UNKNOWN
    double confidence;        // Detection confidence
    std::set<int> co_observed; // IDs of cones seen together
};
```

### Factor Graph Update Strategy
1. Add new keyframe when:
   - Traveled > 1m or rotated > 10°
   - New cones detected
   - Loop closure candidate

2. Create inter-landmark factors when:
   - Multiple cones in single observation
   - Confidence above threshold
   - Consistent co-observation

3. Optimize when:
   - Every N keyframes (batch)
   - Loop closure detected
   - Large residuals detected

## Configuration Parameters

See `config/slam_config.yaml` for all parameters including:
- Factor weights
- Association thresholds  
- Pattern detection settings
- Optimization triggers

## Evaluation Metrics

- **Trajectory Error**: ATE, RPE
- **Map Consistency**: Cone position accuracy
- **Pattern Preservation**: Track structure quality
- **Computational Performance**: Real-time factor

## Future Roadmap

### Sensor Fusion Architecture
The system is designed to eventually incorporate IMU and RTK GPS for robust performance:

1. **IMU Integration**
   - Preintegration between keyframes for smooth motion estimation
   - Bias estimation and gravity alignment
   - Higher frequency pose prediction (100+ Hz)
   
2. **RTK GPS Integration**
   - Global position constraints to prevent long-term drift
   - Absolute coordinate reference for mapping
   - Fallback for challenging scenarios (few visible cones)

3. **Inter-Landmark Factors**
   - Essential for sparse cone environments
   - Provides additional constraints when few cones visible
   - Works synergistically with RTK GPS for loop closure

### Why This Order Matters
- Current system proves SLAM logic works independently of simulator
- Adding IMU/GPS will enhance robustness without changing core architecture
- Inter-landmark factors become more valuable with absolute positioning

## Odometry Architecture Decision

### Primary Control Odometry: IMU+GPS Fusion (100+ Hz)
Based on Formula Student requirements (speeds up to 100 km/h, ~200-300m tracks):

**Why IMU+GPS for Vehicle Control:**
- **High Frequency**: 100+ Hz necessary for stable control at high speeds
- **Low Latency**: Direct sensor data without optimization delays
- **Continuous Operation**: Works even with poor cone visibility
- **Smooth Control**: No jumps from optimization updates

**Why NOT Direct SLAM Output:**
- **Too Slow**: 10Hz insufficient for 100 km/h control
- **High Latency**: Optimization introduces control delays
- **Potential Jumps**: Loop closures can cause instability
- **Landmark Dependent**: Fails without visible cones

### SLAM's Role: Drift Correction
- Provides globally consistent pose estimates
- Corrects IMU drift periodically
- Updates map for path planning
- Runs at sustainable rate (10-30 Hz)

### Multi-Rate Architecture Benefits
1. **Decoupling**: Control loop independent of SLAM computation
2. **Robustness**: Control continues if SLAM delays/fails
3. **Accuracy**: Combines IMU responsiveness with SLAM accuracy
4. **Scalability**: Can optimize SLAM without affecting control

### Implementation Strategy
```
IMU+GPS (100+ Hz) → Vehicle Control
    ↓
    ├→ SLAM (10-30 Hz) → Drift Correction
    └→ Kalman Filter → Fused Estimate
```

## Odometry/Mapping Separation (Production Architecture)

Following GLIM's proven multi-rate architecture, ConeSTELLATION will be separated into two nodes for production:

### Current Architecture (Monolithic)
```
cone_slam_node (10 Hz)
├── Cone Detection Input
├── Odometry Input
├── SLAM Processing
│   ├── Data Association
│   ├── Factor Graph Construction
│   └── ISAM2 Optimization
└── Visualization Output
```

### Target Architecture (Separated)
```
cone_odometry_node (20-50 Hz)          cone_mapping_node (1-10 Hz)
├── Cone Detection Input               ├── Keyframe Input (from odometry)
├── Odometry Input                     ├── Cone Observations (buffered)
├── Fast Pose Tracking                 ├── Global Map Building
│   ├── Frame-to-Frame Matching      │   ├── Landmark Management
│   ├── Local Map Tracking           │   ├── Inter-landmark Factors
│   └── Pose Prediction              │   └── Loop Closure Detection
├── Keyframe Selection                 ├── ISAM2 Optimization
└── High-Rate Pose Output             └── Map Correction Output
```

### Benefits of Separation
1. **Performance**: Odometry runs at sensor rate without optimization pressure
2. **Robustness**: Independent failure modes, easier debugging
3. **Scalability**: Can run on separate machines, mapping can process offline

### Implementation Priority
1. Extract ConeOdometryEstimation module (2 weeks)
2. Refactor ConeMapping for batch processing (1 week)
3. Integration and testing (1 week)

See `glim_features_integration.md` for detailed implementation plan.