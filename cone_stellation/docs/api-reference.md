# ConeSTELLATION API Reference

## Overview

This document provides comprehensive API documentation for the key classes and components in the ConeSTELLATION cone-based SLAM system.

## Core Data Structures

### Cone Class (`cone.hpp`)

The fundamental data structure representing a detected cone in the environment.

```cpp
struct Cone {
    Eigen::Vector3d position;      // 3D position in sensor frame
    std::string color;             // "BLUE", "YELLOW", "ORANGE_BIG", "ORANGE_SMALL", "UNKNOWN"
    double confidence;             // Color classification confidence [0.0, 1.0]
    int track_id;                  // Unique ID for tracking across frames (-1 if untracked)
    double timestamp;              // Detection timestamp
};
```

**Key Methods:**
- `bool isValidColor()`: Checks if cone has a recognized color
- `double distanceTo(const Cone& other)`: Computes Euclidean distance to another cone
- `bool colorMatches(const Cone& other)`: Checks color compatibility for association

### EstimationFrame Class (`estimation_frame.hpp`)

Represents a single frame of sensor data with vehicle pose and cone observations.

```cpp
class EstimationFrame {
public:
    // Frame metadata
    int id;                        // Unique frame identifier
    double timestamp;              // Frame timestamp
    
    // Vehicle state
    Eigen::Isometry3d T_world_sensor;  // Vehicle pose in world frame
    Eigen::Vector3d linear_velocity;    // Linear velocity
    Eigen::Vector3d angular_velocity;   // Angular velocity
    
    // Observations
    std::vector<Cone> cones;       // Detected cones in this frame
    
    // Methods
    Eigen::Isometry3d odom() const;
    void transformCones(const Eigen::Isometry3d& transform);
};
```

### TentativeLandmark Class (`tentative_landmark.hpp`)

Manages cone observations before they are promoted to permanent landmarks.

```cpp
class TentativeLandmark {
private:
    std::vector<Cone> observations_;     // Buffered observations
    std::map<std::string, int> color_votes_;  // Color voting system
    Eigen::Vector3d mean_position_;      // Running mean position
    Eigen::Matrix3d covariance_;         // Position covariance
    
public:
    // Core functionality
    bool addObservation(const Cone& cone, const Eigen::Isometry3d& sensor_pose);
    bool isReadyForPromotion() const;
    Cone getPromotedLandmark() const;
    
    // Configuration
    static constexpr int MIN_OBSERVATIONS = 3;
    static constexpr double MAX_ASSOCIATION_DISTANCE = 1.5;
    static constexpr double OUTLIER_THRESHOLD = 2.0;  // std deviations
};
```

## Mapping Components

### ConeMapping Class (`cone_mapping.hpp`)

The main SLAM backend that manages the factor graph and optimization.

```cpp
class ConeMapping {
private:
    // GTSAM components
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    
    // State management
    std::map<int, gtsam::Symbol> landmark_symbols_;
    std::map<int, TentativeLandmark> tentative_landmarks_;
    
public:
    // Main SLAM pipeline
    void addOdometryFactor(const EstimationFrame::Ptr& prev_frame, 
                          const EstimationFrame::Ptr& curr_frame);
    void addConeObservations(const EstimationFrame::Ptr& frame);
    void optimize();
    
    // Inter-landmark factors
    void addInterLandmarkFactors(const std::vector<int>& covisible_landmarks);
    
    // Loop closure
    void detectAndAddLoopClosures();
    
    // Map access
    std::vector<Cone> getOptimizedLandmarks() const;
    Eigen::Isometry3d getOptimizedPose(int frame_id) const;
};
```

### DataAssociation Class (`data_association.hpp`)

Handles matching between observed cones and map landmarks.

```cpp
class DataAssociation {
public:
    struct AssociationResult {
        int landmark_id;
        double mahalanobis_distance;
        double color_penalty;
    };
    
    // Main association method
    std::vector<AssociationResult> associate(
        const std::vector<Cone>& observations,
        const std::vector<Cone>& landmarks,
        const Eigen::Isometry3d& sensor_pose,
        const AssociationParams& params);
    
    // Parameters
    struct AssociationParams {
        double max_distance = 1.5;
        double color_mismatch_penalty = 10.0;
        double min_color_confidence = 0.8;
        bool use_track_ids = true;
    };
};
```

### LoopClosureDetector Class (`loop_closure_detector.hpp`)

Enhanced loop closure detection for sparse cone environments.

```cpp
class LoopClosureDetector {
private:
    // Feature types for sparse environments
    struct ConstellationFeature {
        std::vector<int> cone_indices;
        Eigen::MatrixXd relative_positions;
        double geometric_hash;
    };
    
    struct PathFeature {
        double curvature;
        double traveled_distance;
        std::string segment_type;  // "straight", "turn", "chicane"
    };
    
public:
    // Loop closure detection
    std::vector<LoopCandidate> detectLoops(
        const EstimationFrame::Ptr& current_frame,
        const std::vector<EstimationFrame::Ptr>& keyframes);
    
    // Geometric verification
    bool verifyLoopClosure(const LoopCandidate& candidate);
    
    // Configuration
    void setMinConstellationSize(int size);  // Default: 3 cones
    void setScoreWeights(double cone_weight, double path_weight, double feature_weight);
};
```

## GTSAM Custom Factors

### ConeObservationFactor (`cone_observation_factor.hpp`)

Factor connecting vehicle pose to cone landmark.

```cpp
class ConeObservationFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
private:
    Cone observation_;
    
public:
    // Error computation
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose,
        const gtsam::Point3& landmark,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override;
    
    // Factory method
    static boost::shared_ptr<ConeObservationFactor> create(
        gtsam::Key pose_key,
        gtsam::Key landmark_key,
        const Cone& observation,
        const gtsam::SharedNoiseModel& noise_model);
};
```

### InterLandmarkFactor (`inter_landmark_factors.hpp`)

Distance constraint between co-observed landmarks.

```cpp
class InterLandmarkFactor : public gtsam::NoiseModelFactor2<gtsam::Point3, gtsam::Point3> {
private:
    double measured_distance_;
    int covisibility_count_;
    
public:
    // Error computation (distance constraint)
    gtsam::Vector evaluateError(
        const gtsam::Point3& landmark1,
        const gtsam::Point3& landmark2,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override;
    
    // Adaptive noise model
    static gtsam::SharedNoiseModel createNoiseModel(
        double distance,
        const InterLandmarkParams& params);
};
```

## Utility Components

### DriftCorrectionManager (`drift_correction_manager.hpp`)

Manages the map→odom transform for drift correction.

```cpp
class DriftCorrectionManager {
private:
    // Transform history for interpolation
    std::deque<TimestampedTransform> transform_history_;
    size_t max_history_size_ = 1000;
    
public:
    // Update drift correction
    void updateDriftCorrection(
        const Eigen::Isometry3d& T_map_base_optimized,
        const Eigen::Isometry3d& T_odom_base_current,
        double timestamp);
    
    // Query interpolated transform
    Eigen::Isometry3d getMapToOdomTransform(double timestamp) const;
    
    // TF broadcasting
    void publishTransform(tf2_ros::TransformBroadcaster& tf_broadcaster);
};
```

## Configuration

### SLAM Configuration (`slam_config.yaml`)

Key parameters for system configuration:

```yaml
# Data Association
data_association:
  max_distance: 1.5
  min_color_confidence: 0.8
  color_mismatch_penalty: 10.0
  use_track_ids: true

# Tentative Landmarks
tentative_landmarks:
  min_observations: 3
  promotion_threshold: 5
  outlier_std_dev: 2.0

# Inter-landmark Factors
inter_landmark:
  enabled: true
  min_covisibility_count: 3
  min_distance: 1.5
  max_distance: 15.0
  max_factors_per_frame: 15
  noise_model:
    base_stddev: 0.05
    distance_factor: 0.01

# Loop Closure
loop_closure:
  enabled: true
  min_constellation_size: 3
  score_weights:
    cones: 0.3
    path: 0.3
    features: 0.4

# Optimization
optimization:
  isam2_relinearize_threshold: 0.1
  isam2_relinearize_skip: 10
  keyframe_delta_trans: 0.5
  keyframe_delta_angle: 0.1
```

## Usage Examples

### Basic SLAM Pipeline

```cpp
// Initialize SLAM system
auto cone_mapping = std::make_shared<ConeMapping>(config);
auto data_association = std::make_shared<DataAssociation>(config);

// Process new frame
void processFrame(const EstimationFrame::Ptr& frame) {
    // 1. Data association
    auto associations = data_association->associate(
        frame->cones, 
        cone_mapping->getOptimizedLandmarks(),
        frame->T_world_sensor);
    
    // 2. Add observations to mapping
    cone_mapping->addConeObservations(frame, associations);
    
    // 3. Add odometry factor
    if (prev_frame) {
        cone_mapping->addOdometryFactor(prev_frame, frame);
    }
    
    // 4. Add inter-landmark factors
    cone_mapping->addInterLandmarkFactors(frame);
    
    // 5. Optimize
    cone_mapping->optimize();
    
    // 6. Update drift correction
    drift_manager->updateDriftCorrection(
        cone_mapping->getOptimizedPose(frame->id),
        frame->T_world_sensor,
        frame->timestamp);
}
```

### Loop Closure Integration

```cpp
// Configure loop detector for sparse environments
loop_detector->setMinConstellationSize(3);
loop_detector->setScoreWeights(0.3, 0.3, 0.4);

// Detect loops periodically
if (frame->id % 10 == 0) {
    auto loop_candidates = loop_detector->detectLoops(frame, keyframes);
    
    for (const auto& candidate : loop_candidates) {
        if (loop_detector->verifyLoopClosure(candidate)) {
            cone_mapping->addLoopClosureFactor(candidate);
        }
    }
}
```

## Thread Safety

The system is designed with future multi-threading in mind:

- `ConeMapping`: Not thread-safe, requires external synchronization
- `DataAssociation`: Thread-safe for read operations
- `DriftCorrectionManager`: Thread-safe with internal mutex protection
- `TentativeLandmark`: Not thread-safe, managed by ConeMapping

## Performance Considerations

- **Optimization Rate**: 10-30Hz depending on graph size
- **Max Landmarks**: Tested up to 1000 landmarks without performance degradation
- **Inter-landmark Factors**: Limited to 15 per frame to avoid over-constraining
- **Loop Closure**: Runs every 10 frames to balance accuracy and performance