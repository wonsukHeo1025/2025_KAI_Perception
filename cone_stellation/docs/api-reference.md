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

## Preprocessing Module (cone_preprocessor.hpp) — KOR 상세 분석

### 역할과 책임
- 관측 전처리 파이프라인으로, 센서 프레임(또는 `base_link`) 기준 원시 콘 관측을 입력받아 아래를 수행합니다.
  - 거리/신뢰도 기반 아웃라이어 제거
  - 단순 기하 패턴(직선) 탐지
  - 관측 ID 할당/유지(현재 구현은 미할당 ID에 대해 증가 ID 부여 수준)

### 공개 API
- `std::shared_ptr<ConeObservationSet> process(const std::vector<ConeObservation>& raw_observations, const Eigen::Isometry3d& sensor_pose, double timestamp)`
  - 입력 관측을 필터링하고, 선택적으로 패턴을 검출한 뒤, 간단 추적을 적용한 `ConeObservationSet`을 반환합니다.
- `std::vector<ConePattern> get_recent_patterns() const`
  - 최근 검출된 패턴을 반환합니다(스레드 안전 복사).

### 구성 파라미터
```cpp
struct Config {
  // Outlier rejection
  double max_cone_distance = 20.0;    // 최대 유효 거리 (m)
  double min_cone_confidence = 0.5;   // 최소 신뢰도

  // Pattern detection
  bool enable_pattern_detection = true;
  double line_fitting_threshold = 0.2; // 직선 적합 허용 거리 (m)
  int min_cones_for_line = 3;          // 직선 판단 최소 콘 수

  // Tracking (현재 단순 ID 할당만 사용)
  double association_threshold = 1.0;
  int max_tracking_frames = 10;
};
```

### 입출력 계약
- 입력: `std::vector<ConeObservation>` (좌표는 `base_link` 기준을 기대), `sensor_pose`, `timestamp`.
- 출력: `ConeObservationSet`
  - `cones`: 거리/신뢰도 필터를 통과한 관측들(필요 시 ID 부여)
  - `detected_patterns`: `enable_pattern_detection`이 참이면 직선 패턴 후보
  - `sensor_pose`, `timestamp` 유지

### 내부 동작 요약
- 유효성 검사: `position.norm() <= max_cone_distance` 그리고 `confidence >= min_cone_confidence`.
- 패턴 탐지: 삼중 조합 O(N^3) 공선성 판정 후 임계거리 내 점들을 추가해 `ConePattern::LINE`을 생성.
- 추적/ID: `id < 0`인 관측에 증가 ID 부여. `association_threshold`/`max_tracking_frames`/`tracked_positions_`는 현재 로직에 적극 반영되지 않음.

### 다른 모듈과의 관계
- 매핑(`ConeMapping`)의 연관 로직은 색상/거리/ID 안정성에 민감합니다. 프리프로세싱 단계에서 ID 안정화(프레임 간 일관성)가 높을수록 매핑의 오연관·플리커가 줄어듭니다.
- 패턴 탐지 결과는 현재 매핑에서 팩터 생성이 비활성화되어 정보 제공/디버깅 용도로만 사용됩니다.

### 예상 문제점과 개선 제안
- ID 안정화 부족: 최근접 NN + 속도 제한/EMA(지수이동평균) 또는 슬라이딩 median으로 `tracked_positions_`를 실제로 활용해 스무딩/재식별이 필요합니다.
- O(N^3) 패턴 탐지는 N이 커지면 비효율적: RANSAC/Hough 변환 기반으로 개선 권장.
- 관측 공분산/신뢰도 활용 부족: 현재 `ros_utils`에서 거리 기반 공분산을 생성하지만, 전처리 단계에서 적응형 가중(예: 거리↑/신뢰도↓ → 가중↓)을 명시적으로 표준화하면 다운스트림 노이즈 모델 일관성이 좋아집니다.
- 좌표 일관성: 입력이 반드시 `base_link` 기준이라는 전제와 TF 적용 경로(`cone_slam_node.cpp`)가 일치하는지 보장해야 합니다.
