# ConeSTELLATION 개발 계획

## 개요
ConeSTELLATION은 GLIM의 모듈식 아키텍처를 참고하여 개발하는 콘 기반 Graph SLAM 시스템입니다. LiDAR 포인트 클라우드 대신 콘 클러스터링 인스턴스를 입력으로 받아, YOLO로 라바콘 색상을 추가한 후 SLAM을 수행합니다.

## 핵심 아키텍처 (GLIM 참고)

### 1. 모듈 구조
```
cone_stellation/
├── include/cone_stellation/
│   ├── common/              # 공통 데이터 구조 및 유틸리티
│   ├── preprocessing/       # 콘 데이터 전처리
│   ├── odometry/           # 콘 기반 포즈 추정 (내부적으로 데이터 연관 포함)
│   ├── mapping/            # 로컬 및 글로벌 맵 구축 (루프 클로저 포함)
│   ├── util/               # 설정, 로깅, 직렬화
│   └── viewer/             # 시각화
├── src/
│   └── cone_stellation/     # 구현 파일
├── config/                 # JSON 설정 파일
├── modules/                # 동적 로딩 가능한 모듈
└── scripts/                # 시뮬레이션 및 테스트 스크립트 (cc_slam_sym에서 가져옴)
```

### 2. 핵심 베이스 클래스 (GLIM 패턴 적용)

#### 2.1 ConePreprocessorBase
```cpp
// GLIM의 PreprocessingModule 패턴 적용
class ConePreprocessorBase {
public:
  virtual ~ConePreprocessorBase() = default;
  virtual ConeCloud::Ptr process(const ConeDetections& raw_cones) = 0;
  virtual void set_config(const Config& config) = 0;
};
```

#### 2.2 ConeOdometryBase
```cpp
// GLIM의 OdometryEstimationBase 패턴 적용
class ConeOdometryBase {
public:
  virtual ~ConeOdometryBase() = default;
  virtual ConeFrame::Ptr estimate(const ConeCloud::Ptr& cones) = 0;
  virtual bool requires_odom() const = 0;
  virtual void insert_odom(const nav_msgs::msg::Odometry& odom) = 0;
  
  // 내부적으로 콘 데이터 연관 수행
  // GLIM처럼 factor 내부에서 correspondence 찾기
};
```

#### 2.3 LocalMapBuilderBase
```cpp
// GLIM의 SubMappingBase 패턴 적용
class LocalMapBuilderBase {
public:
  virtual ~LocalMapBuilderBase() = default;
  virtual void insert_frame(const ConeFrame::Ptr& frame) = 0;
  virtual std::vector<ConeMap::Ptr> get_maps() = 0;
};
```

#### 2.4 GlobalMapperBase
```cpp
// GLIM의 GlobalMappingBase 패턴 적용
class GlobalMapperBase {
public:
  virtual ~GlobalMapperBase() = default;
  virtual void insert_map(const ConeMap::Ptr& map) = 0;
  virtual void optimize() = 0;
  virtual void save(const std::string& path) = 0;
};
```

### 3. 핵심 데이터 구조

#### 3.1 Cone
```cpp
struct Cone {
  Eigen::Vector3d position;      // 3D 위치
  ConeColor color;               // YOLO로 검출된 색상
  double confidence;             // 검출 신뢰도
  int id;                       // 고유 ID
  double timestamp;             // 타임스탬프
};
```

#### 3.2 ConeFrame
```cpp
// GLIM의 EstimationFrame 패턴 적용
struct ConeFrame {
  using Ptr = std::shared_ptr<ConeFrame>;
  
  std::vector<Cone> cones;                    // 검출된 콘들
  Eigen::Isometry3d T_world_robot;           // 로봇 포즈
  double timestamp;                          // 타임스탬프
  std::unordered_map<int, int> associations; // 콘 ID -> 랜드마크 ID
  
  // GLIM처럼 custom data 지원
  std::unordered_map<std::string, std::shared_ptr<void>> custom_data;
};
```

#### 3.3 ConeMap
```cpp
// GLIM의 SubMap 패턴 적용
struct ConeMap {
  using Ptr = std::shared_ptr<ConeMap>;
  
  std::unordered_map<int, Cone> landmarks;   // 랜드마크 콘들
  std::vector<ConeFrame::Ptr> frames;        // 구성 프레임들
  Eigen::Isometry3d origin;                  // 맵 원점
};
```

### 4. 구현 전략

#### Phase 1: 기본 인프라 구축 (1-2주)
1. **디렉토리 구조 생성**
2. **기본 데이터 구조 구현** (Cone, ConeFrame, ConeMap)
3. **베이스 클래스 인터페이스 정의**
4. **CMakeLists.txt 구성** (GLIM 참고)
5. **JSON 설정 시스템 구현** (GLIM의 Config 클래스 차용)

#### Phase 1.5: 시뮬레이션 통합 (3일)
1. **cc_slam_sym 시뮬레이션 재사용**
   - dummy_publisher_node.py 복사 및 수정
   - sensor_simulator.py, motion_controller.py 재사용
   - cone_definitions.py 그대로 사용
   - 필요한 custom_interface 메시지 확인

#### Phase 2: 핵심 모듈 구현 (2-3주)
1. **ConePreprocessor 구현**
   - 노이즈 필터링
   - 중복 제거
   - 색상 검증
   
2. **ConeOdometry 구현**
   - 콘 기반 포즈 추정
   - 내부적으로 콘 매칭/연관 수행
   - Mahalanobis 거리 기반
   
3. **LocalMapBuilder 구현**
   - 키프레임 관리
   - 로컬 맵 생성
   - 콘 랜드마크 관리

#### Phase 3: GTSAM 통합 (2주)
1. **Factor 구현**
   - OdometryFactor (GLIM 참고)
   - ConeObservationFactor
   - PriorFactor
   
2. **GlobalMapper 구현**
   - GTSAM 그래프 구축
   - 최적화 실행
   - 결과 추출

#### Phase 4: 비동기 처리 및 최적화 (1주)
1. **AsyncConeAssociation** (GLIM의 AsyncOdometryEstimation 참고)
2. **AsyncLocalMapBuilder** (GLIM의 AsyncSubMapping 참고)
3. **AsyncGlobalMapper** (GLIM의 AsyncGlobalMapping 참고)
4. **스레드 안전성 보장**

#### Phase 5: ROS2 통합 (1주)
1. **ROS2 노드 구현**
2. **토픽 인터페이스 정의**
3. **서비스 및 파라미터 구현**
4. **Launch 파일 작성**

#### Phase 6: 시각화 및 디버깅 (1주)
1. **ConeViewer 구현** (GLIM의 viewer 모듈 참고)
2. **RViz 플러그인 개발**
3. **디버깅 도구 구현**

### 5. GLIM에서 차용할 핵심 요소

1. **모듈 동적 로딩 시스템**
   ```cpp
   // GLIM의 load_module 함수 패턴
   template<typename T>
   std::shared_ptr<T> load_module(const std::string& module_name);
   ```

2. **콜백 시스템**
   ```cpp
   // GLIM의 CallbackSlot 패턴
   template<typename Func>
   class CallbackSlot;
   ```

3. **설정 시스템**
   ```cpp
   // GLIM의 Config 클래스 활용
   class Config {
     json data;
     template<typename T>
     T param(const std::string& key, const T& default_value);
   };
   ```

4. **비동기 래퍼 패턴**
   ```cpp
   // GLIM의 Async 래퍼 패턴
   template<typename T>
   class AsyncWrapper {
     std::thread thread;
     ConcurrentQueue<typename T::InputType> input_queue;
   };
   ```

### 6. 주요 차이점 (GLIM vs ConeSTELLATION)

| 항목 | GLIM | ConeSTELLATION |
|------|------|----------------|
| 입력 데이터 | 포인트 클라우드 | 콘 검출 결과 (TrackedConeArray) |
| 맵 표현 | Voxel Map | 콘 랜드마크 맵 |
| 등록 방법 | ICP/GICP (내부 대응점) | 콘 매칭 (Factor 내부) |
| 센서 | LiDAR + IMU | LiDAR + Camera + Odom |
| 데이터 연관 | Factor 내부에서 암묵적 | ConeObservationFactor 내부 |

### 7. 예상 도전 과제 및 해결 방안

1. **데이터 연관 문제**
   - 문제: 콘 오검출 및 미검출
   - 해결: Robust한 연관 알고리즘, 색상 정보 활용

2. **스케일 문제**
   - 문제: 적은 수의 랜드마크
   - 해결: 효율적인 그래프 구조, 슬라이딩 윈도우

3. **실시간 처리**
   - 문제: 연산 복잡도
   - 해결: 비동기 처리, GPU 활용 고려

### 8. 테스트 전략

1. **단위 테스트**: 각 모듈별 독립적 테스트
2. **통합 테스트**: 전체 파이프라인 테스트
3. **시뮬레이션 테스트**: 가상 콘 데이터로 테스트
4. **실제 데이터 테스트**: rosbag 활용

### 9. 현재 상태 및 향후 계획 (2025-07-19 업데이트)

#### 완료된 작업
- ✅ 기본 factor graph SLAM 구현
- ✅ Inter-landmark factors (ConeDistanceFactor, ConeLineFactor)
- ✅ Factor graph 시각화
- ✅ 시뮬레이션 환경 통합
- ✅ ROS2 토픽 구조 문서화

#### 현재 이슈
- ✅ Odometry와 Mapping 분리 완료 (external IMU+GPS odometry)
- ⚠️ Sub-mapping 시스템 없음
- ⚠️ 비동기 처리 미구현

#### 우선순위 개발 계획 (Updated 2025-07-21)
1. **Phase 1: ~~ConeOdometryEstimation~~** (COMPLETED via external odometry)
   - We use external IMU+GPS EKF for 100Hz odometry
   - SLAM focuses on mapping and drift correction only
   - DriftCorrectionManager handles map->odom transform

2. **Phase 2: ConeSubMapping 모듈 추가** (1주)
   - Keyframe-based local mapping
   - Submap generation and management
   - Local optimization

3. **Phase 3: Async Wrapper 구현** (1주)
   - AsyncConeOdometry
   - AsyncConeSubMapping
   - Thread-safe queues

4. **Phase 4: Incremental Optimization** (1주)
   - ISAM2 proper usage
   - Factor marginalization
   - Memory management

5. **Phase 5: Advanced Optimization** (2주)
   - **~~Fixed-Lag Smoother~~** (POSTPONED - GLIM uses this for IMU odometry only)
     - Since we have external IMU+GPS odometry, not needed for landmark SLAM
     - Landmark SLAM should remain unbounded for accuracy
   - **Robust Kernels** (GLIM 참조)
     - Huber/Tukey kernels for outlier rejection
     - Adaptive kernel parameter tuning
     - Cone-specific robust factors
   - **Memory Management** (if needed)
     - Periodic landmark pruning for very long operations
     - Keep key landmarks for loop closure
     - Monitor memory usage and implement strategies as needed

6. **Phase 6: Loop Closure Detection and Correction** (3주)
   - **Cone Constellation Descriptor** 
     - Rotation-invariant cone patterns
     - Color-aware geometric descriptors
     - Efficient KD-tree indexing
   - **Loop Candidate Detection** (GLIM의 parallel evaluation 참조)
     - Travel distance-based search
     - Multi-threaded candidate evaluation
     - Probabilistic validation
   - **Loop Closure Constraints**
     - Robust pose graph optimization
     - Chi-squared validation
     - Gradual loop factor integration
   - **Map Correction Broadcasting**
     - Smooth trajectory deformation
     - Landmark position updates
     - Covariance propagation

### 10. GLIM 고급 기능 통합 계획 (2025-07-19 추가)

#### Phase 7: Enhanced Color Voting and Track ID Management (낮은 우선순위)
- **Advanced Color Voting** (현재 TentativeLandmark 확장)
  - Temporal consistency weighting
  - Distance-based confidence scaling
  - Multi-hypothesis color tracking
  - Bayesian color classification
- **Track ID Hysteresis Enhancement**
  - Hungarian algorithm for optimal assignment
  - Velocity-based prediction
  - Occlusion handling with ghost tracks
- **Pattern-based Color Validation**
  - Track boundary color constraints
  - Spatial color consistency checks

### 11. Advanced GLIM Features

#### Advanced Features from GLIM

1. **Multi-threaded Architecture** (Phase 3 확장)
   - TBB (Threading Building Blocks) 통합
   - ConcurrentVector/Queue for lock-free data passing
   - Parallel factor evaluation
   - Thread pool management

2. **Memory Management** (Phase 4 확장)
   - LRU cache for cone descriptors
   - DataStorePolicy for configurable buffers
   - Smart pointer throughout
   - Memory-bounded operations

3. **Advanced Registration** (새로운 모듈)
   - Cone-specific registration metrics
   - Multi-scale matching (individual → constellation)
   - GPU acceleration option
   - Adaptive correspondence thresholds

4. **Modular Callback System** (전체 아키텍처)
   - Frame insertion callbacks
   - Marginalization callbacks
   - Optimization callbacks
   - Performance monitoring hooks

5. **Configuration Management** (Phase 1 확장)
   - JSON-based hierarchical config
   - Runtime parameter updates
   - Profile-based configurations
   - Track-specific profiles

6. **Robust Optimization** (Phase 5 통합)
   - Multiple robust kernels
   - Graduated non-convexity
   - Adaptive parameter tuning
   - Fallback mechanisms

7. **Serialization and Recovery** (새로운 기능)
   - Full graph save/load
   - Session recovery
   - Cone map sharing
   - Multi-format export

### 12. IMU/GPS 통합 계획

#### Phase 8: IMU Integration (2주)
1. **IMU Preintegration Module**
   - GTSAM의 IMU preintegration 활용
   - Between-factor로 포즈 간 제약 추가
   - Bias estimation 포함
   
2. **Motion Model Enhancement**
   - IMU 기반 motion prediction
   - Cone association 개선
   - Outlier rejection 강화

3. **Factor Graph Extension**
   ```cpp
   // IMU preintegration factor
   graph.add(ImuFactor(x_i, v_i, x_j, v_j, b_i, preintegrated_imu));
   
   // Velocity and bias priors
   graph.add(PriorFactor<Vector3>(v_0, zero_velocity, velocity_noise));
   graph.add(PriorFactor<ImuBias>(b_0, zero_bias, bias_noise));
   ```

4. **Configuration**
   ```yaml
   imu:
     accelerometer_noise_density: 0.01
     gyroscope_noise_density: 0.001
     accelerometer_bias_random_walk: 0.0001
     gyroscope_bias_random_walk: 0.00001
     gravity_magnitude: 9.81
   ```

#### Phase 9: GPS Integration (1주)
1. **GPS Factor Implementation**
   - Global position constraints
   - UTM/Local coordinate transformation
   - Covariance scaling based on satellite count

2. **Loose Coupling Approach**
   ```cpp
   // GPS position factor (only when good fix)
   if (gps_msg.status.status >= NavSatStatus::STATUS_FIX) {
     graph.add(GPSFactor(x_i, gps_position, gps_noise));
   }
   ```

3. **Coordinate Frame Management**
   - ENU local frame initialization
   - GPS to map frame alignment
   - Datum management

#### Phase 10: Tight IMU/GPS/Vision Coupling (3주)
1. **Extended State Vector**
   ```cpp
   struct State {
     Pose3 T_world_imu;      // IMU pose in world
     Vector3 velocity;        // Linear velocity
     ImuBias imu_bias;       // Accelerometer and gyro biases
     double gps_time_offset;  // GPS-IMU time sync
   };
   ```

2. **Multi-rate Sensor Fusion**
   - IMU: 100-400 Hz
   - Cones: 10-20 Hz  
   - GPS: 1-10 Hz
   - Synchronized processing pipeline

3. **Initialization Strategy**
   - Static initialization for IMU biases
   - GPS-based global alignment
   - Cone-based scale recovery

4. **Failure Detection and Recovery**
   - Chi-squared test for outliers
   - Sensor health monitoring
   - Graceful degradation modes

#### Implementation Priorities
1. **먼저 IMU 통합** - Motion prediction 개선에 즉각적 도움
2. **다음 GPS 통합** - Global consistency를 위해
3. **마지막 Tight coupling** - 모든 센서가 안정화된 후

#### Testing Strategy
1. **Simulation Testing**
   - Add IMU/GPS to dummy_publisher
   - Inject sensor noise and biases
   - Test failure scenarios

2. **Real Data Testing**
   - Record rosbags with all sensors
   - Compare against ground truth
   - Evaluate improvement metrics

## 현재 구현 상태 (2025-07-19)

### 이미 구현된 기능 ✅
1. **Basic Color Voting** - TentativeLandmark에 구현 (단순 투표 방식)
2. **Tentative Landmark System** - 관측 버퍼링, 승급 기준, Track ID 관리
3. **Factor Graph SLAM** - ISAM2, 커스텀 팩터들 (ConeObservationFactor, Inter-landmark factors)
4. **Basic Visualization** - 랜드마크, 팩터 그래프, 키프레임, 경로

### 아직 구현 안 된 주요 기능 ❌
1. **~~Fixed-Lag Smoother~~** - POSTPONED (external odometry handles this)
2. **Loop Closure Detection** - 콘 constellation 기반
3. **Multi-threading** - 비동기 처리 아키텍처
4. **Robust Optimization** - Huber/Tukey 커널
5. **Odometry/Mapping Separation** - 실시간 성능 확보

### 성능 목표
| 지표 | 현재 | 목표 | 
|------|------|------|
| 최적화 시간 | ~100ms | <50ms |
| 메모리 사용량 | Unbounded | <1GB |
| Loop closure 성공률 | 0% | >90% |
| 색상 정확도 | ~85% | >95% |
| 처리 속도 | 10Hz | 20Hz+ |

## 결론

ConeSTELLATION은 GLIM의 검증된 아키텍처를 콘 기반 SLAM에 맞게 적용하여, 모듈성, 확장성, 성능을 모두 갖춘 시스템으로 개발할 예정입니다. 특히 **Inter-landmark factors**라는 핵심 혁신을 통해 희소한 콘 환경에서도 강건한 SLAM을 구현합니다.

**핵심 교훈**: GLIM의 성공 요인은 Odometry와 Mapping의 분리를 통한 multi-rate 처리입니다. ConeSTELLATION도 이를 적용하여 실시간 성능을 확보해야 합니다.