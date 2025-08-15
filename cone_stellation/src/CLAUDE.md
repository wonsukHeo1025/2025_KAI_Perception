# cone_stellation/src C++ 소스 상세 분석

본 문서는 `cone_stellation/src/` 트리 내 C++ 구현 파일들의 역할, ROS2 인터페이스, 주요 알고리즘, 데이터 흐름, 파라미터, 주의사항을 한국어로 상세히 정리한 자료입니다. 코드의 실제 구현은 많은 부분이 헤더에 존재하며, 본 디렉터리의 `.cpp` 들은 ROS2 노드 구동, 복잡 로직의 별도 컴파일, 시각화/검증 로직 등에 집중되어 있습니다.

- 분석 대상 파일
  - `cone_stellation/src/cone_stellation/ros/cone_slam_node.cpp`
  - `cone_stellation/src/loop_closure_detector.cpp`
  - `cone_stellation/src/cone_stellation/factors/inter_landmark_factors.cpp`
  - `cone_stellation/src/cone_stellation/preprocessing/cone_preprocessor.cpp`

---

## cone_stellation/ros/cone_slam_node.cpp

- **역할**: ROS2 노드 엔트리(메인 SLAM orchestrator). 콘 검출 입력을 받아 전처리 → 키프레임 판정 → 매핑/최적화 → 시각화/퍼블리시를 수행. 드리프트 보정(임시 비활성화) 및 TF 브로드캐스트 관리.
- **클래스**: `cone_stellation::ConeSLAMNode : rclcpp::Node`
- **구성요소 초기화**
  - 전처리기: `ConePreprocessor` (파라미터 기반 설정)
  - 오도메트리: `ConeOdometry2D` → `AsyncConeOdometry` 래핑 후 `start()`
  - 매핑: 기본 `ConeMapping` 사용. 디버깅용 `SimpleConeMapping`은 코드상 강제로 비활성화됨(`use_simple_mapping = false`).
  - 시각화: `viewer::SLAMVisualizer`
  - 드리프트 보정: `DriftCorrectionManager` (현재 업데이트 경로 주석 처리)

- **주요 토픽/TF**
  - 구독(Subscriber)
    - `/cones/fused/ukf` (`custom_interface::msg::TrackedConeArray`, QoS=BestEffort/Volatile, depth=10) → `cone_callback`
    - `/odometry/filtered` (`nav_msgs::msg::Odometry`, QoS=BestEffort/Volatile, depth=100) → `odom_callback`
  - 퍼블리시(Publisher)
    - `/slam/pose` (`geometry_msgs::msg::PoseStamped`, BestEffort/Volatile)
    - `/slam/odometry` (`nav_msgs::msg::Odometry`, BestEffort/Volatile)
  - TF 브로드캐스트
    - `map -> odom`을 항등 변환으로 100ms 주기 타이머에서 지속 퍼블리시(임시 고정)
    - `map -> base_link_slam`은 최신 최적화 결과를 기반으로 시각화 콜백에서 퍼블리시
    - `odom -> base_link`는 EKF가 담당한다는 가정 하에 SLAM 측 퍼블리시는 비활성화

- **주요 파라미터(디폴트)**
  - 전처리(`preprocessing.*`)
    - `max_cone_distance=20.0`, `min_cone_confidence=0.5`, `enable_pattern_detection=true`
    - `line_fitting_threshold=0.2`, `min_cones_for_line=3`, `association_threshold=1.0`, `max_tracking_frames=10`
  - 매핑(`mapping.*`/`association.*`)
    - `enable_inter_landmark_factors=true`, `inter_landmark_distance_noise=0.1`
    - `optimize_every_n_frames=10`, `min_covisibility_count=2`, `max_landmark_distance=10.0`
    - `association.max_association_distance=2.0`
    - `optimize_on_loop_closure=false` (루프클로저 미통합 상태)
  - 텐터티브 랜드마크(`tentative_landmark.*`)
    - `min_observations=3`, `min_time_span=0.5`, `max_position_variance=0.5`, `min_color_confidence=0.6`, `max_observations=20`
  - 키프레임 기준
    - `keyframe.translation_threshold=1.0`, `keyframe.rotation_threshold=0.2` [rad]
  - 오도메트리
    - `odometry.max_correspondence_distance=3.0`, `odometry.use_color_constraint=true`, `odometry.min_correspondences=3`

- **콜백 및 로직**
  - `odom_callback(msg)`
    - 마지막 오도메트리 저장 및 `Eigen::Isometry3d T_odom_base` 구성
    - 드리프트 매니저 입력 경로는 현재 주석 처리
  - `cone_callback(msg)`
    - 오도메트리 미도착 시 리턴(필수 가드)
    - 오도메트리로부터 현재 센서 포즈 `sensor_pose` 구성(Eigen)
    - 메시지에서 내부 관측 포맷으로 변환(`from_ros_msg`)
    - `msg->header.frame_id`에서 `base_link`로의 TF 조회 실패 시 항등 사용
    - 각 콘을 센서 2D → 센서3D(z=0) → `base_link` 좌표로 변환 후, 관측은 차량 기준 상대좌표로 유지(그래프 요인 요구사항)
    - 전처리 수행: `preprocessor_->process(observations, sensor_pose, stamp)`
    - 키프레임 판정: `should_create_keyframe(sensor_pose)`
    - 키프레임이면 `EstimationFrame` 생성(타임스탬프, 포즈, 관측, `is_keyframe=true`)
      - `SimpleConeMapping` 모드: `frame->id=0` 고정으로 추가(디버깅용)
      - 기본 `ConeMapping`: `get_next_pose_id()`로 ID 할당 후 `add_keyframe(frame)`
    - 경로 누적: `nav_msgs::Path slam_path_`에 현재 포즈 push
  - `should_create_keyframe(current_pose)`
    - 이전 키프레임 미존재 시 true
    - 평행이동>threshold 또는 회전>threshold 이면 true
  - `publish_odometry(result)`
    - `odom` 프레임 기준 포즈/속도 퍼블리시, 각속도는 `T_prev_curr`에서 추정
    - 내부 `dt=0.1` 고정값 사용(주의)
  - `visualization_callback()`
    - 시각화 타임스탬프는 최근 오도메트리 or `now()`
    - Simple/Full 모드 분기
      - Simple: MarkerArray 생성(원통, 색상 매핑), `SLAMVisualizer`를 통해 랜드마크/그래프 시각화, 최신 포즈 키 탐색
      - Full: 매핑으로부터 랜드마크/그래프/값 획득 → `SLAMVisualizer`로 시각화, 최신 포즈를 `/slam/pose`로 퍼블리시, `map->base_link_slam` TF 브로드캐스트, 경로/키프레임 갱신 시각화

- **좌표계 및 TF 정리**
  - 입력 관측은 센서 프레임(`msg->header.frame_id`) → `base_link`로 변환 후 그래프에는 차량 기준 상대 좌표로 반영
  - `map -> odom`은 일시 항등으로 고정(주기 타이머)
  - `map -> base_link_slam`은 최적화 결과 기반으로 퍼블리시(시각화 콜백)
  - `odom -> base_link`는 외부 EKF가 퍼블리시(충돌 방지 위해 SLAM 측 비활성)

- **주의/리스크**
  - `map->odom` 항등 고정은 장기 주행 시 틈새 오차 관리가 불가. 드리프트 교정 재연결 필요
  - `publish_odometry`의 `dt=0.1` 하드코딩은 실제 주기 변동 시 속도 추정 오차 유발. 타임스탬프 차이로 계산 권장
  - `TF lookup` 실패 시 항등 사용은 안전하나, 센서 오프셋 무시에 따른 바이어스 가능
  - QoS BestEffort/Volatile 조합은 안정성보다 실시간성을 우선. 로깅/디버깅 시 유실에 유의
  - `MultiThreadedExecutor` 사용으로 시각화와 처리 병행. 공유상태 접근은 내부 컴포넌트에서 스레드 안전성 보장 필요

---

## loop_closure_detector.cpp

- **역할**: 콘 랜드마크의 “별자리(constellation)” 기술자를 기반으로 프레임 간 유사성을 평가하여 루프 클로저 후보를 검출/검증.
- **주요 공개 메서드**
  - `add_keyframe(frame, landmarks, recent_poses)`
    - 프레임 기술자(Descriptor) 생성 후 내부 DB에 저장. 시퀀스 관리 및 로그 풍부
  - `detect_loop_closures(query_frame, landmarks)`
    - 질의 프레임 기술자 생성 → 후보 탐색(`find_candidates`) → 각 후보에 대해 정합 검증(`validate_loop_closure`) → 점수로 정렬/상한
  - `prune_old_descriptors(keep_recent_n)`
    - 메모리 관리: 최근 N개만 유지

- **Descriptor(별자리) 관련**
  - `ConstellationDescriptor`
    - 구성: 중심점, 중심 기준 상대 위치/각도/거리, 색상 카운트, 거리/각도 히스토그램, 공간 공분산, 경로 세그먼트, 기하학 특징
    - `distance_to(other)`
      - 거리/각도 히스토그램과 색상 분포 간 카이제곱 거리 합산
      - 경로 유사도(길이, 평균 곡률, 곡률 프로파일 상관) 반영
      - 기하학 특징 매칭 결과 반영
      - 가중 합산 비율: 별자리 0.3, 경로 0.3, 기하 0.4
    - `is_compatible_with(other)`
      - 콘 개수 차이≤5, 색상 분포 50% 이상 겹침 요구
    - `path_similarity(other)`
      - 총 길이/평균 곡률 차, 곡률 프로파일 정규화 상관계수. 가중 합산
    - `geometric_features_match(other)`
      - 동일 타입 특징 간 각도 변화 20% 이내를 매칭으로 간주, 50% 이상 일치 요구
  - `build_descriptor(frame, landmarks, recent_poses)`
    - `frame->observation_to_landmark`와 맵 `landmarks`로 실제 관측된 맵 상 콘들의 2D 위치/색상 취득
    - 중심 기준 상대 좌표로 변환, 반경 `max_constellation_radius` 내만 포함
    - 콘 수가 `max_cones_per_constellation`를 초과하면 중심에 가까운 순으로 절단 후 색상 카운트 재계산
    - 기하 특징 계산: 
      - 쌍별 거리 히스토그램(최대 상호 거리로 정규화, bin 수 만큼 누적 후 정규화)
      - 삼중 조합 각도 히스토그램(0~π, bin 정규화)
      - 공간 공분산 계산
    - `recent_poses` 존재 시 경로 세그먼트 생성(`build_path_segment`) 및 곡률 기반 특징 검출은 임시 비활성(주석)

- **후보 탐색 및 검증**
  - `find_candidates(query)`
    - 너무 최근 프레임은 시퀀스 간격 `min_keyframes_apart` 미만이면 스킵
    - 중심점 간 거리 `max_distance_for_loop * 2` 초과 시 배제
    - `query.distance_to(descriptor)`로 기술자 거리 계산 후, 근접 시 공간 보너스(거리가 짧을수록 감점) 적용
    - 점수(작을수록 좋음)가 `descriptor_match_threshold * 1.5` 미만인 경우만 채택, 상위 `max_candidates_per_query`까지 반환
  - `validate_loop_closure(query_frame, reference_frame, landmarks, out_candidate)`
    - 두 프레임에서 맵에 매핑된 콘 위치/ID를 수집
    - 색상 일치(혹은 UNKNOWN 허용) 기반으로 초기 대응쌍 후보 생성
    - RANSAC(`estimate_relative_pose`)으로 2D 강체 변환 추정 및 인라이어 집합 산출
    - 점수 = 1.0 - inlier_ratio, 인라이어 매칭을 `out_candidate`에 기록
  - `estimate_relative_pose(query_cones, reference_cones, matches, out_pose, out_inliers)`
    - 3개 샘플을 반복 추출하여 변환을 추정하고 인라이어 최대 해 선택, 충분 시 조기 종료
  - `compute_transform_svd(src, dst)`
    - 2D Kabsch 절차(SVD)로 회전/병진 계산, det<0 보정, `gtsam::Pose2` 변환

- **경로/곡률 처리**
  - `build_path_segment(poses)`
    - 포즈 시퀀스로부터 총 길이, 곡률 프로파일, 평균 곡률 계산 및 검증
  - `compute_curvature_profile(poses)`
    - 인접 3점으로 각 변화/평균 거리로 곡률 추정, 수치 이상값 보호
  - `detect_geometric_features(path)` / `classify_feature(curvatures, total_angle_change)`
    - 슬라이딩 윈도우로 특징 추출, 헤어핀/치케인/직선/좌우회전/전이 등의 라벨 분류(임계 기반)

- **주요 설정 키(구조체 `Config`에 존재, 코드 내 참조 기준)**
  - 히스토그램: `distance_histogram_bins`, `angle_histogram_bins`
  - 별자리 제한: `max_constellation_radius`, `max_cones_per_constellation`, `max_inter_cone_distance`
  - 후보/유사도: `descriptor_match_threshold`, `max_candidates_per_query`, `min_keyframes_apart`, `max_distance_for_loop`
  - RANSAC: `min_matched_cones`, `ransac_iterations`, `ransac_inlier_threshold`
  - 기하 특징: `min_feature_length`, `turn_angle_threshold`, `straight_threshold`, `curvature_threshold`, `hairpin_angle_threshold`

- **복잡도/성능 주의**
  - 각도 히스토그램은 O(N^3) 조합이므로 콘 다수일 때 비용 증가. `max_cones_per_constellation`로 상한 설정 중요
  - 후보 전수 비교 시 프레임 수 증가에 따른 비용 증가. `prune_old_descriptors`로 DB 용량 관리 권장

---

## cone_stellation/factors/inter_landmark_factors.cpp

- **상태**: 실제 팩터 구현은 헤더(`cone_stellation/factors/inter_landmark_factors.hpp`)에 인라인으로 존재. 본 파일은 분리 구현 여지 및 컴파일 분할 포인트.
- **관련 기능(헤더 참조)**: 콘-콘 간 거리 제약(ConeDistanceFactor) 등 상호 랜드마크 팩터 정의와 잡음 모델 구성.

---

## cone_stellation/preprocessing/cone_preprocessor.cpp

- **상태**: 복잡한 전처리 메서드가 필요할 경우 헤더에서 이곳으로 이전 가능. 현재는 헤더 수준 구현이 충분하다는 주석.
- **예상 기능(헤더 참조)**: 거리/신뢰도 필터링, 패턴(라인) 감지, 트래킹 유지, 관측-랜드마크 연관 전처리 등.

---

## 시스템 흐름 요약(E2E)

1. `/cones/fused/ukf` 수신 → 센서→`base_link` 변환 → 전처리
2. 키프레임 조건 충족 시 `EstimationFrame` 생성 → 매핑(`ConeMapping`)에 추가
3. 매핑은 ISAM2 등을 통해 최적화 수행(헤더 구현부)
4. 시각화 콜백에서 랜드마크/그래프/포즈/경로/키프레임 퍼블리시 및 TF(`map->base_link_slam`) 송출
5. 루프클로저는 별자리 기반 후보 탐색/검증 로직 제공(통합 시 `optimize_on_loop_closure` 경로 재활성 필요)

---

## 통합/의존성 포인트

- ROS2: `rclcpp`, 메시지(`geometry_msgs`, `nav_msgs`, `sensor_msgs`, `visualization_msgs`)
- TF2: `tf2_ros`(Buffer/Listener/Broadcaster), `tf2_eigen`, `tf2_geometry_msgs`
- 수학/최적화: `Eigen`, `gtsam`
- 커스텀 인터페이스: `custom_interface::msg::TrackedConeArray`

---

## 개선 제안 및 주의사항 체크리스트

- **map->odom 항등 고정 해제**: 드리프트 매니저와 EKF 간 순환 의존성 없이 안전하게 업데이트하는 경로 복구
- **속도 추정의 `dt` 동적화**: `publish_odometry`에서 이전-현재 타임스탬프 차로 속도/각속도 계산
- **TF 조회 타임 기준 명시**: `TimePointZero` 대신 메시지 타임스탬프 기반 조회 옵션 고려
- **루프클로저 통합**: `mapping.optimize_on_loop_closure` 파이프 재연결, 실패/오검 거부 전략 추가
- **성능 최적화**: 별자리 각도 O(N^3) 경감(샘플링, 근접 그래프 기반 축소), DB 프루닝 정책 조정
- **스레드 안전성**: 비주기 콜백 간 공유 자료 접근에 대한 뮤텍스/락 범위 검토
- **QoS/신뢰성**: 개발/디버깅 시 일시적으로 Reliable/TransientLocal 전환하여 로깅 안정성 확보

---

## 빠른 확인(현 코드 기준)

- 노드 이름: `cone_slam`
- 프레임: 입력 센서 프레임 → `base_link`(관측), 최적화 기준 `map`, 외란 통합 `odom`
- 토픽: 구독(`/cones/fused/ukf`, `/odometry/filtered`), 퍼블리시(`/slam/pose`, `/slam/odometry`)
- TF: `map->odom`(항등, 100ms), `map->base_link_slam`(시각화 콜백)
- 키프레임 조건: 평행이동 > 1.0 m 또는 회전 > 0.2 rad
- 루프클로저: 별자리 기술자+RANSAC, 가중 유사도(0.3/0.3/0.4)

본 문서는 소스 기준으로 작성되었으며, 헤더 구현부에 더 풍부한 세부사항이 존재합니다. 루프클로저와 드리프트 보정의 재통합 시 상호작용(TF, 최적화 트리거, EKF 연동) 설계를 우선 검토하시기 바랍니다.

---

## 실데이터 운영 체크리스트(요약)

### 데이터 경로/시간 정합
- INS/EKF(`src/INS/gps_imu_fusion`) 타임스탬프와 SLAM 입력(`TrackedConeArray`) 동기화 확인(Clock/SimTime)
- TF 조회는 메시지 시각 기반 옵션 검토, 실패 시 지수 백오프/진단 로그

### TF 역할 분리
- EKF: `odom→base_link` 고주기(≥100 Hz)
- SLAM: `map→base_link_slam` + `map→odom`(드리프트 보정). 초기 `map→odom=I` 1회 게시 후 드리프트 매니저 경로만 사용
- 충돌 금지: SLAM이 `odom→base_link` 게시하지 않도록 보장

### 데이터 연관/키프레임
- 1차 게이팅(유클리드+색+예상 이동) → 2차 게이팅(Mahalanobis)
- 전역 매칭(Hungarian/JCBB) 및 트랙ID 신뢰도 가중
- 키프레임 임계값 속도 적응($T_{trans}=a+bv$, $T_{rot}=c+dv$) + 정보기반 보조 기준

### Inter-Landmark 제약 운용
- 공관측 카운트/거리 범위/색 호환/레이트 리밋 조건 충족 시에만 생성
- `(min(i,j),max(i,j))` 레지스트리로 중복 차단, K-프레임 배치 업데이트, 로버스트 커널 적용
