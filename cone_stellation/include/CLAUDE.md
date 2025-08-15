# cone_stellation/include 분석 보고서

본 문서는 `cone_stellation/include` 이하에 존재하는 모든 공개 헤더(`.hpp`)를 기반으로, 시스템 구성, 주요 클래스/알고리즘, 모듈 간 상호작용, 설계 의도와 제약, 잠재적 리스크 및 개선 제안을 정리한 상세 분석서입니다. (작성일: 2025-08-13)

---

## 1) 디렉토리 구조 개요

- `common/`: 핵심 데이터 타입과 레코드
  - `cone.hpp`: 콘 관측/랜드마크/패턴/관측 집합 정의
  - `estimation_frame.hpp`: 추정 프레임(한 시점의 전체 데이터 집합)
  - `tentative_landmark.hpp`: 승격 대기(임시) 랜드마크 관리 및 통계
- `factors/`: GTSAM 기반 커스텀 팩터들
  - `cone_observation_factor.hpp`: 포즈-랜드마크 관측 팩터(직교 위치, 또는 범위/방위 대안)
  - `inter_landmark_factors.hpp`: 랜드마크 간 거리/선형/각도/평행 제약 팩터
- `mapping/`: 맵핑과 최적화 로직
  - `cone_mapping.hpp`: 메인 맵퍼(ISAM2, 관측 처리, 요인 생성, 최적화)
  - `data_association.hpp`: 최근접+컬러/트랙ID 제약 기반 연합
  - `simple_cone_mapping.hpp`: 단순/보수 설정 맵퍼(디버깅/베이스라인)
  - `loop_closure_detector.hpp`: 루프 폐쇄 후보 탐지(설계/인터페이스 정의 중심)
  - `cone_mapping_safe.hpp`: 안전 승격 대체 함수(주의: 현재 컴파일 불가 상태, 상세 하단)
- `odometry/`: 오도메트리 추정
  - `cone_odometry_base.hpp`: 공통 인터페이스/설정
  - `cone_odometry_2d.hpp`: 2D 콘 기반 상대변환 최적화
  - `async_cone_odometry.hpp`: 비동기 추정 래퍼
- `preprocessing/`
  - `cone_preprocessor.hpp`: 관측 전처리(아웃라이어 제거, 패턴 검출, 간단 트래킹)
- `util/`
  - `ros_utils.hpp`: ROS2 메시지 변환/마커 생성 유틸
  - `drift_correction_manager.hpp`: map→odom 드리프트 보정(포즈 이력 보간)
- `viewer/`: 시각화 구성요소
  - `viewer_base.hpp`, `viewer_manager.hpp`
  - `cone_viewer.hpp`, `pose_viewer.hpp`, `track_viewer.hpp`, `optimization_viewer.hpp`
  - `slam_visualizer.hpp`, `slam_visualizer_improved.hpp`, `loop_closure_viewer.hpp`, `visualization_utils.hpp`

---

## 2) 데이터 모델과 공통 타입(`common/`)

- `ConeColor`: Formula Student 표준 색 체계(UNKNOWN/YELLOW/BLUE/ORANGE/RED)
- `ConeObservation`
  - 차량(센서) 좌표계의 2D 위치, 공분산, 색상, confidence, 임시 id
- `ConeLandmark`
  - 전역 맵 프레임의 2D 위치, 색상, 관측 카운트, 신뢰도
  - 공관측(co-observation) 추적: 단순 집합 + 카운트 맵 제공(버그 수정 흔적 반영)
  - 트랙ID 점수 기반 주 트랙ID 유지
- `ConePattern`
  - LINE/CURVE/PARALLEL/CORNER 등 기하 패턴의 타입, 참여 cone id 집합, 파라미터, confidence
- `ConeObservationSet`
  - 단일 포즈에서의 관측 묶음 + 감지 패턴 + 센서 포즈/시각
- `EstimationFrame`
  - 한 시점의 모든 추정 입력/중간결과: 포즈들, 속도/IMU 바이어스(필드만), 콘 관측, 패턴, 연합 결과 등
  - `transform_to_world` 로 관측을 월드 2D로 변환(T_world_sensor × [x,y,0])
  - 사용자 정의 데이터 저장소(boost::any)
- `TentativeLandmark`
  - 임시 랜드마크: 다수 observation 누적, 평균/분산, 색상 투표, 트랙ID 득표 점수, 프레임 목록, 시간구간
  - 승격 조건: 최소 관측수, 시간 폭, 위치 분산 상한, 색상 일관성 비율
  - 정적 파라미터 기본값: 관측≥3, 시간≥0.5s, 분산≤0.2 m^2, 색상 신뢰≥0.8, 버퍼 최대 15

### 2.1) `common/cone.hpp`

- **개요**: 콘 색상/단일 관측/랜드마크/패턴/한 프레임 관측 묶음을 정의. Eigen 사용으로 정렬 주의 필요.
- **세부 분석**
  - **ConeColor**: Formula Student 규약에 맞춘 열거형. `UNKNOWN/YELLOW/BLUE/ORANGE/RED`.
  - **ConeObservation**:
    - 필드: `id`(관측 내 임시ID, 기본 -1), `position`(센서좌표 2D), `covariance`(2×2), `color`, `confidence`.
    - 기본값: `position=0`, `covariance=I`, `color=UNKNOWN`, `confidence=1.0`.
  - **ConeLandmark**:
    - 필드: `id_`, `position_`(맵 좌표 2D), `color_`, `observations_`(카운트), `confidence_`, 공관측 추적 `co_observed_cones_`/`co_observation_counts_`.
    - 트랙ID 관리: `primary_track_id_`와 `track_id_scores_`를 유지. `set_track_id`는 초기 점수 1.0 부여, `update_track_id`는 점수 누적 후 최고 득점 ID로 갱신.
    - 공관측 API: 단건/배열 추가, 집합 포함여부, 공관측 카운트 조회 제공.
  - **ConePattern**:
    - 타입: `NONE/LINE/CURVE/PARALLEL/CORNER`.
    - `cone_ids`, `parameters`(Eigen::VectorXd), `confidence` 포함. 파라미터 의미는 패턴별로 상이(추가 문서화 필요).
  - **ConeObservationSet**:
    - 필드: `cones`, `sensor_pose`(Isometry3d), `timestamp`, `detected_patterns`.
    - `get_cone_ids()`는 `id>=0`만 반환. `has_valid_pattern(min_cones=2)`은 단순 개수판정.
- **문제점/주의사항**
  - **헤더 누락**: `std::map` 사용(`co_observation_counts_`, `track_id_scores_`)에도 `<map>` 미포함. 명시 include 필요.
  - **정렬(Alignment)**: `std::vector<ConeObservation>` 등 Eigen 멤버를 포함한 타입을 컨테이너에 저장 시 `Eigen::aligned_allocator` 사용 권장.
  - **트랙ID 초기 상태 처리**: `update_track_id` 비교에서 `track_id_scores_[primary_track_id_]`는 `primary_track_id_==-1`일 때 불필요한 키 생성 위험. `find()` 체크 또는 가드 필요.
  - **신뢰도 범위**: `update_confidence(double)`는 [0,1] 클램프 없음. 입력 검증/클램프 권장.
  - **맵 오염 방지**: 공관측/트랙ID 맵이 무한 성장 가능. 오래된 키 pruning 정책 고려.
  - **패턴 파라미터**: `parameters`의 차원/의미 문서화 필요. 패턴별 구조체 분리 검토.

### 2.2) `common/estimation_frame.hpp`

- **개요**: 한 시점의 추정 입력/상태/연합 결과를 담는 컨테이너와 서브맵 관리용 `SubMap` 정의.
- **세부 분석**
  - **EstimationFrame**
    - 포즈: `T_world_sensor`(월드→센서), `T_sensor_base`(센서→베이스). `T_world_base()`는 두 변환을 곱해 반환.
    - 모션 상태: `v_world`, `imu_bias_acc/gyro` 필드만 보유(추정/갱신은 외부 책임).
    - 관측: `cone_observations`와 `detected_patterns`를 프레임 단위로 보관.
    - 연합 결과: `observation_to_landmark` 로컬→글로벌 ID 매핑.
    - 변환: `transform_to_world(const ConeObservation&)`는 센서좌표 2D를 [x,y,0]으로 올린 뒤 `T_world_sensor`만 적용하여 월드 2D 반환. 센서→베이스 보정이 필요하면 호출자에서 `T_sensor_base` 고려 필요.
    - 커스텀 데이터: `boost::any` 기반 K/V 저장. 포인터 반환(`any_cast<T>(&)`)로 타입 불일치 시 nullptr.
    - 정렬: `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` 선언됨. 컨테이너 저장 시 정렬 주의.
  - **SubMap**
    - `frames`는 `EstimationFrame::Ptr` 벡터, `local_landmarks`는 서브맵 지역 랜드마크 맵.
    - `get_all_observations()`는 값 복사로 반환(비용 큼). 포인터/참조 반환 대안 고려.
    - `build_local_map()`는 구현 보류(연합 전략에 의존).
- **문제점/주의사항**
  - **성능/복사 비용**: `get_all_observations()`의 대량 복사 가능성. 사본이 필요한지 점검, 필요 시 `reserve()` 또는 이동/참조 사용.
  - **타입 안정성**: `boost::any` 대신 C++17 이상이면 `std::any` 고려. 키 상수화/래퍼 API로 오타 방지.
  - **정렬(Alignment)**: `EstimationFrame` 자체는 정렬 지원하지만, 만약 객체 자체를 `std::vector<EstimationFrame>`로 저장하면 `aligned_allocator` 필요.
  - **좌표 일관성**: 관측이 센서좌표 기준임을 전역에 명확히 문서화. 호출자 측에서 `T_sensor_base` 반영 유무를 혼동하지 않도록 주석/함수명 개선 검토.

### 2.3) `common/tentative_landmark.hpp`

- **개요**: 임시 랜드마크에 관측을 누적하고 통계/득표를 기반으로 승격 준비 상태를 판정.
- **세부 분석**
  - **LandmarkObservation**: 월드/센서 좌표 2D, 색상, 트랙ID, 타임스탬프, confidence, frame_id.
  - **TentativeLandmark**
    - 누적: `observations_`(deque)에 push, 러닝 합 `position_sum_`, 제곱합 `position_squared_sum_` 갱신.
    - 색상 투표: `color_votes_` 증가, `get_primary_color()`에서 최다 득표 반환.
    - 트랙ID 득표: 현재 ID 점수 +1, 다른 ID 점수 0.95로 감쇠. `update_primary_track_id()`로 최고 득표를 기본 ID로 유지(히스테리시스 1.5배).
    - 버퍼 제한: `max_observations_` 초과 시 FIFO 제거하며 통계/투표 롤백.
    - 평균/분산: `get_mean_position()`, `get_position_covariance()`는 대각 성분만 산출(비대칭 공분산 무시).
    - 승격 판정: 관측 수, 시간 폭, 위치 분산 상한, 색상 신뢰도(최다 득표/총 득표) 기준 충족 시 true.
    - 기본 파라미터: inline static으로 초기화(관측≥3, Δt≥0.5s, 분산≤0.2, 색상신뢰≥0.8, 버퍼≤15).
- **문제점/주의사항**
  - **헤더 누락**: `std::set` 사용(`get_observing_frames`)에 `<set>`, `std::max`에 `<algorithm>` 필요.
  - **정렬(Alignment)**: `std::deque<LandmarkObservation>`은 Eigen 멤버를 포함. `Eigen::aligned_allocator<LandmarkObservation>` 사용 권장.
  - **공분산 근사**: `get_position_covariance()`가 대각만 사용(상관항=0 가정). 완전 공분산 추정(2×2)으로 개선 권장.
  - **트랙ID 갱신 히스테리시스**: `track_id_scores_[primary_track_id_]`는 `primary_track_id_==-1`일 때 불필요한 키 생성. `find()`로 안전 접근 또는 초기 가드 필요.
  - **해시 가능성**: `unordered_map<ConeColor,int>`는 구현에 따라 해시가 필요. 표준 해시가 미제공인 환경 대비해 `std::hash<ConeColor>` 특수화 고려.
  - **맵 성장 제어**: `track_id_scores_`/`color_votes_` pruning 미구현. 소수점 감쇠 임계 이하 키 삭제 정책 고려.
  - **주석 상충**: “Initialize static members (should be done in cpp file)” 코멘트와 inline 초기화가 상충. 현재(C++17) inline 변수는 허용되므로 주석 정정 또는 cpp로 이동 중 하나 선택.


---

## 3) 전처리(`preprocessing/cone_preprocessor.hpp`)

- 기능: 아웃라이어 제거, 패턴 검출(현재 라인만 구현), 간이 트래킹(id 할당)
- 주요 설정
  - 거리/신뢰도 기반 유효성, 라인 적합 임계, 최소 콘 수, 연합/트래킹 임계
- 구현 메모
  - `is_valid_observation`: 차량좌표계 거리와 confidence로 필터
  - `detect_line_patterns`: 3점 조합 기반 콜리니어성 검사 후 라인 파라미터(ax+by+c=0) 추정, 임계 내 추가 점 확장
  - `update_tracking`: 관측 id<0 에 대해 증가 id 부여
- 주의/개선점
  - 패턴 히스토리 `recent_patterns_`는 어디서도 push 되지 않아 외부에 히스토리 노출 없음
  - `update_tracking` 내부에 정적 `next_track_id`와 멤버 `next_track_id`가 동시에 존재(중복/불일치 위험)
  - O(n^3) 라인 탐색으로 관측 다수 시 비용 급증 → RANSAC/Hough 권장

### 3.1 상세 분석: `preprocessing/cone_preprocessor.hpp`

- **Config 기본값**: 거리 20m, 신뢰도 0.5, 라인 임계 0.2, 최소 3콘, 연합 1.0m, 트래킹 프레임 10.
- **process(...)**
  - 입력: 원시 관측(차량좌표), 현재 센서 포즈, 타임스탬프.
  - outlier 필터 후 패턴 검출(조건 충족 시), 마지막에 트래킹 업데이트. 결과에 `sensor_pose`/`timestamp` 저장.
- **is_valid_observation(...)**
  - 현재 좌표계 고정 이후 `sensor_pose` 미사용. 거리와 신뢰도만으로 판정. 필요 시 FOV/측정 공분산 기반 게이팅 추가 여지.
- **detect_patterns(...) / detect_line_patterns(...)**
  - 3중 루프 조합 검사로 콜리니어 판단. 라인 파라미터는 두 점 기반 노말벡터로 생성, 나머지 점을 거리 임계로 확장.
  - `are_cones_collinear`는 삼각형 면적/최대 변 길이 비로 임계 적용. `std::max`의 이니셜라이저 리스트 사용 → `<initializer_list>`는 필요 없지만 `<algorithm>`은 이미 포함 전제.
- **fit_line / distance_to_line**
  - 표현식 `ax+by+c=0`. 두 점 기반으로 단순 추정, 노이즈에 민감. 최소제곱/가중치 도입 여지.
- **update_tracking(...)**
  - 정적 지역 `next_track_id`로 음수 ID에 일괄 할당. 클래스 멤버 `next_track_id`와 중복.
  - 멤버 `tracked_positions_`는 미사용. 최근 위치 기반 ID 유지/재할당 로직 미구현.
- **동시성**
  - `recent_patterns_` 접근은 `patterns_mutex_` 보호. 반면 트래킹 상태는 락 없음.
- **개선 제안**
  - 트래킹: 헝가리안/카만 기반 추적, `tracked_positions_` 활용 및 멤버 `next_track_id` 단일화.
  - 성능: RANSAC/Hough로 O(n^3) 완화, 상호 거리 그래프 이용해 후보 축약.
  - 신뢰도: 관측 공분산을 노이즈 모델과 연동, 색상 일관성 반영.
  - API: `recent_patterns_` push 시점 구현, 외부에서 윈도우 크기/버퍼 관리.

---

## 4) 맵핑(`mapping/`)

### 4.1 `cone_mapping.hpp` (핵심 모듈)
- 역할: 키프레임 추가, 요인 그래프 구성, ISAM2 업데이트, 랜드마크 관리, 인터-랜드마크 요인 생성
- 설정 요약
  - ISAM2 재선형 임계/주기, 오도메트리/관측/인터랜드마크/패턴 노이즈
  - 인터랜드마크 사용 여부, 최소 공관측 횟수, 최대 랜드마크 간 거리
  - 연합 최대 거리, 최적화 주기, 루프 시 최적화 플래그
- 주요 흐름
  1) 첫 포즈는 Prior, 이후 프레임은 이전 포즈와 Between(2D) 요인 추가
  2) 각 관측에 대해 기존 랜드마크와 색상-거리 기반 최근접 연합
     - 초기 맵이 작을 때는 즉시 랜드마크 생성(≤30개까지), 일부에 Prior 부여(앵커)
     - 이후는 임시 랜드마크로 누적 후 승격
  3) 프레임 내 동시 관측된 확정 랜드마크 쌍에 대해 공관측 카운트 갱신, 기준 충족 시 거리 요인 생성
  4) 패턴 기반 요인 생성은 현재 비활성화(return)
  5) 누적된 `new_factors_`/`initial_values_`로 ISAM2 update, 현재 추정으로 랜드마크 좌표 갱신
- 연합 세부
  - 색상: 정확히 일치하거나 둘 중 하나 UNKNOWN이면 허용
  - 거리: 월드 좌표에서 임계 내 최단거리
- 임시→확정 승격
  - 관측수/시간/분산/색상신뢰 충족 시 회수, 초반 소수 랜드마크에 Prior 부여(앵커 역할)
  - 과거 프레임 기준 관측 팩터는 안정성 문제로 생략(주석) → 향후 관측이 팩터를 생성
- 인터-랜드마크 요인 생성
  - 프레임 내 관측된 확정 랜드마크 쌍 대상으로 공관측 카운트 누적, 거리 범위(0.1~max) 충족 시 거리 요인 추가
  - 측정값은 관측치가 아닌 현재 랜드마크 위치 간 거리 사용(안정성 위해)
- 로깅이 매우 상세(상태 추적에 유용)
- 리스크/개선
  - 초기 단계에서 “즉시 랜드마크 생성” 정책은 데이터 연합 오류가 많은 환경에서 오염 가능성 ↑
  - 패턴 요인 비활성화 상태 → 차후 성능 개선 위해 복구 필요
  - 공관측 카운트에 기반한 거리 요인은 중복 생성 방지 로직 부재(주석에 언급)

### 4.2 `data_association.hpp`
- 최근접 + 색상 제약 + 트랙ID 보너스(가중치를 음수로 점수 감소) 방식
- 입력 관측은 월드 좌표(호출자가 변환해서 넣는 계약), 랜드마크는 전역 위치
- 게이팅/마할라노비스 거리 함수 제공(현재는 직접 사용X)

### 4.3 `simple_cone_mapping.hpp`
- 보수적 ISAM2 파라미터, 첫 포즈 강한 Prior, 관측당 최대 3개 신규 랜드마크 생성 등 단순화
- `DataAssociation` 사용해 관측-랜드마크 연합
- 디버깅 로그/방어적 코드 풍부, 베이스라인으로 유용

### 4.4 `loop_closure_detector.hpp`
- 콘스텔레이션(랜드마크 배열) + 경로 곡률 기반 희박 지형에서의 루프 검출 설계
- Descriptor/Feature/PathSegment 구조 및 API만 정의됨(구현은 미노출)
- 인터페이스: 키프레임 추가/탐지/가비지 컬렉션, 후보 검증 등

### 4.5 `cone_mapping_safe.hpp` 주의
- 자유함수 `promote_tentative_landmarks_safe()`가 클래스 멤버(`landmarks_`, `next_landmark_id_`, `initial_values_` 등)에 직접 접근함
- 해당 심볼들은 어떤 클래스/네임스페이스에도 속하지 않음 → 현재 형태로는 컴파일 불가능
- 의도: `ConeMapping::promote_tentative_landmarks`의 안전 버전 대체 제시
- 조치 제안: 이 함수는 `ConeMapping` 클래스의 private 메서드로 편입하거나, 적절한 컨텍스트/객체를 파라미터로 전달하도록 리팩터 필요

#### 4.1 상세 분석: `cone_mapping.hpp`

- **구성/의존성**
  - 포함 헤더: `<set>`, `<sstream>`, GTSAM/ROS2/Eigen/자체 헤더. `std::max` 사용 지점이 있어 `<algorithm>` 누락 주의.
  - 네임스페이스: `cone_stellation` 내부 전담.
- **Config 기본값 검토**
  - `isam2_relinearize_threshold=0.1`, `skip=10`, `factorization=QR`, `evaluateNonlinearError=false`, `cacheLinearizedFactors=false`로 메모리/안정성 지향.
  - 관측 노이즈(`cone_observation_noise=0.5`), 인터랜드마크 거리 노이즈(`0.1`)는 상대적으로 타이트. 실제 센서 노이즈 대비 재조정 필요.
  - `enable_inter_landmark_factors=true`, `min_covisibility_count=2`, `max_landmark_distance=10.0`.
- **생성자**
  - ISAM2 파라미터 초기화 및 인스턴스 구성. 로깅 풍부.
- **add_keyframe(flow)**
  - 첫 포즈: `add_prior_factor`로 2D 포즈에 Prior 부여(시그마 0.1).
  - 이후: `add_odometry_factor`로 이전 포즈 대비 2D 변위/요 인자로 Between.
  - 관측 처리: `process_cone_observations` 호출. 프레임 저장 및 최적화 트리거.
- **add_prior_factor**
  - `T_world_sensor`에서 yaw 추출(`atan2(R(1,0),R(0,0))`)→`Pose2`로 삽입.
- **add_odometry_factor**
  - `T_prev.inverse()*T_curr`로 상대변환 계산, 2D 변환으로 변환 후 Between 추가.
  - 초기값: 이전 포즈 추정이 있으면 `prev_pose*delta`, 없으면 현재 절대포즈 사용.
- **process_cone_observations**
  - 프레임 단위로 관측 루프. 우선 확정 랜드마크 연합(`associate_with_confirmed_landmark`): 색상 일치 정책(UNKNOWN 허용), 월드 좌표 거리 게이팅.
  - 확정 실패 시: 맵 크기<30이면 즉시 신규 랜드마크 생성(+일부 Prior), 아니면 임시 랜드마크 연합/생성.
  - 승격 후 방금 프레임에서 텐터티브→확정 매핑을 반영하여 관측 팩터 추가.
  - 인터-랜드마크 팩터: 프레임 내 공관측된 확정 랜드마크 쌍마다 카운트 갱신 후 거리 요인 생성 여부 판단.
  - 패턴 요인 생성 루프 존재하나 현재 즉시 `return`으로 비활성화.
- **associate_with_confirmed_landmark**
  - 월드 좌표로 변환 후 색상 호환성 검사→최단거리 선택(임계는 `max_association_distance`).
- **add_observation_factor**
  - 측정치는 차량좌표계 2D(`obs.position`)를 사용해 커스텀 팩터 추가. 노이즈는 `config_.cone_observation_noise` 대각.
- **create_inter_landmark_factors / should_create_inter_landmark_factor**
  - 공관측 카운트와 거리 범위(0.1~max) 기반 승인. 중복 방지 장치 코멘트만 존재(실제 구현 없음).
  - 공관측 카운트 로깅 및 요인 생성 결과 요약 제공.
- **create_distance_factor**
  - 현재 `landmarks_` 값의 유클리드 거리로 측정 생성(안정성 지향). GTSAM 값/초깃값 존재 여부 확인 후 요인 삽입.
- **promote_tentative_landmarks**
  - 준비된 텐터티브를 확정으로 승격, 첫 3개에 Prior. 과거 프레임 관측 팩터는 크래시 위험으로 스킵(주석으로 잔존 코드).
  - 로깅으로 조건 충족 여부 상세 출력. `std::max` 사용으로 `<algorithm>` 필요.
- **optimize**
  - 새 팩터/값 없으면 스킵. 널 팩터 방어 제거, ISAM2 업데이트, 로컬 캐시 정리, 추정으로 랜드마크 위치 갱신.
- **문제점/리스크**
  - 헤더 누락: `<algorithm>`(std::max), `<cmath>`(atan2) 명시 포함 권장.
  - 인터-랜드마크 중복: 동일 쌍에 대해 프레임마다 중복 요인 생성 가능. 생성 레지스트리 필요.
  - 초기 즉시 생성 정책: 맵 오염 위험. 임시 누적→승격 우선, 또는 초기 Prior 더 보수적 적용 권장.
  - 스레드 안전성: 외부에서 동시 호출 시 맵/그래프 상태 보호 필요(현재 락 없음).
  - 관측 노이즈 고정: 관측 공분산 이용한 어댑티브 노이즈 모델 고려.

#### 4.2 상세 분석: `data_association.hpp`

- **구성/의존성**
  - 포함 헤더: `<unordered_set>`, ROS2/Eigen/자체 헤더. `std::numeric_limits` 사용하나 `<limits>` 누락.
- **Config**
  - `max_association_distance`, `use_color_constraint`, `gating_threshold`(카이제곱), `track_id_weight`, `use_track_id`.
- **associate(...)**
  - 입력 전제: 관측 좌표는 월드 프레임. 사용자는 호출 전 변환 필요.
  - 각 관측에 대해 미사용 랜드마크 중 색상 호환/거리 임계 내 최저 점수 선택. 트랙ID 일치 시 점수에 `-track_id_weight` 보너스.
  - 결과: 관측 인덱스→랜드마크ID(-1은 신규) 매핑, 1:1 매칭 유지.
- **mahalanobis_distance(...)**
  - 일반식 구현(사용처 미연결). 향후 게이팅 통합 여지.
- **문제점/리스크**
  - 헤더 누락: `<limits>` 필요.
  - 미사용 파라미터: `gating_threshold`, `pose_covariance` 미사용. 마할라노비스 게이팅으로 실제 적용 권장.
  - 점수 음수화: 트랙ID 보너스로 음수 점수 가능. 의도적이지만 로깅/임계 논리와의 일관성 검토.
  - 색상 UNKNOWN 처리: 현재 허용 로직은 ConeMapping과 일관성 유지 필요.

#### 4.3 상세 분석: `simple_cone_mapping.hpp`

- **개요**: 보수적 ISAM2 파라미터와 단순 데이터 연합으로 소규모 그래프를 안정적으로 유지하는 베이스라인.
- **핵심 흐름**
  - 첫 포즈 Prior(아주 타이트), 이후 `prev_pose.between(pose2d)` 기반 오도메트리.
  - 관측: 센서좌표 측정값을 팩터에 사용하고, 월드좌표는 값 초기화/연합에 사용.
  - 프레임당 신규 랜드마크 최대 3개 생성, 첫 3개에 Prior.
  - ISAM2 업데이트 후 `last_pose_` 갱신.
- **세부 포인트**
  - `DataAssociation` 사용: 월드 관측과 현재 내부 랜드마크 맵으로 최근접 연합.
  - 값/요인 디버그 로깅이 상세하여 추적 용이.
- **문제점/리스크**
  - 정렬/Allocator: `SimpleLandmark`는 Eigen 포함이나 `unordered_map` 값으로 안전. 다만 벡터 등 연속 컨테이너 저장 시 aligned allocator 필요.
  - 예외 처리: ISAM2 예외 재던짐. 상위 호출부에서 복구 전략 필요.
  - 노이즈 하드코딩: 관측 노이즈(0.2)와 오도메트리 노이즈 고정. 파라미터화 권장.

#### 4.4 상세 분석: `loop_closure_detector.hpp`

- **구성/의존성**
  - 포함 헤더: `<vector>`, `<unordered_map>`, `<queue>`, `<mutex>`, GTSAM/Eigen/ROS2/자체 헤더. `std::array` 사용하나 `<array>` 미포함.
- **데이터 타입**
  - `LoopCandidate`: 점수 낮을수록 우선. `operator<`는 힙 사용 고려하여 역순 정의.
  - `PathSegment`: 포즈 시퀀스, 총 길이, 평균 곡률, 곡률 프로파일.
  - `GeometricFeature`: 타입/각도 변화/세그먼트 길이/진입·이탈 방향.
  - `ConstellationDescriptor`: 중심/상대 위치/공분산/히스토그램/색상 카운트/경로/기하 특징.
- **중요 API**
  - `add_keyframe`, `detect_loop_closures`, `prune_old_descriptors`, `get_descriptor`.
  - 내부: `build_descriptor`, `compute_geometric_features`, `find_candidates`, `validate_loop_closure`, `estimate_relative_pose`, `compute_transform_svd`, `build_path_segment`, `compute_curvature_profile`, `detect_geometric_features`, `classify_feature`.
- **문제점/리스크(치명 포함)**
  - 헤더 누락: `std::array` 사용에 `<array>` 필요.
  - 색상 카운트 크기 오류: `std::array<int, 4> color_counts`이나 `ConeColor`는 5종(UNKNOWN 포함). 구현(.cpp)에서 `static_cast<int>(color)`로 인덱싱하므로 RED(4) 접근 시 OOB 발생 가능. 해결: 크기 5로 수정하거나 색상→인덱스 맵핑 함수 도입.
  - 미사용 포함: `<queue>`는 현재 사용되지 않음.
  - 스레드 안전: 외부에서 병렬 호출 시 `mutex_`로 보호되나, 반환 객체 참조 수명 주의.
  - 파라미터 일관성: `path_match_weight/geometric_feature_weight`가 .cpp의 가중과 일치하는지 검증 필요.

#### 4.5 상세 분석: `cone_mapping_safe.hpp`

- **개요**: `ConeMapping` 내부 승격 로직의 “안전 버전”을 자유함수로 제시. 현재 상태로는 컴파일 불가.
- **라인별 이슈**
  - 네임스페이스 외부 정의, 클래스 멤버(`tentative_landmarks_`, `next_landmark_id_`, `landmarks_`, `initial_values_`, `new_factors_`) 직접 접근 → 불가.
  - 필요한 헤더 전부 미포함(Eigen/GTSAM/ROS2/자체 타입). ODR/링크 에러 유발.
  - 정책 차이: “첫 몇 개 랜드마크 Prior 추가” 기준이 `landmarks_.size()<=3`로 정의되어 기존 구현(landmark_id<3)과 상이.
- **권장 조치**
  - `ConeMapping`의 private 메서드로 편입하거나, 컨텍스트 객체를 파라미터로 받아 동작하도록 리팩터.
  - 헤더 분리/구현 파일로 이동, 테스트 케이스로 동등성 검증.

---

## 5) 팩터(`factors/`)

- `ConeObservationFactor`
  - 포즈2-포인트2 바이너리 팩터, 측정은 차량좌표계에서의 상대 위치(Point2)
  - 에러 = pose.transformTo(landmark) − measured
- `ConeRangeBearingFactor`(대안)
  - 범위/방위(2×1) 측정 모델, 로버스트 커널과 함께 사용 용이
- `ConeDistanceFactor`
  - 두 랜드마크 간 거리 제약(1×1 잔차)
  - 주의: l1==l2 근접 시 분모 0 위험 → 호출 전 거리 하한 체크 필수(맵퍼에서 일부 방어)
- `ConeLineFactor`
  - 세 점의 정렬성 제약(정규화된 2D 크로스 결과 사용)
  - 단순 Jacobian 구현 포함(수치적 안정성은 입력 조건에 좌우)
- `ConeAngleFactor`
  - 세 점 사이 각도 제약(잔차만, Jacobian TODO)
- `ConeParallelLinesFactor`
  - 두 선 세그먼트의 평행성 제약(Jacobian TODO)

### 5.1) `factors/cone_observation_factor.hpp` 상세 분석

- 구조 개요
  - `ConeObservationFactor`: `NoiseModelFactor2<Pose2, Point2>` 상속. 차량좌표계에서의 2D 상대 위치 측정을 모델링.
  - `ConeRangeBearingFactor`: 동일 관계를 범위/방위(극좌표)로 모델링한 대안 팩터.

- 핵심 로직 및 라인별 메모(요지)
  - 1–8: GTSAM 기본 포함, `Pose2`, `Point2`, 행렬/벡터 타입 포함.
  - 17–31: `ConeObservationFactor` 선언부
    - 23–24: 측정치 `measured_`를 차량 프레임 `Point2`로 보관.
    - 26–29: 생성자에서 `pose_key`, `landmark_key`, `measured`, `noise_model` 설정.
    - 36–49: `evaluateError` 구현
      - 42–44: `pose.transformTo(landmark)`로 랜드마크를 차량 프레임으로 변환. 선택적 Jacobian 전달 시 GTSAM 내부에서 d(pred)/d(pose), d(pred)/d(landmark)를 계산해 채워줌.
      - 46–47: 에러 = 예측 − 측정. 표준 2×1 잔차.
    - 54–77: `clone/print/equals` 오버라이드. 디버깅 편의 제공, 동등성 비교에서 측정치 비교 포함.
  - 85–168: `ConeRangeBearingFactor` 선언부
    - 91–99: 측정값 `range_`, `bearing_` 보관. 생성자에서 초기화.
    - 105–147: `evaluateError`
      - 111–116: `relative = pose.transformTo(landmark)` → 예측 범위/방위 계산.
      - 118–121: 잔차 벡터 = [pred_range − range, normalized_bearing_error]. 방위 잔차는 `Pose2::Logmap` 트릭으로 2π 래핑을 암묵적으로 정규화.
      - 123–144: Jacobian 구성
        - 125–131: 상대 좌표에 대한 측정 Jacobian `H_rel` 구성(거리/방위의 도함수).
        - 135–138: 체인 룰 적용 위해 `pose.transformTo`의 Jacobian을 취득(`H_pose_transform` 2×3, `H_landmark_transform` 2×2).
        - 140–144: `H_pose = H_rel * H_pose_transform`, `H_landmark = H_rel * H_landmark_transform`.

- 정확성/안정성 평가
  - `ConeObservationFactor`
    - 차량 프레임에서의 직접 좌표 잔차로 모델링되어 직관적이며 Jacobian은 GTSAM이 제공 → 안정적.
    - SLAM 시각화에서 실제 측정점 노출을 원할 경우 측정치 게터가 없어서 접근 곤란. 시각화 개선용 `const gtsam::Point2& measured() const` 추가 권장.
  - `ConeRangeBearingFactor`
    - 방위 잔차 정규화에 `Logmap` 사용은 각도 래핑 처리에 유효.
    - Jacobian 유도는 표준적이며 체인 룰 적용도 적절.
    - 취약점: `predicted_range → 0` 근접 시 `H_rel`의 분모 `predicted_range`, `range_sq`로 인해 수치 불안정/분모 0 가능. 최소 거리 ε 가드 필요.

- 성능/설계 메모
  - Range/Bearing 모델은 비선형성이 커 초기 추정 민감. 초기값 품질이 낮으면 좌표 잔차 모델이 더 견고한 경우가 많음.
  - 로버스트 커널과 병용 시 외란에 강인. 현재 팩터 자체는 커널 비의존(외부에서 노이즈 모델 래핑).

- 발견된 문제/개선 제안
  - [개선] `ConeObservationFactor`에 `measured()` 게터 추가하여 시각화/디버깅에서 관측 벡터 활용 가능하게.
  - [안정성] `ConeRangeBearingFactor`에서 `predicted_range < ε` 시 Jacobian을 안전하게 처리하거나, 잔차를 스무딩(예: sqrt(range^2+ε^2)) 처리.
  - [일관성] `print()`에서 노이즈 모델 출력 외에 측정치/키 포맷을 통일된 프리픽스로 로깅하면 디버깅 가독성 향상.

### 5.2) `factors/inter_landmark_factors.hpp` 상세 분석

- 구조 개요
  - `ConeDistanceFactor`: 두 랜드마크 간 유클리드 거리 잔차(1×1)를 최소화.
  - `ConeLineFactor`: 세 랜드마크의 정렬성(정규화된 2D 크로스) 잔차.
  - `ConeAngleFactor`: 세 점에서 중앙 점의 끼인각 잔차.
  - `ConeParallelLinesFactor`: 두 선분(각 두 점)의 방향 평행 잔차(정규화된 방향 벡터의 크로스).

- 핵심 로직 및 라인별 메모(요지)
  - 17–49: `ConeDistanceFactor`
    - 21–22: `NoiseModelFactor2<Point2, Point2>` 상속, 측정 거리 보관.
    - 24–42: `evaluateError`
      - 25–33: `diff = l2 − l1`, `distance = ||diff||`, 잔차 = `distance − measured_distance_`.
      - 35–40: Jacobian: `H1 = −diff^T / distance`, `H2 = diff^T / distance` (형상 1×2).
      - 위험: `distance == 0` 시 분모 0. 호출 전/팩터 내부 가드 필요.
  - 56–115: `ConeLineFactor`
    - 63–80: 잔차 = 정규화된 2D 크로스 `cross(v1, v2) / (||v1||·||v2||)`.
    - 85–111: Jacobian을 `Matrix12`(1×2)로 각 점에 대해 근사 유도. `||v1||·||v2||` 작을 때 0으로 처리해 NaN 방지.
    - 주의: 정규화된 교차량은 스케일 불변성을 주지만 노이즈 분포가 비가우시안화될 수 있음.
  - 122–169: `ConeAngleFactor`
    - 134–152: 중앙점 l2에서의 끼인각 계산. `acos` 전에 `cos_angle` 클램프(수치 안정).
    - 153–161: 잔차 = `angle − measured_angle_`. Jacobian은 전부 0으로 설정(TODO) → 잘못된 정보 행렬을 유도할 수 있음.
    - 개선: 각도 잔차는 `wrapToPi(angle − measured)`로 래핑 필요. Jacobian 유도 또는 수치 Jacobian 사용 권장.
  - 175–223: `ConeParallelLinesFactor`
    - 190–208: 각 선분을 정규화한 후 방향 크로스(스칼라)를 잔차로 사용.
    - 215–220: Jacobian 전부 0(TODO). 선분 길이가 매우 짧으면 0으로 조기 반환.

- 정확성/안정성 평가
  - `ConeDistanceFactor`
    - 물리적으로 직관적, 거리 측정의 노이즈 모델을 1D로 단순화. 단, 0거리 특이점 방어 필요.
    - 제곱거리 접근(Residual = ||diff||² − d²)으로 분모 회피 가능하지만 단위/노이즈 모델 재해석 필요.
  - `ConeLineFactor`
    - 세 점의 정렬성은 트랙 직선 구간에서 강력한 제약. 단, Jacobian 유도식의 정확성 검증 필요(테스트 권장).
  - `ConeAngleFactor`/`ConeParallelLinesFactor`
    - Jacobian이 0으로 채워져 있어 최적화에 부정적 영향(허용하면 수렴 저해/왜곡 가능).
    - 현재 형태로 활성 사용하기보다, Jacobian 구현 또는 수치 Jacobian 경로로 전환 권장.

- 성능/설계 메모
  - 거리/정렬 잔차는 저차원이라 최적화에 부담이 적고 희박 환경에서 구조 유지에 유용.
  - 패턴 기반 제약(각/평행)은 강한 구조를 제공하나 정확한 도함수 없이는 역효과 가능.

- 발견된 문제/개선 제안
  - [안정성] `ConeDistanceFactor`에 ε-가드 추가 또는 호출부에서 최소 거리 필터를 엄격화.
  - [정확성] `ConeAngleFactor`, `ConeParallelLinesFactor`에 Jacobian 구현. 임시로는 H 미제공(=GTSAM 수치 미분 활용) 경로로 바꾸는 편이 안전.
  - [각 잔차] 각도 잔차는 래핑(`wrapToPi`) 적용으로 2π 경계 문제 방지.
  - [중복 제약] 동일 점 집합에 대한 중복 요인 생성 방지(레지스트리/해시)로 과구속 방지.

---

## 6) 오도메트리(`odometry/`)

- `ConeOdometryBase`
  - 공통 설정: 대응 최대거리/색상 제약/컬러 페널티, LM 파라미터, 외란 임계, 최소 대응 수
- `ConeOdometry2D`
  - 두 프레임 콘 대응을 찾아 임시 랜드마크와 관측 팩터로 구성된 소규모 그래프를 LM 최적화
  - 이전 포즈는 원점에 고정 Prior, 현재 포즈 추정치는 0에서 시작
  - 로버스트 커널(Huber) 적용으로 아웃라이어 억제
  - 최적화 후 inlier 카운트 산출, 3 자유도 변환을 3D Isometry로 반환(Z-회전)
- `AsyncConeOdometry`
  - 별도 스레드에서 프레임 큐 처리 → 결과 큐로 비동기 전달
  - 최초 프레임은 항등, 이후 누적(world_prev × T_prev_curr)

### 6.1) `odometry/cone_odometry_base.hpp`

- **파일 개요**: 콘 기반 오도메트리 추정의 추상 베이스. 프레임 간 상대변환 추정 API와 대응 찾기 인터페이스를 정의.
- **라인별 핵심**
  - L3–L5: `<memory>`, `Eigen` 포함. Eigen 타입을 멤버로 보유하지 않으므로 정렬 연산자 필요 없음.
  - L7–L9: 공통 타입(`cone.hpp`, `estimation_frame.hpp`) 의존성 선언.
  - L19–L22: `ConeOdometryBase` 정의, `Ptr` 별칭 제공.
  - L23–L38: `Config` 기본 파라미터 정의
    - `max_correspondence_distance=3.0`, `use_color_constraint=true`, `color_mismatch_penalty=10.0`
    - LM: `max_iterations=50`, `convergence_threshold=1e-6`
    - 로버스트: `outlier_threshold=2.0`, 최소 대응: `min_correspondences=3`
  - L40–L42: 기본 생성/소멸자.
  - L50–L53: `estimate(prev, curr)` 순수가상. 반환은 `T_prev_curr ∈ SE(3)`(여기서는 보통 평면 회전/평행이동만 사용).
  - L57–L68: `num_inliers()`, `get_correspondences()`, `name()` 순수가상.
  - L79–L83: `find_correspondences(prev_cones, curr_cones, initial_guess)` 순수가상. 초기치 기반 게이팅/매칭용 훅.
- **문제점/주의**
  - 필수 헤더 누락: `std::string`, `std::vector`, `std::pair` 사용 선언이 있으므로 `<string>`, `<vector>`, `<utility>`를 명시 포함 권장(현재는 파생/다른 헤더를 통해 우연히 포함될 수 있음).
  - 파라미터 검증 부재: `Config` 값 범위(음수 거리/임계 등)에 대한 유효성 보장이 없음.
  - 좌표계 계약 문서화: 인자 설명에 “vehicle frame” 언급이 있으나, `EstimationFrame`의 `T_sensor_base` 고려 여부를 인터페이스 주석에 명확히 기재 필요.
- **개선 제안**
  - `struct Config`에 팩토리/검증 함수 추가, 잘못된 값에 대한 클램프.
  - 인터페이스 주석에 반환/입력 좌표계 계약(Prev→Curr, 센서/베이스 기준)을 명시.
  - 필요 헤더를 명시적으로 포함하여 포함 순서 의존 제거.

### 6.2) `odometry/cone_odometry_2d.hpp`

- **파일 개요**: GTSAM을 이용한 2D(평면) 오도메트리. 두 프레임의 콘 대응으로 소규모 팩터그래프(Pose2×2 + Landmark2×N)를 구성하고 LM 최적화로 `T_prev_curr`를 추정.
- **라인별 핵심**
  - L3–L12: `unordered_map`, GTSAM 핵심(geometry, graph, LM), `Symbol`, `numericalDerivative` 포함. ROS 로그 매크로를 사용하지만 이 파일 내에 `rclcpp/rclcpp.hpp`는 포함되어 있지 않음(주의).
  - L34–L36: 생성자에서 `config_` 보관.
  - L40–L48: `estimate(prev, curr)` 시작. 관측 유무 검사 후 경고 로그 및 항등 반환.
  - L50–L58: 초기치는 항등. `find_correspondences` 호출해 대응 확보.
  - L59–L64: 최소 대응 수 미만이면 경고 및 항등 반환.
  - L66–L82: 그래프/초기값 준비, 이전 포즈를 원점으로 고정하는 강한 Prior(σ=1e-6) 부여.
  - L84–L109: 각 대응마다 임시 랜드마크 `l_prevIdx`를 생성하고, 이전/현재 프레임 관측을 각각 `ConeObservationFactor`로 추가. 관측 노이즈는 고정(σ=0.1m), Huber 커널로 로버스트화.
  - L112–L125: LM 최적화 실행 및 실패 시 항등 반환.
  - L127–L139: 결과에서 `Pose2`를 추출, `relative_pose = prev.between(curr)` 계산.
  - L141–L145: 2D 결과를 3D Isometry로 승격(Z-축 회전).
  - L147–L162: 잔차 기반 인라이어 카운트(두 프레임 모두에서 잔차 < `outlier_threshold`).
  - L164–L171: 대응/인라이어/델타 로깅 후 결과 반환.
  - L186–L249: 최근접(초기치 보정 포함) + 색상 제약 기반 대응 탐색
    - L194–L198: `initial_guess`를 2D `Pose2`로 축약(atan2로 yaw 추출).
    - L200–L206: 현재 콘을 이전 프레임으로 변환.
    - L219–L225: 색상 제약: 둘 다 UNKNOWN이 아니고 다르면 skip(색상 사용 시).
    - L228–L233: 거리 + 색상 불일치 페널티 적용.
    - L241–L246: 1:1 할당 유지(이전 프레임 콘 재사용 금지).
- **문제점/주의**
  - 헤더 누락: `RCLCPP_*` 매크로 사용하므로 `<rclcpp/rclcpp.hpp>` 명시 포함 필요. `std::atan2` 사용을 위해 `<cmath>` 포함 권장.
  - 로깅 파라미터: `params.verbosity` 주석은 “타입 불일치” 언급. GTSAM 4.x에서는 `params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;` 또는 `params.setVerbosityLM("SILENT");`가 올바름.
  - 관측 노이즈: 모든 관측에 고정 σ=0.1m를 사용. 각 콘의 `covariance`를 반영하지 않아 정보 활용이 제한됨.
  - 임계 혼용: `outlier_threshold`를 로버스트 커널 스케일과 잔차 판정(인라이어 카운트)에 모두 사용. 별도 파라미터 분리(커널 스케일 vs 인라이어 기준) 권장.
  - 초기치 활용 부족: `estimate()`에서 `initial_guess`를 항상 항등으로 설정. 직전 추정/기관측 이력 기반 예측치를 사용하는 것이 수렴/성능에 유리.
  - 색상 페널티 중복: 색상 사용 모드에서 불일치는 즉시 탈락(continue)이므로 페널티 분기가 실질적으로 UNKNOWN 포함 케이스에만 작동. 의도 명확화 필요.
  - 자료구조 선택: `prev_used`는 값이 중요하지 않으므로 `std::unordered_set<int>`가 더 적합/간결.
  - 예외 안전성: `result.at<Point2>(landmark)` 실패 가능성에 대비한 가드(존재 확인) 추가 시 디버깅 용이.
- **개선 제안**
  - 포함 정리: `<rclcpp/rclcpp.hpp>`, `<cmath>` 추가. 필요 시 `<Eigen/Geometry>`도 명시.
  - 노이즈 모델: 관측 공분산을 `noiseModel::Gaussian::Covariance`로 반영하고 프레임별 스케일링 도입.
  - 파라미터 분리: `robust_kernel_threshold`와 `inlier_residual_threshold`를 구분.
  - 초기치: `AsyncConeOdometry` 혹은 외부 모션모델에서 전달된 추정치를 사용하도록 API 확장.
  - 대응: 마할라노비스 게이팅/트랙ID 보너스 사용(프로젝트 내 `data_association` 로직 재사용 고려).

### 6.3) `odometry/async_cone_odometry.hpp`

- **파일 개요**: 오도메트리 추정을 별도 스레드에서 수행하는 래퍼. 프레임 큐 입력 → 결과 큐 출력의 비동기 파이프라인 제공.
- **라인별 핵심**
  - L3–L8: 원자성/스레드/동기화/컨테이너/ROS 로깅 포함. 시간측정을 위해 `<chrono>`가 필요하지만 미포함(아래 참고).
  - L28–L34: `OdometryResult` 구조체. `T_prev_curr`, `T_world_sensor`, `num_inliers`, `timestamp` 보관.
  - L36–L44: 생성/소멸. 소멸자에서 `stop()`으로 안전 종료 보장.
  - L48–L57: `start()`는 중복 시작 방지, 스레드 실행, 시작 로그 출력.
  - L62–L75: `stop()`는 kill 플래그 설정→CV 깨움→`join()`→상태 초기화.
  - L82–L95: `insert_frame()` 큐 용량 확인 후 push, 가득 차면 throttle 경고와 함께 드롭.
  - L101–L110: `get_result()`는 최신 결과 1건만 반환하며 내부 최신 포인터를 비움.
  - L116–L121: `get_all_results()`는 결과 큐를 통째로 비우며 반환.
  - L134–L201: `process_loop()`
    - 첫 프레임은 항등 누적 및 즉시 결과 발행.
    - 이후: `odometry_->estimate(prev, curr)` 실행, 시간 측정, `T_world_prev × T_prev_curr`로 누적.
    - 결과 발행 후 상태 업데이트.
  - L203–L213: `publish_result()`는 최신/큐에 결과 저장, 큐 길이 제한 유지.
- **문제점/주의**
  - 헤더 누락: 시간 측정(`std::chrono`)과 `std::vector` 사용을 위해 `<chrono>`, `<vector>` 명시 포함 필요. `Eigen::Isometry3d`를 직접 사용하므로 `<Eigen/Geometry>` 포함도 안전.
  - 포맷 문자열: `duration`은 일반적으로 `long long`인데 로그에서 `%ld` 사용. 플랫폼에 따라 포맷 불일치 가능. `%lld` 또는 `static_cast<long long>(duration)`과 `%lld` 사용 권장.
  - 스레드 안전성: `get_odometry()`로 외부에서 추정기를 취득해 동시에 설정 변경 시 경쟁 위험. 외부 쓰기 접근을 금지하거나 별도 락 설계 필요.
  - 쓰로틀 시계: `RCLCPP_WARN_THROTTLE`에서 매 호출마다 `Clock::make_shared()`를 생성. 멤버로 `rclcpp::Clock`를 보관/재사용하면 오버헤드 감소.
  - 종료 시 대기: kill 후 결과가 남아있을 수 있음. 종료 시 큐 정리/드레인 정책 명시 필요.
- **개선 제안**
  - 포함 정리: `<chrono>`, `<vector>`, `<Eigen/Geometry>` 추가.
  - 로깅: `%lld` 포맷 사용 또는 `RCLCPP_*`의 `fmt` 스타일 사용으로 타입 안전화.
  - API: 결과 콜백 등록(옵저버 패턴) 추가로 폴링 없이 소비 가능.
  - 스케줄링: 큐가 가득 찰 때 드래핑 정책을 LIFO/Latest Only 등 옵션화.

---

## 7) 유틸리티(`util/`)

- `ros_utils.hpp`
  - `TrackedConeArray` → `ConeObservation` 벡터 변환(문자열 색상 파싱, 거리 기반 공분산 생성)
  - 랜드마크/요인/키프레임 시각화 마커 생성기(프레임ID/타임스탬프 인자화)
- `drift_correction_manager.hpp`
  - `odom→base_link` 연속 포즈 이력 보관 후, SLAM의 `map→base_link`와 결합하여 `map→odom` 드리프트 추정
  - 시간 간 보간(SLERP/선형)으로 쿼리 시점 추정, 보정 크기 로깅

### 7.1 상세 분석: `util/ros_utils.hpp`

- **from_ros_msg(TrackedConeArray)**
  - `track_id`를 `ConeObservation.id`로 복사, 2D 위치 채움, 색상 문자열 소문자 변환 후 매칭. 미매칭 시 `UNKNOWN`로 로깅.
  - 공분산: 거리 기반 단순 모델 σ=0.1+0.02·range, R=σ²·I.
  - 문제/개선: 
    - 색상 파싱은 서브스트링 매칭이므로 오검 위험. 정확한 enum 매핑/strict 비교 권장.
    - 공분산 모델 고정. 센서별 분산/각도 의존 모델로 개선 여지.
    - `<string>` 명시 include 권장.
- **create_cone_markers(landmarks)**
  - DELETEALL 후 실린더+텍스트 마커 생성. 색상 매핑 일관. 투명도 0.8.
  - 개선: z스케일/프레임 파라미터화, 마커 수가 많을 때 네임스페이스 분할/배치 최적화.
- **create_factor_markers(graph, values)**
  - DELETEALL 후 요인별 라인 마커. 키 타입으로 요인 구분(odometry/observation/inter-landmark) 색/두께 지정.
  - Values에서 Pose2/Point2를 안전히 추출해 선분 생성. 예외 무시.
  - 개선/주의: 그래프가 커지면 마커 수 급증. 헤더에선 간단하지만 퍼포먼스 고려 필요. 3키 이상 요인 처리 없음.
- **create_keyframe_markers(poses)**
  - DELETEALL 후 ARROW+TEXT. RPY에서 yaw만 사용. 색상 시안.
  - 개선: 텍스트 겹침 회피, 스케일 동적 조정.

### 7.2 상세 분석: `util/drift_correction_manager.hpp`

- **역할**: 오도메트리 이력으로 쿼리 시점 포즈 보간 후 SLAM 포즈와 조합해 `T_map_odom` 갱신.
- **핵심 메서드**
  - `add_odometry_pose(t, T_odom_base)`: 버퍼 삽입 및 오래된 항목 제거.
  - `update_slam_pose(t, T_map_base)`: 보간 포즈 찾은 뒤 `T_map_base * (T_odom_base)^{-1}`.
  - `interpolate_odometry(t)`: `lower_bound`로 구간 찾고 translation 선형/rotation SLERP.
  - `get_map_to_odom()`, `set_history_duration()`, `get_buffer_size()`.
- **문제/개선**
  - 헤더 누락: `<algorithm>`(lower_bound, abs) 필요.
  - 스레드: 상호 잠금 범위 적절하나, 콜백 주기와 비용 고려해 비차단 큐/lock-free 고려 가능.
  - 외삽 로직: 앞/뒤 구간에서 첫/마지막 값 사용. 선택적으로 선형 외삽 옵션 제공 가능.
  - 드리프트 측정 로깅: yaw 추출 시 sign/범위 주의.

---

## 8) 뷰어(`viewer/`)

- `ViewerBase`: 공통 인터페이스(초기화/업데이트/클리어/이름)
- `SLAMVisualizer`
  - 랜드마크, 요인 그래프(유형별 제한/색상), 키프레임, 경로 퍼블리시
  - `visualizeFactorGraph`는 최근 요소 우선 노출, 루프 폐쇄 추정 분리
- `SLAMVisualizerImproved`
  - `ConeObservationFactor` 인식하여 실제 관측 화살표를 별도 네임스페이스로 표시
- `LoopClosureViewer`
  - 루프 후보 라인/변환 화살표/콘스텔레이션 반투명 구체(디스크립터 확산 반영)
- `ViewerManager`
  - 뷰어 집합 초기화/업데이트/토글/주기 타이머 관리 API
- `ConeViewer`, `PoseViewer`, `TrackViewer`, `OptimizationViewer`, `VisualizationUtils`
  - 각 역할별 마커 생성과 퍼블리시 헬퍼 제공

주의: 여러 뷰어 헤더들이 선언만 있고 정의는 별도 `.cpp`가 필요합니다. 본 레포가 "헤더 온리" 지향을 언급하지만, 이들 파일은 구현이 헤더에 존재하지 않습니다(프로젝트에 대응 소스가 있어야 빌드 가능).

### 8.1) `viewer/slam_visualizer.hpp` 상세 분석

- 구조/역할
  - `SLAMVisualizer`는 맵 랜드마크, 요인 그래프, 키프레임, 경로를 ROS2 토픽으로 시각화.
  - QoS를 BestEffort/Volatile로 설정하여 실시간성 우선.

- 구현 주요 포인트(라인 기준 요지)
  - 37–51: 퍼블리셔 초기화(QoS 설정 포함). 네임스페이스: `/slam/*`.
  - 70–68: `clear()`는 세 토픽 별 `DELETEALL` 게시.
  - 75–137: `visualizeLandmarks`
    - 첫 마커로 `DELETEALL`를 push 한 뒤 모든 랜드마크 실린더와 텍스트 마커 push.
    - 색상은 `setMarkerColor`로 ConeColor에 따라 설정, 알파 0.8.
  - 145–305: `visualizeFactorGraph`
    - 요인들을 odometry/observation/inter-landmark/loop-closure로 분류한 후 최근 N개만 그리도록 제한.
    - 문제: 165–199에서 요인 분류 시, `observation_factors.emplace_back(factor_index, &factor);` 형태로 루프 변수 `factor`의 주소를 저장. 루프 종료 후 해당 주소는 유효하지 않아 미정의 동작 발생 가능.
    - 216–250: 람다 `visualize_factor`에서 `keys.size()==2`만 처리. `values.exists` 확인 후 선분을 그림.
    - 201–211: 주기적으로 전체 삭제 마커를 넣는 타임 기반 정리.
  - 312–391: `visualizeKeyframes`에서 tf2 quaternion 사용. 파일에는 tf2 quaternion 헤더가 직접 포함되지 않음.
  - 420–466: `setMarkerColor`, `extractPosition` 유틸 내부 구현.

- 발견된 문제/개선 제안
  - [버그] 요인 분류 시 shared_ptr 변수의 주소를 저장하는 패턴은 잘못됨. 인덱스만 저장 후 `graph[index]`로 다시 접근하거나, `graph`의 내부 저장소에서 안정 참조를 얻는 방식으로 수정 필요.
  - [포함 누락] `tf2::Quaternion` 사용에 대해 `#include <tf2/LinearMath/Quaternion.h>` 추가 권장(간접 포함 의존 제거).
  - [성능] 큰 그래프에서 반복적 `values.exists`/`values.at` 호출 비용이 큼. 최근 값 캐시 또는 시각화 샘플링 개선 권장.
  - [안전] `keys.size()<2` 조기 continue는 적절하나, 다른 타입의 요인(예: 단항) 무시됨을 문서화 필요.

### 8.2) `viewer/slam_visualizer_improved.hpp` 상세 분석

- 구조/역할
  - `SLAMVisualizer`를 상속하여 관측 팩터의 실제 관측 레이/포인트를 별도 네임스페이스로 상세 시각화.

- 구현 주요 포인트/이슈
  - 26–29: `visualizeFactorGraph(const ..., const ...) override`로 선언되어 있으나, 기반 클래스의 동명 함수는 virtual로 선언되어 있지 않음. `override` 지정자는 컴파일 에러 유발.
  - 49–54: `ConeObservationFactor`만 별도 처리하고 그 외는 선분으로 처리.
  - 140–176: 측정치 getter 부재로 실제 측정점 대신 최적화된 랜드마크 위치를 사용한다고 주석. 시각화 정확도 저하.

- 발견된 문제/개선 제안
  - [컴파일] 기반 클래스(`SLAMVisualizer`)의 `visualizeFactorGraph`를 `virtual`로 변경하거나, 여기서 `override` 제거 + 함수명을 별도로 둘 것.
  - [기능] `ConeObservationFactor`에 `measured()` 게터 추가 후, `pose.transformFrom(measured)`을 사용해 실제 관측점을 렌더링.

### 8.3) `viewer/loop_closure_viewer.hpp` 상세 분석

- 구조/역할
  - 루프 폐쇄 후보, 변환 화살표, 콘스텔레이션 구체를 시각화.

- 구현 주요 포인트/이슈
  - 39–40: `MarkerPublisher`라는 타입을 사용해 생성하나, 본 리포 내 정의 없음. 외부 의존 또는 누락된 구현.
  - 145–173, 175–205, 207–243: `create_*` 계열에서 `line.header = create_header();` 형태로 미정의 `create_header()` 사용. 컴파일 불가.
  - 65–75: 루프 선분과 변환 화살표 게시. 점/색/폭 적절히 설정.

- 발견된 문제/개선 제안
  - [컴파일] `create_header()` 구현 추가(예: frame_id/time 설정) 또는 직접 필드 설정으로 대체.
  - [의존] `MarkerPublisher` 구현/포함 확인. 미제공 시 `rclcpp::Publisher<MarkerArray>`로 대체.
  - [표현] 점/구체 스케일 파라미터를 Config로 더 노출해 조정성 향상.

### 8.4) `viewer/visualization_utils.hpp` 상세 분석

- 구조/역할
  - Eigen↔ROS 메시지 변환, 색상/마커 유틸 정적 함수 집합(선언만 존재).

- 발견된 문제/개선 제안
  - [구현] 선언만 있으므로 .cpp 구현이 필요. 현재 헤더온리 아키텍처라면 `inline` 정의 제공 검토.
  - [일관] 마커 공통 기본값(프레임/색상/두께/수명)을 중앙에서 관리하도록 팩토리 스타일 제공 권장.

### 8.5) `viewer/viewer_manager.hpp` 상세 분석

- 구조/역할
  - 여러 뷰어의 생성/초기화/업데이트/클리어/주기 타이머 관리를 담당.

- 발견된 문제/개선 제안
  - [수명] `rclcpp::TimerBase::SharedPtr` 주기 타이머의 콜백이 `this` 캡처 시 객체 수명에 주의(구현부 확인 필요).
  - [토글] `enableViewer`/`isViewerEnabled`는 이름 문자열 기반 → 오타 위험. enum 기반 API 또는 상수 키 권장.

### 8.6) `viewer/pose_viewer.hpp` 상세 분석

- 구조/역할
  - 차량 포즈/궤적/속도 벡터/좌표축 시각화. `PoseData` 구조에 시간/자세/속도 포함.

- 발견된 문제/개선 제안
  - [프레임] `frame_id_` 기본값 "map". 외부 TF 체계에 맞춰 설정 노출 필요.
  - [ID관리] 궤적/텍스트/프레임 마커 ID 충돌 방지 로직이 구현부에 필요.
  - [성능] 궤적 길이(`max_trajectory_length_`) 관리로 마커 점 수 제한해야 RViz 부하 감소.

### 8.7) `viewer/optimization_viewer.hpp` 상세 분석

- 구조/역할
  - 최적화 이력(positions/covariances/cost/iter)과 제약(odometry/loop/cone_observation)을 시각화.

- 발견된 문제/개선 제안
  - [스케일] `covariance_scale_`는 시각화만 반영. 공분산 → 타원체 분해(EVD/SVD) 구현부 안정성 확인 필요.
  - [색상] 제약 타입별 색상 맵 `getConstraintColor` 고정값 → 구성 가능성 노출 권장.

### 8.8) `viewer/track_viewer.hpp` 상세 분석

- 구조/역할
  - 좌/우 경계 및 센터라인을 라인스트립/Path로 시각화. 색상 상수(좌Blue/우Yellow/센터Green).

- 발견된 문제/개선 제안
  - [프레임] `frame_id_` 기본 "map". 외부에서 설정/동기화 필요.
  - [해상도] 점 밀도가 높을 경우 마커 메시지 크기 증대. 리샘플링/간소화 옵션 고려.

### 8.9) `viewer/cone_viewer.hpp` 상세 분석

- 구조/역할
  - 단순 콘 검출 결과를 실린더 등으로 시각화. `ConeVisualization {position,type,confidence,id}`.

- 발견된 문제/개선 제안
  - [ID/삭제] 프레임마다 누적되는 마커를 정리하기 위해 `DELETEALL` 또는 일관된 ID 재사용 전략 필수(구현부 확인 필요).
  - [타입] 문자열 기반 색/타입 → 열거형 기반 API로 오타/매핑 리스크 축소 검토.

### 8.10) `viewer/viewer_base.hpp` 상세 분석

- 구조/역할
  - 모든 뷰어의 공통 인터페이스(순수 가상)와 이름/뮤텍스 제공.

- 발견된 문제/개선 제안
  - [스레드] 파생 클래스에서 `mutex_` 사용 일관성 필요. 모든 퍼블리시/상태 접근 경로에 락 적용 권장.
  - [가상 소멸자] `virtual ~ViewerBase() = default;` 적절. 상속 소멸 안전성 확보.

---

## 9) 전반적 데이터 플로우

1) 센서로부터 `TrackedConeArray` 수신 → `ros_utils::from_ros_msg`로 `ConeObservation` 변환
2) `ConePreprocessor`가 유효성/패턴/간이ID 처리 → `ConeObservationSet`
3) `EstimationFrame`에 포즈/관측/부가데이터 저장, 키프레임이면 맵퍼에 전달
4) `ConeMapping`이 Prior/Between/관측 팩터/인터랜드마크 팩터 구성 후 ISAM2 업데이트
5) 최적화 결과로 포즈/랜드마크 업데이트, 뷰어에 그래프/랜드마크/키프레임 시각화
6) 병행하여 `DriftCorrectionManager`가 map→odom 보정 추정, TF 브로드캐스트 등으로 소비 가능
7) `LoopClosureDetector`는 설계상 디스크립터 구축/후보 검색/검증을 제공(구현 필요)

---

## 10) 잠재적 문제/버그/주의사항

- 전처리
  - `ConePreprocessor::update_tracking`에 정적 지역 `next_track_id`와 멤버 `next_track_id`가 동시에 존재 → 일관성 깨짐 위험. 하나로 통일 필요.
  - `recent_patterns_`가 어디서도 갱신되지 않음 → `get_recent_patterns()`는 빈 결과를 반환할 가능성 큼.
  - 라인 탐색 O(n^3) → 관측이 많으면 병목.
- 맵핑
  - `cone_mapping_safe.hpp`는 현재 상태로는 컴파일 불가(자유함수가 클래스 멤버에 접근). 오용 시 빌드 에러.
  - 인터랜드마크 거리 요인 중복 생성 방지 장치 부재(주석으로만 언급). 별도 레지스트리/해시 필요.
  - 랜드마크 즉시 생성 임계(30개) 정책은 환경에 따라 노이즈 흡수 위험 → 데이터 연합 강화/임시 누적 선호 고려.
  - 패턴 기반 요인 생성 비활성화(기능 상실). 성능 향상을 원하면 재활성/견고화 필요.
- 팩터
  - `ConeDistanceFactor`: 두 점 매우 근접 시 Jacobian 분모 0 → 호출 전 거리 하한 보장 필요.
  - `ConeAngleFactor`, `ConeParallelLinesFactor`: Jacobian 미구현 → 최적화에 불리/수렴 지연 가능. 사용 전 구현 보강 요망.
- 뷰어/구현 파일
  - 여러 뷰어는 헤더에 선언만 있고 구현은 외부 `.cpp` 필요. 현재 워크스페이스에 구현이 없다면 링크 에러 가능.

---

## 11) 설정 파라미터 하이라이트(조정 가이드)

- 전처리
  - `max_cone_distance`, `min_cone_confidence`, `line_fitting_threshold`, `min_cones_for_line`
- 맵핑
  - ISAM2: `isam2_relinearize_threshold`, `isam2_relinearize_skip`
  - 노이즈: `odometry_noise`, `cone_observation_noise`, `inter_landmark_distance_noise`
  - 인터랜드마크: `enable_inter_landmark_factors`, `min_covisibility_count`, `max_landmark_distance`
  - 연합: `max_association_distance`
  - 최적화 주기: `optimize_every_n_frames`
- 오도메트리(2D)
  - 대응: `max_correspondence_distance`, `use_color_constraint`, `color_mismatch_penalty`
  - LM: `max_iterations`, `convergence_threshold`, `outlier_threshold`
- 임시 랜드마크 승격
  - `min_observations_`, `min_time_span_`, `max_position_variance_`, `min_color_confidence_`, `max_observations_`

---

## 15) 실데이터 전환 가이드: 연합/키프레임/TF 전략

### 데이터 연관(현장 강건화 체크리스트)
- 1차 게이팅: 유클리드 거리 + 색상 호환 + 속도 기반 예상 이동 범위(INS 속도/요 회전율로 예측)
- 2차 게이팅: Mahalanobis $d_M^2 = \nu^\top S^{-1}\nu \le \chi^2_{0.95,2}$ (`pose_cov` 근사 + 관측 공분산)
- 스코어링: $d + w_c[\text{색 불일치}] - w_t[\text{트랙ID 일치}]$ → Hungarian/JCBB로 전역 최적 매칭
- 트랙 ID 신뢰도: 최근 $N$프레임 일치율 기반 가중, UNKNOWN 색상 시 가중 상향
- 동적 콘: 급격 이동/반복 미매칭을 블랙리스트 윈도우로 임시 제외

### 키프레임(적응형 정책)
- 기본: $\|\Delta p\|>T_{trans}$ 또는 $|\Delta\theta|>T_{rot}$
- 속도 적응: $T_{trans}=a+bv$, $T_{rot}=c+dv$ (고속 시 더 큰 시차 요구)
- 정보기준: 관측 수/방향 엔트로피 증가/공분산 감소 기대

### TF/INS 통합(역할 분리)
- EKF(INS/GNSS): `odom→base_link` 고주기 게시(≥100 Hz)
- SLAM: `map→base_link_slam`, `map→odom`(드리프트 보정) 게시. `odom→base_link`는 SLAM 금지
- 절차: 초기 `map→odom=I` 1회 → 최적화 `T_map_base`와 보간 `T_odom_base`로 $T_{map\,odom}=T_{map\,base}T_{odom\,base}^{-1}$ → 5–10 Hz로 업데이트
- 금지: 고주파 `map→odom` 흔들림, 양방향 동일 변환 중복 게시

### Inter-Landmark Feature 운용 요령
- 생성 조건: 공관측 $\ge2$~3, $0.3\,\text{m}\le d\le d_{max}$, 색 호환, 프레임 레이트 리밋
- 중복 방지: `(min(i,j),max(i,j))` 레지스트리, 다중 요인 누적 금지(교체/재가중)
- 스케줄: 관측/오도메트리 반영 후 매 $K$프레임 배치 적용(5–10)
- 로버스트: Huber/Tukey 커널, 노이즈 $\sigma_d$ 0.05–0.2 m 스윕

---

## 16) 명시적 루프클로저 설계 가이드(GLIM 대비, ConeSTELLATION 적용)

### 배경/전략
- GLIM: 서브맵/스캔매칭/확장 모듈(DBoW, ScanContext) 등 풍부한 특징을 통해 후보를 찾고 Between/Matching 요인을 추가.
- ConeSTELLATION: 입력이 콘들뿐(희박 특징) → “암묵적 루프”(inter-landmark)로 드리프트 억제 + “명시적 루프”(별자리/경로/GNSS) 혼합.

### 파이프라인(권장)
1) 저비용 게이팅
   - 위치/자세 근접: $$\|\hat p_i - \hat p_j\| < d_{gate},\; |\Delta\theta| < \theta_{gate}$$
   - GNSS(옵션): $$\|p^{GNSS}_i - p^{GNSS}_j\| < d^{GNSS}_{gate}$$
   - 시퀀스 간격: 최근 $N$ 키프레임 제외
2) 별자리 디스크립터 매칭(`loop_closure_detector.*`)
   - 거리/각도 히스토그램 유사도 + 색상 분포 + 최소 콘 수
   - 상위 $K$ 후보만 유지
3) 변환 추정(RANSAC + 2D Kabsch)
   - 잔차: $$\|R\,p^i_k + t - p^j_{\pi(k)}\| < \epsilon$$
   - Kabsch 요약: $$H = \sum (p_k-\bar p)(q_k-\bar q)^\top = U\Sigma V^\top,\; R=VU^\top,\; t=\bar q - R\bar p$$
4) 그래프에 루프 요인 추가
   - 포즈 Between 요인(루프) + 필요 시 일부 랜드마크 보조 요인
   - 로버스트 커널, 중복 방지 레지스트리
5) 최적화 트리거
   - `optimize_on_loop_closure=true` 경로 활성화

### 파라미터(초안)
- $d_{gate}=5$~$15$ m, $\theta_{gate}=10^\circ$~$25^\circ$, $\epsilon=0.5$~$1.0$ m, 후보 $K=3$~$5$
- 히스토그램 bin: 16~32, 최소 공통 콘 수: 4~6

### 통합 포인트
- 입력은 `/cones/fused/ukf` 권장(센서 프레임 원본). `*_map` 사용 시 월드-관측 팩터 경로 분기 필요.
- GNSS 절대 위치 팩터(약한 prior)를 선택적으로 추가해 루프 실패 대비 전역 드리프트 억제.

### 주의
- 희박 환경에서 오연관 리스크가 높으므로 강한 게이팅 + 로버스트 커널을 기본값으로 채택.
- 루프 빈도 제한(레이트 리밋)과 중복 루프 방지 레지스트리를 운영.

## 12) 확장/개선 제안

- 전처리/패턴
  - 라인 검출 RANSAC/Hough 전환, 패턴 히스토리 관리 구현, 트래킹 ID 충돌 방지 정교화
- 맵핑
  - 인터랜드마크 요인 중복 방지(쌍 키셋 기록), 승격 로직에서 후관측 자동 팩터화 안전 경로 마련
  - 패턴 기반 요인 재활성화 및 견고한 매핑(삼중/다중 제약)
  - 데이터 연합 고도화(JCBB/멀티가설, 마할라노비스 게이팅 실사용)
- 팩터
  - 각/평행 팩터 Jacobian 구현, 수치 안정성 점검
- 루프 폐쇄
  - 디스크립터/곡률/기하 특징의 실제 구현 및 RANSAC 검증, 멀티스레드 매칭
- 인프라
  - `cone_mapping_safe.hpp` 정비(클래스 메서드로 통합), 뷰어 구현 파일 정합성 확인
  - 파라미터 YAML/동적 리컨피그(ROS2) 연동

---

## 13) 예시 처리 시퀀스(간략)

1) `/tracked_cones` 수신 → `from_ros_msg`로 `std::vector<ConeObservation>` 생성
2) `ConePreprocessor::process`로 유효성/패턴/트래킹 → `ConeObservationSet`
3) `EstimationFrame` 작성(포즈, 관측 세트, id, timestamp)
4) 키프레임이면 `ConeMapping::add_keyframe`
   - Prior/Between/관측/거리 요인 구성 → ISAM2 업데이트
   - `get_landmarks`/`get_factor_graph`/`get_poses`로 뷰어에 전달
5) `DriftCorrectionManager`는 SLAM 포즈로 map→odom 업데이트

---

## 14) 결론

본 코드베이스는 GLIM 아키텍처를 변용한 콘 기반 희박 지형 SLAM의 골격을 갖추고 있습니다. 핵심 차별점은 프레임 내 공관측을 활용한 인터-랜드마크 거리 제약으로, 희박 관측에서도 트랙 형상을 유지하도록 돕습니다. 다만 전처리 트래킹 id 불일치, 안전 승격 대체 헤더의 비컴파일 상태, 패턴 요인 비활성, 일부 팩터의 Jacobian 미구현 등 안정성과 성능을 위해 다듬어야 할 지점이 명확합니다. 위 개선 제안을 반영하면 실제 주행 환경에서의 견고성과 수렴 특성을 유의미하게 향상시킬 수 있습니다.
