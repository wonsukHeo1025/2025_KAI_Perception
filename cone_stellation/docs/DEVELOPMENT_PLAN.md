# ConeSTELLATION Development Plan (최신 버전)

## 1. 개요 (Introduction)

ConeSTELLATION은 Formula Student Driverless 자율주행을 위한 콘 기반 Graph SLAM 시스템입니다. LiDAR 클러스터링과 YOLO 기반 색상 분류를 통한 콘 검출을 처리하며, GLIM의 모듈화 아키텍처에서 영감을 받아 설계되었습니다.

**핵심 혁신**: 희박한 콘 관측 환경(프레임당 2-10개)에서도 안정적인 지도 구축을 위한 Inter-landmark 제약과 보수적 ISAM2 최적화

## 2. 현재 상태 (2025년 8월)

### ✅ 구현 완료 및 작동 중
- **핵심 SLAM**: GTSAM 기반 팩터 그래프, ISAM2 최적화
- **시각화**: RViz 종합 디스플레이 및 성능 최적화
- **센서 시뮬레이터**: 현실적 노이즈 모델의 IMU/GPS 시뮬레이션
- **IMU-GPS 통합**: robot_localization을 통한 EKF 융합 (100Hz)
- **TF Tree 관리**: 모든 좌표계 관계 해결 및 브로드캐스팅
- **Inter-landmark Factors**: 공관측된 콘 간 거리 제약 작동
- **데이터 연관**: 색상 제약 및 트랙 ID 지원
- **Rosbag 호환성**: 기록된 데이터 재생으로 신뢰성 있는 작동

### ⚠️ 부분 구현 (개선 필요)
- **드리프트 보정**: map→odom 변환이 항등으로 고정됨 (코드 주석 처리됨)
- **루프 클로저**: 별자리 기반 설계 완료, 통합 필요
- **AsyncConeOdometry**: 스레드 시작되나 프레임 주입 없음
- **패턴 팩터**: Line/Angle/Parallel - Jacobian 미구현

### ❌ 미구현 또는 버그
- **설정 무시**: `use_simple_mapping` 파라미터가 강제로 false 설정됨
- **Inter-landmark 중복**: 동일 쌍에 대한 중복 팩터 생성
- **패키지 의존성**: package.xml에 tf2_eigen, tf2_geometry_msgs 누락
- **팩터 Jacobian**: Line/Angle/Parallel 팩터의 Jacobian 구현 필요

## 3. 시스템 아키텍처

### 3.1 전체 설계
```
┌─────────────────────────────────────────────┐
│     외부 센서 (100Hz)                         │
│     IMU + RTK GPS → robot_localization      │
└────────────────┬────────────────────────────┘
                 │ Fused Odometry (100Hz)
                 ↓
┌─────────────────────────────────────────────┐
│     ConeSTELLATION SLAM (10-30Hz)          │
│                                             │
│  콘 검출 → 데이터 연관 →                      │
│  팩터 그래프 → 최적화 → 지도                  │
│                                             │
│  출력: map→odom 드리프트 보정                 │
└─────────────────────────────────────────────┘
```

### 3.2 아키텍처 결정사항

**멀티레이트 하이브리드 아키텍처**:
- **제어 레이어** (100Hz): robot_localization을 통한 IMU+GPS 융합
- **SLAM 레이어** (10-30Hz): 매핑 및 드리프트 보정
- **근거**: 레이싱 중 제어 안정성이 전역 정확도보다 중요

**TF Tree 정책**:
- `odom→base_link`: EKF가 고주기로 발행
- `map→base_link_slam`: SLAM이 발행
- `map→odom`: SLAM이 드리프트 보정 책임 (현재 항등으로 고정됨 - 수정 필요)

## 4. 개발 로드맵

### Phase 1: 긴급 안정화 (1-2주) 🔴

**목표**: 핵심 버그 수정 및 기본 기능 복구

**작업 목록**:
1. **패키지 의존성 수정**
   ```xml
   <!-- package.xml에 추가 -->
   <depend>tf2_eigen</depend>
   <depend>tf2_geometry_msgs</depend>
   ```

2. **하드코딩 제거**
   - 위치: `src/cone_stellation/ros/cone_slam_node.cpp:59-66`
   - 작업: `use_simple_mapping` 강제 덮어쓰기 제거

3. **Inter-landmark 중복 방지**
   ```cpp
   // (min(i,j), max(i,j)) 레지스트리 구현
   std::set<std::pair<int,int>> created_factors_;
   ```

4. **드리프트 보정 복구**
   - DriftCorrectionManager 재연결
   - map→odom 동적 계산 및 발행

  5. **CMake TBB 정합(필수 링크 오류 예방)**
     - `find_package(TBB REQUIRED)` 추가 후 타깃 링크를 `TBB::tbb`로 표준화
     ```cmake
     find_package(TBB REQUIRED)
     target_link_libraries(cone_slam_node TBB::tbb)
     ```

  6. **뷰어 override 컴파일 이슈 해결**
     - 선택 A(권장): `SLAMVisualizer::visualizeFactorGraph(...)`에 `virtual` 지정
     - 선택 B(대안): `SLAMVisualizerImproved`의 `override` 제거
     ```cpp
     // 선택 A 예시 (SLAMVisualizer 선언부)
     virtual void visualizeFactorGraph(const gtsam::NonlinearFactorGraph& graph,
                                       const gtsam::Values& values,
                                       const rclcpp::Time& timestamp = rclcpp::Time());
     ```

  7. **누락 헤더 보강(플랫폼별 빌드 실패 예방)**
     - `include/cone_stellation/viewer/slam_visualizer.hpp`: `#include <tf2/LinearMath/Quaternion.h>`
     - `include/cone_stellation/odometry/async_cone_odometry.hpp`: `#include <chrono>`, `#include <vector>`, `#include <Eigen/Geometry>`
     - `include/cone_stellation/mapping/data_association.hpp`: `#include <limits>`
     - `include/cone_stellation/util/drift_correction_manager.hpp`: `#include <algorithm>`
     - `include/cone_stellation/odometry/cone_odometry_2d.hpp`: `#include <rclcpp/rclcpp.hpp>`, `#include <cmath>`
     - `include/cone_stellation/common/cone.hpp`: `#include <map>`
     - `include/cone_stellation/common/tentative_landmark.hpp`: `#include <set>`, `#include <algorithm>`

  8. **map→odom 항등 루프 제거 + DriftCorrectionManager 기반 저주파 업데이트**
     - 초기 1회만 `map→odom = I` 게시
     - 이후 5–10 Hz로 드리프트 매니저 결과를 사용해 `map→odom` 업데이트(타임스탬프 일치)
     - 항등 TF를 지속 게시하는 타이머는 제거

  9. **Inter-landmark 중복 방지 레지스트리 즉시 도입(운용 규칙 포함)**
     - 키: `(min(i,j), max(i,j))`
     - 동일 쌍에 대해 프레임/시간 윈도우당 1회만 생성(중복 누적 금지)

  10. **AsyncConeOdometry 배선(기본 비활성, 옵션 제공)**
      - 파라미터 `odometry.enable_async` 추가(기본 false)
      - 활성 시: 프레임 생성 지점에서 `async_odometry_->insert_frame(frame)` 주입
      - 소비 경로: 결과는 `ConeMapping` 초기값 힌트로 선택적 사용(불안정 시 쉽게 비활성화 가능)

  11. **SLAMVisualizer 팩터 분류 버그 수정(안전 참조)**
      - 요인 분류 시 루프 변수의 주소를 저장하지 말고 인덱스만 저장 → 사용 시 `graph[index]`로 접근

  12. **ConeDistanceFactor 특이점(분모 0) 가드**
      - 거리 하한 ε 가드 추가 또는 호출부 게이팅 강화
      ```cpp
      gtsam::Vector2 diff = l2 - l1;
      double distance = diff.norm();
      if (distance < 1e-6) {
        if (H1) *H1 = gtsam::Matrix12::Zero();
        if (H2) *H2 = gtsam::Matrix12::Zero();
        return (gtsam::Vector1() << -measured_distance_).finished();
      }
      ```

**검증 기준**:
- 빌드 성공 (의존성 오류 없음)
- 파라미터 기반 설정 작동
- TF tree 일관성 유지

**검증 기준(보완)**:
- 빌드: `--cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo`로 빌드 플래그 확인, TBB 링크 성공
- TF: `map→odom`이 초기 1회 항등 후 저주파(5–10 Hz)로만 업데이트되는지 확인(충돌/루프 없음)
- Inter-landmark: 동일 쌍 요인 중복 0건(로그/그래프 카운트로 검증)
- Async 경로: `odometry.enable_async=false` 기본에서 정상 동작, true 시 프레임 주입/소비 경로가 작동하되 맵퍼 안정성 저하 없을 것

### Phase 2: 데이터 연관 고도화 (2-3주) 🟡

**목표**: 강건한 데이터 연관 및 노이즈 필터링

**개선사항**:
1. **Mahalanobis 게이팅 도입**
   ```cpp
   // 혁신 공분산을 이용한 통계적 게이팅
   double d_M2 = innovation.transpose() * S.inverse() * innovation;
   if (d_M2 > chi2_threshold) reject();
   ```

2. **트랙 ID 가중치 개선**
   - 최근 N 프레임 매칭 비율로 신뢰도 계산
   - UNKNOWN 색상에서 트랙 ID 가중치 상향

3. **텐터티브 랜드마크 강화**
   - 관측 이상치 거부 (2σ 임계값)
   - 승격 조건 엄격화: min_obs=3, time_span≥0.5s, variance≤0.2m²

### Phase 3: GLIM 아키텍처 도입 (3-4주) 🟢

**목표**: 검증된 GLIM 패턴 적용으로 안정성/성능 향상

**구현 내용**:

1. **계층적 설정 시스템**
   ```yaml
   global_config:
     preprocessing:
       max_cone_distance: 20.0
       min_cone_confidence: 0.5
     mapping:
       isam2:
         relinearize_threshold: 0.1
         relinearize_skip: 10
     loop_closure:
       descriptor_match_threshold: 0.5
   ```

2. **비동기 파이프라인**
   ```cpp
   // Producer-Consumer 패턴
   Queue<Frame> preprocessing_queue;
   Queue<Frame> mapping_queue;
   Queue<Result> result_queue;
   ```

3. **로깅/모니터링**
   - spdlog 통합 (링버퍼)
   - 성능 메트릭 대시보드
   - QoS 헬퍼 함수

### Phase 4: 루프 클로저 통합 (2-3주) 🔵

**목표**: GLIM 스타일 암묵적 루프 클로저 구현

**접근법**:
1. **암묵적 루프 검출**
   - 공간 근접성 기반 (거리 < 5m)
   - 랜드마크 중첩도 검사
   - 서브맵 개념 도입

2. **기하학적 검증**
   - RANSAC 기반 변환 추정
   - 인라이어 비율 > 60%
   - Huber 커널로 로버스트화

3. **통합 전략**
   - 별자리 디스크립터는 보조 수단으로만 사용
   - 복잡한 특징 기반 방식 회피

### Phase 5: 성능 최적화 (2-3주) ⚡

**목표**: 실시간 성능 달성 (50km/h에서 안정 작동)

**최적화 영역**:

1. **팩터 그래프 최적화**
   - 팩터 샘플링 (밀집 관측 시)
   - ISAM2 변수 재정렬
   - 선택적 로버스트 커널 적용

2. **데이터 구조 개선**
   - KD-tree 기반 최근접 탐색
   - 공간 인덱싱 (랜드마크 조회)
   - 메모리 풀 활용

3. **멀티스레딩**
   - 관측 처리와 그래프 최적화 분리
   - Lock-free 큐 구현
   - SIMD 연산 활용

### Phase 6: 프로덕션 준비 (2-3주) 🏁

**목표**: 배포 가능한 시스템 완성

**작업 내용**:
- 실제 Formula Student 데이터로 종합 테스트
- 성능 벤치마킹 및 프로파일링
- 문서화 및 사용자 가이드
- CI/CD 파이프라인 구축
- 디버깅/진단 도구 개발

## 5. 기술 사양

### 5.1 성능 목표

| 메트릭 | 목표값 | 검증 방법 |
|--------|--------|-----------|
| 제어 주기 | 100Hz | /odometry/filtered 주파수 모니터링 |
| SLAM 주기 | 10-30Hz | 처리 타임스탬프 분석 |
| 드리프트 보정 지연 | <50ms | TF 타임스탬프 차이 |
| 위치 정확도 | <0.5m/1km | Ground truth 비교 |
| 콘 위치 오차 | <0.3m RMS | 알려진 랜드마크 검증 |
| 방향 오차 | <2° RMS | IMU/GPS 헤딩 비교 |
| 최고 주행 속도 | 50km/h | 실시간 처리 안정성 |

### 5.2 모듈별 상세 설계

#### 전처리 (Preprocessing)
- 거리/신뢰도 필터링
- 라인 패턴 검출 (RANSAC/Hough 개선)
- 간단한 트래킹 (ID 할당)
- 관측 공분산 모델링

#### 데이터 연관 (Data Association)
- 1차 게이팅: 유클리드 거리 + 색상 호환
- 2차 게이팅: Mahalanobis 거리 (χ² 검정)
- 전역 매칭: Hungarian/JCBB 알고리즘
- 트랙 ID 신뢰도 가중

#### 매핑 (Mapping)
- ISAM2 증분 최적화
- 텐터티브 → 확정 랜드마크 승격
- Inter-landmark 제약 생성 (중복 방지)
- 패턴 기반 제약 (Jacobian 구현 후)

#### 루프 클로저 (Loop Closure)
- 서브맵 기반 관리
- 공간 근접성 검출
- 기하학적 일관성 검증
- 로버스트 통합

## 6. 테스트 전략

### 단위 테스트
- 팩터 Jacobian 수치 미분 검증
- 데이터 연관 정확도
- TF tree 일관성

### 통합 테스트
- Rosbag 재생 (알려진 궤적)
- 멀티레이트 타이밍 검증
- 메모리 누수 감지

### 성능 테스트
- CPU/메모리 프로파일링
- 지연 측정
- 콘 밀도별 처리 시간

## 7. 리스크 관리

### 기술적 리스크

| 리스크 | 영향도 | 완화 전략 |
|--------|--------|-----------|
| 아키텍처 마이그레이션 불안정 | 높음 | SimpleConeMapping 폴백 유지 |
| 성능 저하 | 중간 | 각 단계별 벤치마킹 |
| Jacobian 구현 오류 | 높음 | 수치 미분 검증, 점진적 활성화 |
| 루프 클로저 오검출 | 중간 | 보수적 임계값, 기하학적 검증 강화 |

### 일정 리스크
- Phase 1 지연 시 전체 일정에 영향
- 병렬 개발 가능한 부분 식별
- 최소 기능 세트(MVP) 정의

## 8. 즉시 실행 사항 (Week 1)

### Day 1-2: 긴급 수정
```bash
# 1. package.xml 수정
cd /home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation
# tf2 의존성 추가

# 2. 하드코딩 제거
# src/cone_stellation/ros/cone_slam_node.cpp:59-66

# 3. 빌드 및 테스트
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select cone_stellation
```

### Day 3-4: Inter-landmark 안정화
- 중복 방지 레지스트리 구현
- 팩터 생성 레이트 제한
- 테스트 및 검증

### Day 5: 드리프트 보정 복구
- DriftCorrectionManager 연결
- map→odom 계산 로직 구현
- TF 발행 및 검증

## 9. 성공 지표

### 기술 KPI
- 시스템 크래시: 0회 (실제 rosbag 데이터)
- 제어 루프 주파수: 100Hz 유지
- 드리프트: <0.5m/1km
- TF tree 일관성: 100%

### 개발 KPI
- 모든 설정 파라미터 작동
- 테스트 커버리지: >80%
- 문서화 완성도: API 100%, 사용자 가이드 작성
- Formula Student 팀 채택: 5개 이상

## 10. 참고 자료

- GLIM 아키텍처: `/home/user1/ROS2_Workspace/GLIM_ws/src/glim/`
- GLIM 논문: "GLIM: 3D Range-Inertial Localization and Mapping"
- 문제 분석: `glim_to_cone_stellation_review_ko.md`
- 제품 요구사항: `PRD.md`
- 디버그 기록: `debug_log.md`

---

**마지막 업데이트**: 2025-08-13
**다음 마일스톤**: Phase 1 완료 (2025-08-27)