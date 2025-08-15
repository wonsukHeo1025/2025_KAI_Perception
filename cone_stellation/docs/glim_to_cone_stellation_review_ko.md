## 목적
- GLIM과 cone_stellation을 비교·분석하여, cone_stellation에 적용 가능한 구조적 개선, 에러 수정, 품질 향상 전략을 제시합니다.
- 설계 철학, 아키텍처, 모듈 경계, 스레딩/실행 모델, 데이터/TF 흐름, 팩터그래프 모델링, 빌드/패키징 관점의 제안을 포함합니다.

---

## 패키지 전반 분석: GLIM (glim, glim_ros2, glim_ext)
- 구성
  - **glim**: SLAM 핵심(전처리, 오도메트리, 서브/글로벌 매핑, 공용 유틸/로깅/설정)
  - **glim_ros2**: ROS2 노드(`GlimROS`)로 센서 구독, 설정 로딩, 비동기 파이프 orchestration, 결과 publish
  - **glim_ext**: 확장 모듈(루프검출, GNSS, IMU validator 등)을 so 플러그인으로 동적 로드
- 주요 특성
  - **비동기 파이프라인**: IMU/Points/Image → Preprocess → AsyncOdometry → AsyncSubMapping → AsyncGlobalMapping
  - **설정 계층화**: `GlobalConfig` 하위에 odometry/sub/global/sensors/ros 등 모듈별 conf
  - **로깅/가시화**: spdlog 링버퍼, 타이머 기반 결과 수집/배출, QoS 헬퍼, 디버그 토픽 일관성
  - **팩터그래프**: GTSAM + gtsam_points(GPU) 훅, ISAM2 파라미터 조정, 배치 업데이트/검증 로직
- 코드 인용
```1:40:/home/user1/ROS2_Workspace/GLIM_ws/src/glim_ros2/src/glim_ros/glim_ros.cpp
GlimROS::GlimROS(...) {
  // config_path, GlobalConfig, 모듈 동적 로딩, QoS/구독 생성, 1ms 타이머, 결과 파이프 연결
}
```

---

## 패키지 전반 분석: ConeSTELLATION (cone_stellation)
- 디렉터리
  - `include/cone_stellation/...`: 전처리, 오도메트리(비동기 래퍼), 매핑(Simple/Full), 데이터연관, 팩터, 뷰어, 유틸
  - `src/...`: 전처리 구현, 팩터 일부 구현, 루프클로저 WIP
  - `scripts/`: 시뮬레이터/퍼블리셔/테스트 스크립트(토픽 체크, SLAM-only 등)
  - `launch/`: 단일 노드/테스트 런치
  - `docs/`: 설계/이슈/PRD/토픽 명세 등
- 노드 흐름
  - `ConeSLAMNode`: 파라미터 로딩 → 전처리 생성 → AsyncConeOdometry 시작(현재 프레임 주입 없음) → Simple 또는 Full Mapping 선택 → 구독(cone, odom) → 시각화 타이머 → TF(map→odom 항등, map→base_link_slam)
- 매핑
  - `SimpleConeMapping`: 보수적 ISAM2, 최근접+색/트랙ID 연관, 최대 3개 신규 랜드마크/프레임, 첫 포즈/몇 랜드마크에 Prior
  - `ConeMapping`: 텐터티브→승격, 관측/오도메트리 팩터 + 인터-랜드마크 거리/패턴 팩터, 주기 최적화
- 팩터
  - `ConeObservationFactor`(정상 야코비안), `ConeDistanceFactor`(정상), `ConeLine/Angle/Parallel`(잔차만, Jacobian TODO)

코드 인용
```24:76:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/src/cone_stellation/ros/cone_slam_node.cpp
if (use_simple_mapping) simple_mapping_...; else mapping_...;
```

---

## GLIM에서 배울 수 있는 핵심 철학/아키텍처 포인트
- **모듈화와 동적 확장**: 오도메트리/로컬/글로벌 모듈 경계를 명확히, so 플러그인 전략 수용 여지
- **일관된 설정 관리**: 공통 Config 계층을 한 군데에서 선언/주입, 런타임 재로딩 훅
- **비동기 파이프라인**: 센서→전처리→오도메트리→(서브)→글로벌로 스레딩, 타이머는 수집/서비스만
- **안전한 ISAM2 업데이트**: 팩터/값 배치, 재선형화 스킵/임계 조정, 캐시/분해법 선택으로 수치 안정성 확보

---

## 발견된 이슈와 수정 제안
- 설정 무시(하드코딩)
  - 현상: `mapping.use_simple_mapping` 강제 false
  - 조치: 강제 덮어쓰기 제거 → 파라미터 신뢰
```59:66:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/src/cone_stellation/ros/cone_slam_node.cpp
// FORCE USE OF CONEMAPPING ... (삭제 권장)
```
- 팩터 Jacobian 미구현
  - 영향: 정보행렬 왜곡/발산 위험 → 구현 전 비활성 유지
- 인터-랜드마크 거리 팩터 중복 삽입
  - 조치: `(min(i,j),max(i,j))` 세트로 중복 방지
- 비활성 컴포넌트(AsyncConeOdometry 주입 없음)
  - 조치: 실제 사용 배선 또는 제거
- 데이터 연관에서 Mahalanobis 게이팅 미사용
  - 조치: $d_M^2 \le \chi^2$ 기반 게이팅 도입
- 빌드/패키징 불일치
  - `package.xml`에 `tf2`, `tf2_eigen`, `tf2_geometry_msgs` 추가, `tbb` 경로 정리
- 초기 랜드마크 전략
  - 조치: 텐터티브 승격 정책 보강(관찰수/시간/분산/색신뢰도)
- TF/프레임
  - 조치: 센서-기체 정렬 파라미터화, TF 실패 백오프/진단, map→odom 항등 게시 조건화

---

## 비교: GLIM ↔ ConeSTELLATION 적용 포인트
- **설정 체계**: GlobalConfig 유사 계층 도입 → 파라미터 일원화/테스트 변이 용이
- **스레딩**: 오도메트리·매핑 스레드 분리, producer-consumer 큐/백프레셔
- **로깅**: spdlog 기본 로거/링버퍼 싱크, 통일된 throttle 매크로
- **팩터관리**: 앵커 최소화, 중복·불완전 팩터 비활성, robust loss 선택적 적용

---

## 성능·복잡도/메모리 분석(근사)
- 포즈 수 $N$, 랜드마크 수 $M$, 관측 수 $K$.
  - 팩터 수: $\mathcal{O}(N)$(오도메트리) + $\mathcal{O}(K)$(관측) + 선택 팩터
  - iSAM2 업데이트: 희소 정상방정식 QR → 평균 $\mathcal{O}(n^{1.5}\text{~}2)$, 그래프 구조/키패턴에 의존
  - 메모리: 값(포즈2, 포인트2) $\mathcal{O}(N+M)$, 팩터/클릭 트리 $\mathcal{O}(K)$
- 최적화 빈도: `optimize_every_n_frames` 증가 시 CPU↓, 지연/추정 품질 트레이드오프

---

## QoS/토픽 설계 가이드
- 센서(Detection/IMU): SensorDataQoS + 깊은 큐(드롭 방지), BestEffort 가능
- 추정/시각화: Depth 10~100, BestEffort, Volatile
- TF: `map→base_link_slam`는 SLAM만, `odom→base_link`는 EKF만 게시(충돌 금지)

---

## 단계별 적용 계획(안)
1) 안정화: 하드코딩 제거, Jacobian 미구현 팩터 비활성, 중복 방지, `package.xml` 보강, `tbb` 정리
2) 연관 고도화: Mahalanobis 게이팅, 트랙ID 신뢰도/색 가중, outlier robust loss
3) 오도메트리 경로: AsyncConeOdometry 실제 배선 또는 제거, 결과를 odom/키프레임 초기값으로 활용
4) 구조 개선: (옵션) 로컬/글로벌 맵 분리, 설정/로깅/QoS 체계 정비
5) 고급 팩터: Jacobian 구현 완료 후 점진 활성화 + A/B 실험

---

## 테스트/검증 체크리스트
- 토픽/TF: 센서→기체 TF 정상, map/odom 트리 충돌 없음
- 그래프 품질: 팩터 수/중복 0, relinearize 통계, 최적화 시간
- 정확도: ATE/RPE, 랜드마크 반복성/재식별률
- 회귀: 시뮬/리얼 로그 재생, 파라미터 스윕

---

## 추가 코드 인용
- 데이터 연관: 최근접+색+트랙 ID 스코어
```54:91:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/include/cone_stellation/mapping/data_association.hpp
for (const auto& [landmark_id, landmark] : landmarks) {
  // 색 제약, 거리 게이팅, 트랙 ID 보너스
}
```
- 거리 팩터 Jacobian(정상 구현 사례)
```24:43:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/include/cone_stellation/factors/inter_landmark_factors.hpp
gtsam::Vector evaluateError(...){
  // residual = ||l2-l1|| - d, H1/H2 = ±diff^T/||diff||
}
```

---

## 수학적·원리적 부록

### 1) 좌표계/TF와 관측 모델 정식화
- 좌표계: `map`(W), `base_link`(B), `sensor`(S). 2D 가정으로 포즈는 Pose2.
- 포즈 변수: $x_k = (p_k,\, \theta_k) \in SE(2)$, $p_k \in \mathbb{R}^2$, $\theta_k \in \mathbb{R}$
- 랜드마크: $\ell_j \in \mathbb{R}^2$
- 관측 모델(차량 프레임):
  $$ z^{pred}_{k,j} = R(\theta_k)^\top (\ell_j - p_k) \in \mathbb{R}^2 $$
  $$ r = z^{pred}_{k,j} - z_{k,j} $$
- 회전 행렬:
  $$ R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix} $$
- 구현 일치: 차량 프레임 측정치를 그래프에 직접 사용
```36:49:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/include/cone_stellation/factors/cone_observation_factor.hpp
predicted = pose.transformTo(landmark); error = predicted - measured_
```

### 2) ConeObservationFactor의 Jacobian 유도
- $$ r(p,\theta,\ell) = R(\theta)^\top (\ell - p) - z $$
- 변수별:
  $$ \frac{\partial r}{\partial \ell} = R(\theta)^\top, \qquad \frac{\partial r}{\partial p} = -R(\theta)^\top $$
  $$ \frac{\partial r}{\partial \theta} = \frac{\partial R(\theta)^\top}{\partial \theta}(\ell - p) = -J\, R(\theta)^\top (\ell - p),\; J=\begin{bmatrix}0&-1\\1&0\end{bmatrix} $$

### 3) Odometry BetweenFactor(2D)
- $$ \Delta p = R(\theta_{k-1})^\top (p_k - p_{k-1}),\quad \Delta \theta = \mathrm{wrap}(\theta_k - \theta_{k-1}) $$
- $$ r = [\Delta p - \hat{\Delta p},\; \Delta\theta - \hat{\Delta\theta}] $$
```218:247:/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/include/cone_stellation/mapping/cone_mapping.hpp
T_prev_current → (dx,dy,dtheta) → BetweenFactor<Pose2>
```

### 4) Inter-landmark Distance Factor 수식과 Jacobian
- $$ r = \|\ell_j - \ell_i\| - d_{ij} $$
- $$ \frac{\partial r}{\partial \ell_i} = -\frac{(\ell_j-\ell_i)^\top}{\|\ell_j-\ell_i\|},\quad \frac{\partial r}{\partial \ell_j} = +\frac{(\ell_j-\ell_i)^\top}{\|\ell_j-\ell_i\|} $$
- 특이 케이스: $\|\ell_j-\ell_i\| \to 0$ 회피(최소 거리 게이팅)

### 5) Line/Angle/Parallel 팩터 기하와 안정화
- Line(공선성):
  $$ r = \frac{(\ell_2-\ell_1) \times (\ell_3-\ell_1)}{\|\ell_2-\ell_1\|\,\|\ell_3-\ell_1\|} \to 0 $$
- Angle:
  $$ r = \arccos\!\left( \frac{(\ell_1-\ell_2)\cdot(\ell_3-\ell_2)}{\|\ell_1-\ell_2\|\,\|\ell_3-\ell_2\|} \right) - \hat{\alpha} $$
- Parallel:
  $$ r = \frac{(\ell_2-\ell_1) \times (\ell_4-\ell_3)}{\|\ell_2-\ell_1\|\,\|\ell_4-\ell_3\|} \to 0 $$
- 공통: 분모 보호, robust loss, Jacobian 정식 구현 필요

### 6) 노이즈 모델링/튜닝
- 관측(2D): $R_z = \mathrm{diag}(\sigma_x^2,\,\sigma_y^2)$
- 오도메트리: $R_{odom} = \mathrm{diag}(\sigma_{dx}^2,\,\sigma_{dy}^2,\,\sigma_{d\theta}^2)$
- 거리: $\sigma_d$; 과신 방지 위해 0.05~0.2m 스윕 권장
- 일부 팩터에 Huber/Tukey 적용 고려

### 7) 데이터 연관: Mahalanobis 게이팅
- 혁신: $\nu = z - h(\hat{x},\hat{\ell})$
- 혁신 공분산: $S = HPH^\top + R$
- 마할라노비스: $d_M^2 = \nu^\top S^{-1} \nu$
- 수용 기준: $d_M^2 \le \chi^2_{\alpha, df=2}$ (예: $\alpha=0.95 \Rightarrow 5.99$)

### 8) 키프레임/가시성 원리
- 변위 임계: $\|\Delta p\| > T_{trans}$ 또는 $|\Delta\theta| > T_{rot}$
- 가시성: 최소 콘 수·시차 기준 충족 시 키프레임

### 9) iSAM2 선형화/재선형화/안정성
- 근사: $r(x) \approx r(x_0) + J(x-x_0)$, $J^T W J \Delta = -J^TWr$
- 파라미터: `relinearizeThreshold/Skip`로 속도↔정확도 조절, QR 분해 선택

### 10) 텐터티브 승격 기준(통계)
- 관찰 수 $\ge N_{min}$, 시간 스팬 $\ge T_{min}$, 위치 분산 $\le \sigma_{max}^2$, 색 신뢰도 $\ge c_{min}$

### 11) 변환/프레임
- 센서→기체: $z_B = T_{B\,S} z_S$, 2D 투영 후 그래프 사용
- TF 실패 시 항등 대체는 임시책(바이어스 유발) → 캘리브레이션 반영 필요

### 12) 관측가능성
- 직진 위주 주행은 yaw/측방 구속 약함 → 키프레임/패턴 팩터로 보강

---

## 요약
- GLIM의 모듈화/비동기/설정·로깅 철학을 이식하면 안정성과 유지보수성이 향상됩니다.
- 즉시 수정: 하드코딩 제거, 미완 팩터 비활성, 중복 방지, `package.xml` 보강, `tbb` 정리, 오도메트리 경로 확정.
- 이후: 연관 고도화 → 구조 분리 → 고급 팩터 단계적 활성화 권장.
