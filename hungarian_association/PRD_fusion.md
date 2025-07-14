# PRD: Hungarian Association 패키지 개선 전략

## 개요
이 문서는 Formula Student 자율주행 차량용 YOLO-LiDAR 멀티카메라 퓨전 시스템과 칼만 필터링 기반 트래킹 시스템의 체계적인 개선 전략 및 개발 현황을 담고 있습니다.

**최종 업데이트**: 2025-07-01

## 시스템 사양
- **LiDAR**: Ouster OS1-32ch (20Hz)
- **카메라**: Logitech C922 + 저가형 웹캠 (640x360 @ 30fps)
- **IMU**: Ouster 내장 6축 IMU (100Hz)
- **차량 속도**: 최대 50-60km/h
- **검출 범위**: 10m (필수), 20m (목표)

## 주요 문제점
1. **칼만 필터 모션 모델 버그**: 센서 뒤로 넘어간 콘이 센서와 함께 전진하는 문제
2. **색상 오분류**: 커브 구간에서 2D 투영으로 인한 매칭 오류
3. **고속 주행 시 강건성**: 50km/h 주행 시 예측 정확도 저하

## 개선 우선순위 및 로드맵

### Phase 0: 긴급 수정 (1주) 🚨
- 칼만 필터 모션 모델 버그 수정
- 센서 뒤 콘 처리 로직 수정
- 디버깅 시각화 도구 구현

### Phase 1: 핵심 개선 (2주)
- 헝가리안 알고리즘 기반 데이터 연관
- 적응적 노이즈 모델
- 깊이 기반 매칭 우선순위

### Phase 2: 시각화 및 모니터링 (1주)
- RViz 예측 화살표 구현
- 불확실성 시각화
- 성능 모니터링 대시보드

### Phase 3: 최적화 및 고급 기능 (2주)
- 코드 벡터화 및 병렬화
- FOV 기반 사전 필터링
- 색상 분류 강건성 향상

---

## 상세 개선 사항

### 1. yolo_lidar_multicam_fusion.py 개선

#### 1.1 성능 최적화

##### 1.1.1 비용 행렬 계산 벡터화
**현재 코드 (line 250-254):**
```python
for i in range(num_boxes):
    box_center_x = yolo_bboxes[i, 0]
    box_center_y = yolo_bboxes[i, 1]
    distances = np.linalg.norm(cone_image_points - [box_center_x, box_center_y], axis=1)
    cost_matrix[i, :] = np.where(distances < self.max_matching_distance, distances, self.max_matching_distance + 1.0)
```

**개선안:**
```python
# 브로드캐스팅을 활용한 벡터화
box_centers = yolo_bboxes[:, :2]  # (P, 2)
# (P, 1, 2) - (1, M, 2) = (P, M, 2)
diff = box_centers[:, np.newaxis, :] - cone_image_points[np.newaxis, :, :]
distances = np.linalg.norm(diff, axis=2)
cost_matrix = np.where(distances < self.max_matching_distance, 
                      distances, 
                      self.max_matching_distance + 1.0)
```

##### 1.1.2 카메라 FOV 기반 사전 필터링
**구현 계획:**
- 각 카메라의 FOV 파라미터 추가
- 투영 전 3D 포인트의 방향 벡터 계산
- FOV 밖의 포인트 사전 제거

```python
def filter_points_by_fov(self, points_sensor, T_sensor_to_cam, h_fov, v_fov):
    """카메라 FOV 내의 포인트만 필터링"""
    # 구현 예정
```

##### 1.1.3 메모리 풀링
- 자주 사용되는 배열을 미리 할당
- 콜백마다 재사용

#### 1.2 알고리즘 개선

##### 1.2.1 적응적 매칭 임계값
**구현 계획:**
```python
def adaptive_matching_threshold(self, depth):
    """깊이에 따른 적응적 매칭 임계값 계산"""
    # 가까운 거리: 엄격한 임계값
    # 먼 거리: 완화된 임계값
    base_threshold = self.max_matching_distance
    depth_factor = np.clip(depth / 20.0, 0.5, 2.0)  # 20m 기준
    return base_threshold * depth_factor
```

##### 1.2.2 카메라 간 중복 처리 개선
- 신뢰도 점수 도입
- 가장 좋은 뷰의 매칭 선택

#### 1.3 캘리브레이션 검증
- 변환 행렬의 직교성 확인
- 재투영 오차 계산
- 실시간 캘리브레이션 품질 모니터링

---

### 2. kalman_filtering.py 개선

#### 2.0 🚨 긴급: 모션 모델 버그 수정

##### 현재 문제 분석
**문제가 있는 코드 (line 98-104):**
```python
# 현재 구현 - 버그가 있는 것으로 의심됨
vel_kplus1_in_k = current_vel_sensor + accel_sensor * dt * -1
predicted_vel_sensor = R_compensation @ vel_kplus1_in_k
delta_pos_sensor_in_k = current_vel_sensor * dt + 0.5 * accel_sensor * dt**2
predicted_pos_cone = R_compensation @ (current_pos_cone - delta_pos_sensor_in_k)
```

**문제점:**
1. 가속도에 -1을 곱하는 이유가 불명확
2. 센서가 뒤로 넘어간 콘을 처리하는 로직 부재
3. 좌표계 변환이 일관성 없음

##### 수정안
```python
@staticmethod
def static_fx(x, dt, fx_args=None):
    """
    수정된 상태 전이 함수
    - 센서 중심 좌표계에서 콘의 상대 운동을 올바르게 모델링
    - 센서 뒤로 넘어간 콘도 올바르게 처리
    """
    if fx_args is None:
        return x
    
    R_imu_to_sensor, omega_imu_vec, accel_imu_vec = fx_args
    current_pos_cone = x[0:3]  # 센서 프레임에서의 콘 위치
    current_vel_sensor = x[3:6]  # 센서의 속도 (센서 프레임)
    
    # 1. IMU 데이터를 센서 프레임으로 변환
    omega_sensor = R_imu_to_sensor @ omega_imu_vec
    accel_sensor = R_imu_to_sensor @ accel_imu_vec
    
    # 2. 센서의 회전으로 인한 변환 계산
    rotation_vector = omega_sensor * dt
    if np.linalg.norm(rotation_vector) > 1e-6:
        R_delta = Rotation.from_rotvec(rotation_vector).as_matrix()
    else:
        R_delta = np.eye(3)
    
    # 3. 센서 속도 업데이트 (센서 프레임에서)
    # 센서가 가속하면, 센서 프레임에서 본 센서의 속도가 변함
    predicted_vel_sensor = current_vel_sensor + accel_sensor * dt
    
    # 4. 콘 위치 업데이트
    # 센서가 움직이면 콘은 반대 방향으로 상대 이동
    relative_motion = current_vel_sensor * dt + 0.5 * accel_sensor * dt**2
    
    # 회전 전 위치 (센서가 이동한 만큼 콘은 반대로)
    pos_before_rotation = current_pos_cone - relative_motion
    
    # 회전 적용 (센서가 회전하면 콘도 반대로 회전)
    predicted_pos_cone = R_delta.T @ pos_before_rotation
    
    # 5. 센서 속도도 회전 적용
    predicted_vel_sensor = R_delta.T @ predicted_vel_sensor
    
    return np.concatenate((predicted_pos_cone, predicted_vel_sensor))
```

##### 추가 개선사항
1. **센서 뒤 콘 처리**
```python
def is_cone_behind_sensor(self, cone_position):
    """콘이 센서 뒤에 있는지 확인"""
    # x축이 전방이라고 가정
    return cone_position[0] < -0.5  # 0.5m 마진

def handle_behind_cone(self, track):
    """센서 뒤로 넘어간 콘 처리"""
    if self.is_cone_behind_sensor(track.get_predicted_position_xyz()):
        # 트랙 품질 저하
        track.quality_score *= 0.9
        # 더 큰 프로세스 노이즈 적용
        track.ukf.Q *= 1.5
```

2. **디버깅 시각화**
```python
def publish_debug_info(self, track):
    """디버깅을 위한 추가 정보 발행"""
    debug_msg = DebugInfo()
    debug_msg.track_id = track.track_id
    debug_msg.position = track.get_predicted_position_xyz()
    debug_msg.velocity = track.ukf.x[3:6]
    debug_msg.is_behind = self.is_cone_behind_sensor(debug_msg.position)
    debug_msg.quality = track.quality_score
    self.debug_publisher.publish(debug_msg)
```

#### 2.1 데이터 연관 알고리즘 업그레이드

##### 2.1.1 헝가리안 알고리즘 적용
**현재: 그리디 매칭 (line 445-463)**
**개선안:**
```python
from scipy.optimize import linear_sum_assignment

def hungarian_data_association(self, detections, tracks):
    """헝가리안 알고리즘을 사용한 전역 최적 매칭"""
    # 비용 행렬 계산
    cost_matrix = self.compute_association_cost_matrix(detections, tracks)
    
    # 게이팅 적용
    cost_matrix = self.apply_gating(cost_matrix)
    
    # 헝가리안 알고리즘
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 유효한 매칭만 필터링
    valid_matches = []
    for det_idx, track_idx in zip(row_indices, col_indices):
        if cost_matrix[det_idx, track_idx] < self.gating_threshold:
            valid_matches.append((det_idx, track_idx))
    
    return valid_matches
```

##### 2.1.2 게이팅 로직 구현
```python
def apply_gating(self, cost_matrix):
    """마할라노비스 거리 기반 게이팅"""
    # 각 트랙의 불확실성을 고려한 게이팅
    # 구현 예정
```

#### 2.2 적응적 프로세스 노이즈

##### 2.2.1 IMU 기반 동적 Q 행렬
```python
def compute_adaptive_Q(self, imu_data, dt):
    """IMU 데이터 품질과 운동 상태에 따른 적응적 Q"""
    # 가속도 크기에 따른 조정
    accel_magnitude = np.linalg.norm(imu_data.linear_acceleration)
    
    # 각속도에 따른 조정
    angular_rate = np.linalg.norm(imu_data.angular_velocity)
    
    # 기본 Q에 스케일 팩터 적용
    motion_factor = 1.0 + 0.1 * accel_magnitude + 0.05 * angular_rate
    Q_adaptive = self.Q_base * motion_factor
    
    return Q_adaptive
```

##### 2.2.2 거리 기반 측정 노이즈
```python
def compute_adaptive_R(self, measurement_range):
    """측정 거리에 따른 적응적 R"""
    # 거리가 멀수록 노이즈 증가
    range_factor = 1.0 + (measurement_range / 50.0) ** 2
    return self.R_base * range_factor
```

#### 2.3 트랙 관리 개선

##### 2.3.1 N-스캔 초기화 로직
```python
class TentativeTrack:
    """확정 전 임시 트랙"""
    def __init__(self, initial_detection):
        self.detections = [initial_detection]
        self.confirmation_threshold = 3  # 3번 연속 검출 시 확정
        
    def add_detection(self, detection):
        self.detections.append(detection)
        return len(self.detections) >= self.confirmation_threshold
```

##### 2.3.2 트랙 품질 점수
```python
def compute_track_quality(self, track):
    """트랙의 품질/신뢰도 점수 계산"""
    factors = {
        'age': min(track.age / 100.0, 1.0),  # 오래된 트랙일수록 신뢰
        'consistency': 1.0 - (track.missed_detections / 10.0),
        'uncertainty': 1.0 / (1.0 + np.trace(track.ukf.P)),
        'color_confidence': track.get_color_confidence()
    }
    
    weights = {'age': 0.3, 'consistency': 0.3, 
               'uncertainty': 0.2, 'color_confidence': 0.2}
    
    quality = sum(factors[k] * weights[k] for k in factors)
    return np.clip(quality, 0.0, 1.0)
```

---

### 3. 시각화 개선 (visualize_fused_cones_rviz_marker.py)

#### 3.1 예측 화살표 구현

##### 3.1.1 속도 벡터 시각화
```python
def create_velocity_arrow(self, cone_position, velocity, track_id):
    """칼만 필터의 속도 예측을 화살표로 시각화"""
    arrow_marker = Marker()
    arrow_marker.header.frame_id = "os_sensor"
    arrow_marker.header.stamp = self.get_clock().now().to_msg()
    arrow_marker.ns = f"velocity_arrows"
    arrow_marker.id = track_id + 1000  # 콘 마커와 ID 충돌 방지
    arrow_marker.type = Marker.ARROW
    arrow_marker.action = Marker.ADD
    
    # 시작점: 콘 위치
    start_point = Point()
    start_point.x = cone_position[0]
    start_point.y = cone_position[1]
    start_point.z = cone_position[2]
    
    # 끝점: 예측 위치 (1초 후)
    prediction_time = 1.0  # 초
    end_point = Point()
    end_point.x = cone_position[0] + velocity[0] * prediction_time
    end_point.y = cone_position[1] + velocity[1] * prediction_time
    end_point.z = cone_position[2] + velocity[2] * prediction_time
    
    arrow_marker.points = [start_point, end_point]
    
    # 화살표 크기 (속도에 비례)
    speed = np.linalg.norm(velocity)
    arrow_marker.scale.x = 0.05 * (1.0 + speed)  # 샤프트 직경
    arrow_marker.scale.y = 0.08 * (1.0 + speed)  # 헤드 직경
    arrow_marker.scale.z = 0.1   # 헤드 길이
    
    # 색상 (속도에 따라 변화)
    arrow_marker.color.a = 0.8
    if speed < 1.0:
        # 느림: 녹색
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
    elif speed < 3.0:
        # 중간: 노란색
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
    else:
        # 빠름: 빨간색
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 0.0
    
    return arrow_marker
```

##### 3.1.2 불확실성 시각화
```python
def create_uncertainty_ellipse(self, cone_position, covariance, track_id):
    """위치 불확실성을 타원으로 시각화"""
    # 공분산 행렬에서 타원 파라미터 계산
    # 구현 예정
```

#### 3.2 추가 시각화 요소
- 트랙 ID 텍스트 마커
- 트랙 품질 표시 (투명도로 표현)
- 트랙 히스토리 궤적

---

### 4. 성능 모니터링 및 디버깅

#### 4.1 처리 시간 측정
```python
class PerformanceMonitor:
    def __init__(self, node):
        self.node = node
        self.timers = {}
        
    def start_timer(self, name):
        self.timers[name] = self.node.get_clock().now()
        
    def end_timer(self, name):
        if name in self.timers:
            elapsed = (self.node.get_clock().now() - self.timers[name]).nanoseconds / 1e6
            self.node.get_logger().debug(f"{name}: {elapsed:.2f} ms")
            return elapsed
```

#### 4.2 품질 메트릭
- 매칭 성공률
- 평균 재투영 오차
- 트랙 수명 통계
- 색상 분류 정확도

---

## 구현 순서 및 일정

### Week 1: 긴급 수정 🚨
1. 칼만 필터 모션 모델 버그 수정
   - [x] static_fx 함수 재구현 - **2D로 단순화 완료**
   - [x] 센서 뒤 콘 처리 로직 추가 - **잠재적 관찰 중**
   - [ ] 단위 테스트 작성
2. 디버깅 도구 구현
   - [x] 트랙 상태 시각화 - **속도 화살표 구현 완료**
   - [ ] IMU 데이터 로깅
   - [ ] 모션 예측 검증 도구

### Week 2-3: 핵심 개선
1. 데이터 연관 개선
   - [ ] 헝가리안 알고리즘 구현
   - [ ] 마할라노비스 거리 기반 게이팅
   - [ ] 적응적 노이즈 모델
2. 퓨전 로직 개선
   - [ ] 깊이 기반 매칭 우선순위
   - [ ] 비용 행렬 계산 벡터화

### Week 4: 시각화 및 모니터링
1. RViz 시각화 개선
   - [x] 속도 벡터 화살표 - **완료**
   - [ ] 불확실성 타원
   - [ ] 트랙 품질 표시
2. 성능 모니터링
   - [ ] 처리 시간 측정
   - [ ] 품질 메트릭 대시보드

### Week 5-6: 최적화 및 안정화
1. 성능 최적화
   - [ ] FOV 기반 사전 필터링
   - [ ] 메모리 풀링
   - [ ] 병렬 처리
2. 강건성 향상
   - [ ] N-스캔 초기화
   - [ ] 색상 분류 개선
   - [ ] 에러 복구 메커니즘

---

## 테스트 및 검증 계획

### 단위 테스트
- 각 개선 사항에 대한 개별 테스트
- 성능 벤치마크

### 통합 테스트
- 전체 파이프라인 테스트
- 실제 데이터셋 활용

### 성능 평가
- 처리 속도 측정
- 정확도 평가 (ground truth 대비)
- 메모리 사용량 모니터링

---

## 위험 요소 및 대응 방안

### 기술적 위험
1. **실시간 성능 저하**
   - 대응: 단계별 최적화, 필요시 C++ 포팅 고려

2. **캘리브레이션 오차**
   - 대응: 온라인 캘리브레이션 검증, 자동 보정

3. **센서 동기화 문제**
   - 대응: 타임스탬프 검증 강화, 예측 기반 보상

### 프로젝트 위험
1. **요구사항 변경**
   - 대응: 모듈화된 설계로 유연성 확보

2. **통합 복잡도**
   - 대응: 점진적 통합, 롤백 계획 수립

---

## 시스템 정보 (업데이트됨)

### 하드웨어 사양
- [x] LiDAR: Ouster OS1-32ch (20Hz)
- [x] 카메라: Logitech C922 + 저가형 웹캠 (640x360 @ 30fps)
- [x] IMU: Ouster 내장 6축 IMU (100Hz)
- [ ] 컴퓨팅 플랫폼 사양
- [ ] ROS2 버전

### 운영 환경
- [x] 차량 속도: 최대 50-60km/h
- [x] 환경: 주차장(테스트), 비행장(실제)
- [x] 검출 범위: 10m(필수), 20m(목표)
- [ ] 날씨/조명 조건

### 성능 요구사항
- [x] 목표 FPS: 20Hz
- [x] 현재 성능: 18-19.5Hz
- [x] 동시 추적: 30-40개 콘
- [ ] 정확도 목표 수치

### 주요 문제점
- [x] 칼만 필터 모션 모델 버그
- [x] 고속 주행 시 강건성
- [x] 색상 오분류 (커브 구간)

---

## 개발 현황 (2025-07-01)

### 완료된 작업

#### 1. 칼만 필터 2D 단순화 ✅
- **변경 내용**:
  - 상태 벡터: 6D [x,y,z,vx,vy,vz] → 4D [x,y,vx,vy]
  - 측정 벡터: 3D → 2D
  - Z값은 필터링 없이 측정값 직접 사용
  - 데이터 연관: 2D XY 평면 거리 사용
- **효과**:
  - 계산량 감소 (시그마 포인트 13개 → 9개)
  - Z축 노이즈가 XY 추정에 영향 없음
  - 평평한 환경에 최적화

#### 2. 시각화 개선 ✅
- **속도 예측 화살표**:
  - 각 트랙의 속도를 1초 후 예측 위치로 표시
  - 속도에 따른 색상 변화 (녹색/노란색/빨간색)
  - 0.1 m/s 이상일 때만 표시 (노이즈 필터링)
- **라이다 원본 시각화 제거**:
  - 불필요한 회색 원통 마커 제거
  - 깔끔한 시각화

### 진행 중인 관찰 사항

#### 센서 뒤 콘 처리
- **현재 상태**: 잠재적 이슈로 관찰 중
- **증상**: 센서 관측 범위를 벗어난 콘의 거동
- **관찰 결과**: 현재는 정상 작동하는 것으로 보임
- **계획**: 명확한 오류 발생 시 수정 예정

### 미완료 작업

#### 1. 데이터 연관 개선
- 현재 그리디 매칭 → 헝가리안 알고리즘 필요
- 마할라노비스 거리 기반 게이팅 미구현

#### 2. 적응적 파라미터
- 고정 노이즈 모델 사용 중
- IMU 기반 동적 Q 행렬 미구현
- 거리 기반 측정 노이즈 미구현

#### 3. 퓨전 로직 최적화
- 비용 행렬 계산 벡터화 필요
- FOV 기반 사전 필터링 미구현

---

## 다음 단계

1. 센서 뒤 콘 처리 지속 관찰
2. 헝가리안 알고리즘 구현 (데이터 연관)
3. 적응적 노이즈 모델 구현
4. 성능 최적화

이 문서는 지속적으로 업데이트될 예정입니다.