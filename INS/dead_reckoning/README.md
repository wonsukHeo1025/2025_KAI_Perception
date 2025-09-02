# Dead Reckoning Package

IMU 데이터를 이용한 Dead Reckoning 구현 패키지입니다.

## 기능

- `/ouster/imu` 토픽에서 IMU 데이터 구독
- 각속도와 선가속도를 이용한 위치 및 자세 추정
- **IMU 캘리브레이션 도구** (바이어스 보정)
- RViz에서 다음 요소들을 시각화:
  - TF (map → base_link)
  - Path (이동 경로)
  - 원점 마커 (흰색 구)
  - 좌표축 (X: 빨강, Y: 초록, Z: 파랑)

## 빌드 방법

```bash
cd ~/ROS2_Workspace/ros2_ws
colcon build --packages-select dead_reckoning
source install/setup.zsh
```

## 실행 방법

### 1. IMU 캘리브레이션 (권장 - 먼저 실행)
```bash
ros2 run dead_reckoning imu_calibration
```

**캘리브레이션 사용법:**
1. IMU 센서를 수평한 표면에 놓으세요
2. 센서가 완전히 정지된 상태에서 캘리브레이션을 시작하세요
3. 키보드 명령:
   - `1`: 10초 캘리브레이션
   - `2`: 30초 캘리브레이션
   - `3`: 1분 캘리브레이션
   - `4`: 5분 캘리브레이션
   - `5`: 30분 캘리브레이션
   - `6`: 1시간 캘리브레이션
   - `s`: 캘리브레이션 결과 저장
   - `q`: 종료

**권장 캘리브레이션 시간:**
- 빠른 테스트: 30초 (키 `2`)
- 일반적인 사용: 1분 (키 `3`)
- 정밀한 캘리브레이션: 5분 이상 (키 `4`, `5`, `6`)

### 2. 노드만 실행
```bash
ros2 run dead_reckoning dead_reckoning_node
```

### 3. 런치 파일로 실행 (RViz 포함)
```bash
ros2 launch dead_reckoning dead_reckoning_launch.py
```

### 4. RViz 없이 실행
```bash
ros2 launch dead_reckoning dead_reckoning_launch.py use_rviz:=false
```

## 토픽

### 구독하는 토픽
- `/ouster/imu` (sensor_msgs/Imu): IMU 데이터

### 발행하는 토픽
- `/dead_reckoning/path` (nav_msgs/Path): 이동 경로
- `/dead_reckoning/markers` (visualization_msgs/MarkerArray): 원점과 좌표축 마커
- `/tf` (tf2_msgs/TFMessage): map → base_link 변환

## 알고리즘

1. **IMU 캘리브레이션**:
   - 자이로스코프 바이어스: 정지 상태에서의 평균 각속도
   - 가속도계 바이어스: 측정값에서 이론적 중력값을 뺀 값
   - 중력 방향 자동 감지 및 크기 측정

2. **자세 추정**: 각속도를 이용한 쿼터니언 기반 자세 업데이트

3. **위치 추정**: 
   - 바이어스 보정된 IMU 데이터 사용
   - 캘리브레이션된 중력 크기로 중력 보정
   - 가속도를 월드 좌표계로 변환
   - 이중 적분을 통한 위치 계산

## 캘리브레이션 파일

캘리브레이션 결과는 `config/` 디렉토리에 JSON 형식으로 저장됩니다:
- 파일명: `imu_calibration_YYYYMMDD_HHMMSS.json`
- Dead reckoning 노드는 자동으로 최신 캘리브레이션 파일을 로드합니다

**캘리브레이션 데이터 구조:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "collection_duration": 60,
  "sample_count": 6000,
  "accel_bias": {"x": 0.1, "y": -0.05, "z": 0.2},
  "gyro_bias": {"x": 0.001, "y": -0.002, "z": 0.0005},
  "gravity_magnitude": 9.81
}
```

## 주의사항

- **캘리브레이션 필수**: 정확한 Dead Reckoning을 위해 먼저 IMU 캘리브레이션을 수행하세요
- **정지 상태 캘리브레이션**: 캘리브레이션 중에는 IMU를 절대 움직이지 마세요
- **수평 배치**: IMU를 수평한 표면에 놓고 캘리브레이션하세요
- **오차 누적**: IMU의 특성상 시간이 지날수록 오차가 누적됩니다
- **추가 센서 융합**: 실제 사용 시에는 GPS, 비전 등 추가 센서와의 융합이 권장됩니다

## 의존성

- rclpy
- sensor_msgs
- geometry_msgs
- nav_msgs
- tf2_ros
- tf2_geometry_msgs
- visualization_msgs
- numpy
- scipy 