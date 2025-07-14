# Robot Localization을 이용한 Ouster OS1 IMU 및 RTK GPS 통합 계획

## 1. 현재 상황 분석

### 사용 가능한 센서
- **Ouster OS1 내장 6축 IMU**
  - 토픽: `/ouster/imu`
  - 메시지 타입: `sensor_msgs/msg/Imu`
  - QoS: BEST_EFFORT (중요!)
  - 데이터: 3축 가속도, 3축 각속도, 방향(quaternion)
  - 녹화된 데이터: 2시간 정지 상태 bag 파일

- **Ublox ZED-F9P RTK GPS** (예정)
  - 토픽: `/ublox_gps_node/fix` (예상)
  - 메시지 타입: `sensor_msgs/msg/NavSatFix`
  - RTK 상태 정보 포함
  - 녹화 예정: 모레

### 기존 구현 분석
1. **gps_imu_fusion 패키지**
   - 자체 EKF 구현과 robot_localization 설정 모두 포함
   - 완전한 설정 파일 존재 (`my_robot_localization.yaml`)
   - launch 파일 구조 적절

2. **dead_reckoning 패키지**
   - Ouster IMU 데이터 처리 경험
   - IMU 캘리브레이션 도구 포함
   - QoS 설정 검증됨

## 2. 단계별 구현 계획

### Phase 1: IMU 단독 운용 (현재 가능)

#### 1.1 IMU 데이터 검증
```bash
# bag 파일에서 IMU 토픽 확인
ros2 bag info <your_bag_file>
ros2 bag play <your_bag_file> --topics /ouster/imu
```

#### 1.2 IMU 캘리브레이션
- dead_reckoning 패키지의 캘리브레이션 도구 활용
- 정지 상태 2시간 데이터로 bias 계산
- 중력 벡터 추정

#### 1.3 robot_localization 설정 수정

**수정할 파일**: `gps_imu_fusion/config/imu_only_localization.yaml` (새로 생성)

```yaml
ekf_filter_node:
  ros__parameters:
    frequency: 100.0
    sensor_timeout: 0.1
    two_d_mode: false
    transform_time_offset: 0.0
    print_diagnostics: true
    
    # 프레임 설정
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom
    
    # IMU0 설정 (Ouster OS1)
    imu0: /ouster/imu
    imu0_config: [false, false, false,  # 위치
                  true,  true,  true,   # 방향 (절대)
                  false, false, false,  # 선속도
                  true,  true,  true,   # 각속도
                  true,  true,  true]   # 선가속도
    imu0_nodelay: false
    imu0_differential: false
    imu0_relative: false
    imu0_queue_size: 10
    imu0_remove_gravitational_acceleration: true
    
    # 초기 공분산 (IMU만 사용시 위치 드리프트 예상)
    initial_estimate_covariance: [1e-9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 1e-9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 1e-9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1e-9, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 1e-9, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1e-9, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-1, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-1, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-1]
```

#### 1.4 Launch 파일 생성

**새 파일**: `gps_imu_fusion/launch/imu_only_launch.py`

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('gps_imu_fusion'),
        'config',
        'imu_only_localization.yaml'
    )
    
    return LaunchDescription([
        # EKF 노드 (IMU만 사용)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config_file],
            remappings=[
                # QoS 호환성을 위한 리매핑 필요시 추가
            ]
        ),
        
        # Static TF: base_link → imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_base_to_imu',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link']
            # Ouster IMU 위치에 맞게 조정 필요
        ),
    ])
```

### Phase 2: RTK GPS 통합 (GPS 데이터 확보 후)

#### 2.1 GPS 데이터 검증
- RTK 상태 확인 (Fix/Float/Single)
- 좌표계 변환 확인
- 시간 동기화 검증

#### 2.2 robot_localization 전체 통합

**수정할 파일**: `gps_imu_fusion/config/my_robot_localization.yaml`

주요 수정사항:
```yaml
# IMU 토픽 변경
imu0: /ouster/imu

# navsat_transform_node remapping
navsat_transform_node:
  ros__parameters:
    # 자기 편각 설정 (한국 기준 약 -8도)
    magnetic_declination_radians: -0.1396  # -8 degrees
```

**Launch 파일 수정**: `robot_localization_launch.py`
```python
remappings=[
    ('imu', '/ouster/imu'),
    ('gps/fix', '/ublox_gps_node/fix'),
]
```

## 3. 테스트 및 검증 계획

### 3.1 IMU 단독 테스트
1. **정지 상태 드리프트 측정**
   ```bash
   ros2 launch gps_imu_fusion imu_only_launch.py
   ros2 bag play <stationary_bag> --clock
   ```

2. **드리프트 분석**
   - `/odometry/filtered` 토픽 모니터링
   - 위치 드리프트 정량화
   - 방향 안정성 확인

### 3.2 GPS/IMU 통합 테스트
1. **센서 융합 검증**
   - GPS 수신 상태별 동작 확인
   - GPS 단절 시 IMU 단독 운용
   - RTK Fix 상태에서의 정확도

2. **실시간 성능 평가**
   - CPU 사용률
   - 지연 시간
   - 공분산 수렴성

## 4. 튜닝 가이드라인

### 4.1 IMU 노이즈 파라미터
- 가속도계 노이즈: 정지 상태 데이터에서 계산
- 자이로 노이즈: Allan Variance 분석 권장
- 바이어스 안정성: 장시간 데이터 필요

### 4.2 프로세스 노이즈
- 위치: GPS 없을 때 큰 값 필요
- 속도: 동적 환경에 따라 조정
- 방향: IMU 품질에 따라 조정

### 4.3 센서 공분산
- IMU: 제조사 스펙 참조
- GPS: RTK 상태별 다른 값 적용

## 5. 예상 문제점 및 해결방안

### 5.1 QoS 호환성
- Ouster IMU는 BEST_EFFORT 사용
- robot_localization 기본값과 다를 수 있음
- 필요시 QoS 어댑터 노드 구현

### 5.2 시간 동기화
- IMU와 GPS 타임스탬프 차이
- PTP 또는 chrony 사용 권장
- `use_sim_time` 파라미터 주의

### 5.3 좌표계 정의
- Ouster IMU 좌표계 확인 필요
- ENU vs NED 변환
- Static TF 정확한 설정

## 6. 다음 단계

1. **즉시 가능한 작업**
   - IMU 단독 설정 파일 생성
   - Launch 파일 작성
   - 정지 상태 bag 파일로 테스트

2. **GPS 데이터 확보 후**
   - GPS/IMU 통합 설정
   - 야외 테스트
   - 파라미터 최적화

3. **장기 목표**
   - 휠 오도메트리 추가
   - 다중 센서 융합
   - 동적 공분산 조정