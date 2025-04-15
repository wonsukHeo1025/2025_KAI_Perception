# GNSS 및 IMU 융합 시스템

이 저장소는 GNSS(GPS) 및 IMU 센서 데이터를 활용한 정밀 위치 측정 및 융합 시스템을 위한 ROS2 패키지들을 포함하고 있습니다. RTK GPS, IMU 센서 및 이들의 데이터 융합을 위한 EKF 알고리즘이 구현되어 있습니다.

## 모듈 구성

### 1. RTK_GPS_NTRIP

#### 기능
RTK_GPS_NTRIP 패키지는 고정밀 GPS 측위를 위한 RTK(Real-Time Kinematics) 기술과 NTRIP(Networked Transport of RTCM via Internet Protocol) 서비스를 활용하는 ROS2 패키지입니다. u-blox GPS 수신기와 연동하여 작동하며, NTRIP 서버로부터 보정 데이터를 수신하여 센티미터 수준의 정밀도를 제공합니다.

#### 구성 요소
- **ublox_msgs**: u-blox GPS 수신기와 통신하기 위한 메시지 정의
- **ublox_serialization**: u-blox 데이터의 직렬화/역직렬화 도구
- **ublox_gps**: u-blox GPS 수신기의 ROS2 드라이버
- **ublox**: u-blox 패키지의 메타패키지
- **ntrip_client**: NTRIP 서버로부터 RTK 보정 데이터를 수신하는 클라이언트
- **rtcm_msgs**: RTCM(Radio Technical Commission for Maritime Services) 메시지 정의
- **fix2nmea**: GPS fix 메시지를 NMEA 형식으로 변환

#### 사용자 수정 코드 및 스크립트 역할
1. **ublox_gps/config/rover.yaml**
   - 이 파일에서 GPS 수신기의 설정을 조정해야 합니다.
   - 주요 수정 항목:
     - `device`: GPS 수신기의 장치 경로 (예: `/dev/ttyACM0`)
     - `frame_id`: TF 프레임 ID 설정
     - `rate`: GPS 메시지 발행 빈도 설정
     - `nav_rate`: 내비게이션 솔루션 갱신 빈도
     - `dynamic_model`: 동적 플랫폼 모델 설정 (automotive, pedestrian 등)

2. **ntrip_client/src/ntrip_client.cpp**
   - NTRIP 클라이언트의 파라미터를 설정합니다.
   - GNSS/reference 관련 토픽 이름, 연결 재시도 간격 등을 조정할 수 있습니다.

3. **fix2nmea/config/default.yaml**
   - GPS fix 메시지를 NMEA 형식으로 변환하는 설정
   - 필요한 경우 토픽 이름 및 변환 설정을 수정합니다.

#### 스크립트 역할 및 관계
- **ublox_gps/src/node.cpp**: u-blox GPS 수신기와 통신하는 ROS2 노드, GPS 데이터를 추출하여 ROS2 토픽으로 발행
- **ntrip_client/src/ntrip_client.cpp**: NTRIP 서버에 연결하여 RTK 보정 데이터를 수신하고 GPS 수신기로 전달
- **ublox_gps/src/gps.cpp**: GPS 수신기 드라이버의 핵심 구현, `node.cpp`에서 사용됨
- **fix2nmea/src/fix2nmea.cpp**: NavSatFix 메시지를 NMEA 형식으로 변환하는 노드, 필요한 경우 다른 시스템과의 통합에 사용됨

#### 사용법

1. **ublox_gps 드라이버 실행**
   ```bash
   ros2 launch ublox_gps ublox_gps_node-launch.py
   ```

2. **NTRIP 클라이언트 실행**
   ```bash
   ros2 run ntrip_client ntrip_client_node --ros-args -p host:=<NTRIP_HOST> -p port:=<PORT> -p mountpoint:=<MOUNTPOINT> -p username:=<USERNAME> -p password:=<PASSWORD>
   ```

3. **토픽 확인**
   ```bash
   ros2 topic list | grep gps
   ros2 topic echo /gps/fix
   ```

### 2. myahrs_ros2_driver

#### 기능
myahrs_ros2_driver는 Withrobot의 MYAHRS+ IMU(관성 측정 장치) 센서를 위한 ROS2 드라이버입니다. 이 센서는 3축 가속도계, 자이로스코프, 지자기계를 포함하며, 센서 융합 알고리즘을 통해 정확한 방향 데이터를 제공합니다.

#### 주요 기능
- IMU 원시 데이터(가속도, 각속도, 자기장) 발행
- 오일러 각(roll, pitch, yaw) 발행
- 쿼터니언 방향 데이터 발행
- 센서 보정 및 설정 기능

#### 사용자 수정 코드 및 스크립트 역할
1. **myahrs_ros2_driver/myahrs_ros2_driver/myahrs_ros2_node.py**
   - IMU 센서 드라이버의 주요 설정을 수정할 수 있습니다.
   - 주요 수정 항목:
     - 시리얼 포트 설정 (`port`): IMU 센서 연결 경로
     - 프레임 ID (`frame_id`): TF 프레임 설정
     - 토픽 이름: 데이터를 발행할 토픽 이름
     - 데이터 필터링 설정
     - 발행 주기

2. **myahrs_ros2_driver/launch/myahrs_ros2.launch.py**
   - IMU 노드 실행을 위한 launch 파일
   - 포트, 프레임 ID 등 기본 파라미터 설정 가능

#### 스크립트 역할 및 관계
- **myahrs_ros2_driver/myahrs_ros2_driver/myahrs_ros2_node.py**: 메인 드라이버 노드, IMU 센서와 통신하고 데이터를 ROS2 토픽으로 발행
- **myahrs_ros2_driver/myahrs_ros2_driver/myahrs_interface.py**: IMU 센서와의 직접적인 통신 인터페이스 구현, 메인 노드에서 활용됨
- **myahrs_ros2_driver/myahrs_ros2_driver/tf_broadcaster.py**: IMU 데이터를 TF로 발행하는 기능 구현

이 드라이버는 gps_imu_fusion 패키지에서 IMU 센서 데이터를 입력으로 사용합니다.

#### 사용법

1. **드라이버 실행**
   ```bash
   ros2 run myahrs_ros2_driver myahrs_ros2_node --ros-args -p port:=/dev/ttyACM0 -p frame_id:=imu_link
   ```

2. **토픽 확인**
   ```bash
   ros2 topic list | grep imu
   ros2 topic echo /imu/data
   ros2 topic echo /imu/mag
   ros2 topic echo /imu/rpy
   ```

### 3. gps_imu_fusion

#### 기능
gps_imu_fusion 패키지는 GPS 데이터와 IMU 데이터를 Extended Kalman Filter(EKF)를 사용하여 융합하는 ROS2 패키지입니다. robot_localization 패키지를 기반으로 하며, 좀 더 정확하고 안정적인 위치 및 방향 추정을 제공합니다.

#### 주요 구성 요소
- **config**: EKF 파라미터 설정 파일
- **ekf_tuning_scripts**: EKF 파라미터 튜닝을 위한 스크립트
- **launch**: 시스템 실행을 위한 launch 파일
- **src**: EKF 구현 및 관련 소스 코드
- **ekf_tuning_results**: 다양한 파라미터 설정에 따른 결과
- **ekf_tuning_bags**: 튜닝에 사용된 rosbag 파일

#### 사용자 수정 코드 및 스크립트 역할
1. **config/ekf.yaml**
   - EKF 파라미터 설정 파일로, 가장 중요한 수정 대상입니다.
   - 주요 수정 항목:
     - `frequency`: EKF 예측 단계 실행 빈도
     - `sensor_timeout`: 센서 데이터가 수신되지 않는 경우의 타임아웃 설정
     - `two_d_mode`: 2D 모드 사용 여부
     - `map_frame`, `odom_frame`, `base_link_frame`: TF 프레임 설정
     - `odom0` ~ `odom9`, `imu0` ~ `imu9`, `pose0` ~ `pose9`: 각 센서 입력의 설정
       - `topic`: 데이터를 수신할 토픽
       - `config`: 각 상태 변수(x, y, z, roll, pitch, yaw 등)의 사용 여부
       - `differential`: 차분 모드 사용 여부
       - `queue_size`, `nodelay`: 메시지 큐 설정
     - `process_noise_covariance`: 시스템 모델 노이즈
     - `initial_estimate_covariance`: 초기 상태 추정 불확실성

2. **launch/ekf_launch.py**
   - EKF 노드를 실행하기 위한 launch 파일
   - 필요에 따라 EKF 설정 파일 경로, 노드 이름 등을 수정할 수 있습니다.

3. **ekf_tuning_scripts/** 디렉토리
   - `compare_results.py`: 서로 다른 EKF 설정의 결과를 비교하는 스크립트
   - `plot_ekf_results.py`: EKF 결과를 시각화하는 스크립트
   - 데이터 분석 및 파라미터 튜닝에 활용할 수 있습니다.

#### 스크립트 역할 및 관계
- **src/ekf_node.cpp**: (가정) EKF 알고리즘을 구현한 메인 노드, robot_localization 패키지의 EKF 구현을 확장하거나 수정
- **src/ekf_localization.cpp**: (가정) 실제 EKF 알고리즘 로직을 구현한 클래스, main 노드에서 사용됨
- **ekf_tuning_scripts/**: 다양한 분석 및 튜닝 스크립트, 실제 EKF 구현과는 직접 연결되지 않으나 파라미터 최적화에 중요

이 패키지는 RTK_GPS_NTRIP의 GPS 데이터와 myahrs_ros2_driver의 IMU 데이터를 입력으로 사용하여 융합된 로컬라이제이션 정보를 출력합니다.

#### 사용법

1. **EKF 노드 실행**
   ```bash
   ros2 launch gps_imu_fusion ekf_fusion.launch.py 
   ```

2. **파라미터 튜닝**
   
   파라미터 튜닝은 config 디렉토리의 YAML 파일을 수정하여 수행합니다.
   
   주요 파라미터:
   - process_noise_covariance: 시스템 모델의 불확실성
   - initial_estimate_covariance: 초기 상태 추정의 불확실성
   - use_control: 제어 입력 사용 여부
   - sensor_timeout: 센서 데이터 타임아웃 설정
   
   자세한 튜닝 방법은 `파라미터 수정 전략` 및 `초기 불확실성 수정 전략` 문서를 참조하세요.

3. **결과 확인**
   ```bash
   ros2 topic echo /odometry/filtered
   ```

### 4. visualization_tutorials

#### 기능
visualization_tutorials 패키지는 ROS의 시각화 도구인 RViz를 활용한 다양한 튜토리얼과 예제를 제공합니다. 이 시스템에서는 GPS 및 IMU 데이터 융합의 결과를 시각화하는 데 활용할 수 있습니다.

#### 주요 구성 요소
- **interactive_marker_tutorials**: 대화형 마커를 사용한 시각화 예제
- **librviz_tutorial**: RViz 라이브러리를 프로그래밍 방식으로 사용하는 방법
- **rviz_plugin_tutorials**: RViz 플러그인 개발 예제
- **rviz_python_tutorial**: Python에서 RViz 활용 방법
- **visualization_marker_tutorials**: 다양한 시각화 마커 활용 예제

#### 사용자 수정 코드 및 스크립트 역할
1. **RViz 설정 파일**
   - 본 시스템을 위한 RViz 구성 파일을 생성하거나 수정해야 합니다.
   - 주요 수정 항목:
     - `Fixed Frame`: 시각화의 기준이 되는 TF 프레임 설정
     - 디스플레이 설정: GPS, IMU, 융합된 오도메트리 등의 시각화 설정
     - 마커 설정: 필요한 경우 마커의 모양, 크기, 색상 등을 조정

2. **visualization_marker_tutorials/src/**
   - 필요한 경우 마커 예제 코드를 참조하여 융합 결과를 시각화하는 커스텀 노드를 개발할 수 있습니다.
   - **points_and_lines.cpp**: 점과 선 기반 마커 예제
   - **basic_shapes.cpp**: 기본 도형 마커 예제

#### 스크립트 역할 및 관계
- **visualization_marker_tutorials/src/basic_shapes.cpp**: 기본 도형 마커 생성 예제, 센서 위치나 이동 경로 표시에 활용 가능
- **interactive_marker_tutorials/src/basic_controls.cpp**: 대화형 마커 예제, 사용자가 시각화된 개체와 상호작용 할 수 있는 기능 제공
- **rviz_plugin_tutorials/src/**: RViz 플러그인 개발 예제, 필요한 경우 GPS/IMU 데이터를 위한 커스텀 시각화 플러그인 개발에 참조 가능

이 패키지는 gps_imu_fusion의 결과를 시각화하는 데 활용되며, 필요에 따라 추가적인 시각화 기능을 개발할 수 있습니다.

#### 사용법

1. **마커 튜토리얼 실행**
   ```bash
   ros2 run visualization_marker_tutorials basic_shapes
   ```

2. **인터랙티브 마커 예제 실행**
   ```bash
   ros2 run interactive_marker_tutorials basic_controls
   ```

3. **RViz에서 시각화**
   ```bash
   ros2 run rviz2 rviz2
   ```
   
   RViz에서 필요한 시각화 플러그인을 추가하여 GPS, IMU 및 융합 데이터를 시각화할 수 있습니다.

## 전체 시스템 실행 순서

1. **IMU 드라이버 실행**
   ```bash
   ros2 launch myahrs_ros2_driver myahrs_ros2_driver.launch.py 
   ```

2. **GPS 드라이버 실행**
   ```bash
   ros2 launch ublox_gps ublox_gps_node-launch.py
   ```

3. **NTRIP 클라이언트 실행**
   ```bash
   ros2 launch ntrip_client ntrip_client_launch.py
   ```

4. **gps nmea fix**
   ```bash
   ros2 run fix2nmea fix2nmea
   ```

5. **EKF 융합 노드 실행**
   ```bash
   ros2 launch gps_imu_fusion ekf_fusion.launch.py 
   ```

## 주의사항 및 팁

1. GPS 및 IMU 센서의 물리적 장착 위치와 방향에 주의하세요. 정확한 변환(transform)을 설정해야 합니다.
2. EKF 파라미터 튜닝은 시스템 성능에 큰 영향을 미칩니다. `gps_imu_fusion` 패키지의 튜닝 문서를 참조하세요.
3. NTRIP 서비스를 사용하기 위해서는 유효한 계정 정보가 필요합니다.
4. 센서 초기화 및 보정(calibration)이 올바르게 수행되었는지 확인하세요.
5. 테스트 환경에 따라 센서 데이터의 품질이 달라질 수 있으므로, 다양한 조건에서 시스템을 테스트하는 것이 좋습니다. 