# Perception Package Overview

자율주행 Formula Student 차량을 위한 인지 시스템 패키지 모음입니다. LiDAR-카메라 센서 융합, SLAM, GPS/IMU 통합 등 핵심 인지 기능을 제공합니다.

## 패키지 구성

### 핵심 센서 융합 패키지

#### **calico** (Cone Attribute Linking by Image and Cluster Output)
- **기능**: 고성능 C++ 카메라-LiDAR 센서 융합
- **입력**: 
  - LiDAR 콘 위치 (`/sorted_cones_time`)
  - YOLO 탐지 결과 (다중 카메라 지원)
  - IMU 데이터 (`/imu/data`)
- **출력**: 
  - 융합된 콘 정보 (`/tracked_cones`)
  - RViz 시각화 마커
- **핵심 기술**: 
  - 헝가리안 알고리즘 기반 데이터 연관
  - UKF(Unscented Kalman Filter) 추적
  - IMU 보상 (EMA/Butterworth 필터)
- **상태**: 운용 중 (Python hungarian_association 대체)

#### **hungarian_association** 
- **기능**: Python 기반 카메라-LiDAR 센서 융합 (Legacy)
- **입력/출력**: calico와 동일
- **특징**: scipy, filterpy 활용한 원본 구현체
- **상태**: Deprecated (calico로 대체됨)

### 센서 처리 패키지

#### **cone_detection**
- **기능**: LiDAR 포인트 클라우드에서 콘 검출
- **입력**: 원시 포인트 클라우드 (`/velodyne_points`)
- **출력**: 클러스터링된 콘 위치
- **핵심 기술**: 
  - 지면 제거 (RANSAC)
  - DBSCAN 클러스터링
  - 노이즈 필터링

#### **yolo_ros**
- **기능**: YOLO 기반 실시간 객체 탐지
- **입력**: 카메라 이미지
- **출력**: 
  - 탐지 결과 (`DetectionArray`)
  - 바운딩 박스, 마스크, 키포인트
- **특징**: 
  - YOLOv5/v8/v9/v10/v11 지원
  - CUDA 가속
  - 다중 카메라 동시 처리

#### **usb_cam**
- **기능**: USB 카메라 ROS2 드라이버
- **출력**: 압축/비압축 이미지 스트림
- **특징**: 다양한 픽셀 포맷 지원 (MJPEG, YUYV, RGB 등)

#### **filc** (Fusion of Interpolated LiDAR and Camera)
- **기능**: LiDAR 포인트 클라우드 고해상도 보간
- **입력**: 
  - Ouster LiDAR 데이터
  - 다중 카메라 이미지
- **출력**: 색상 정보가 추가된 고밀도 포인트 클라우드
- **핵심 기술**: 
  - 다양한 보간 알고리즘 (LINEAR, CUBIC, BICUBIC, LANCZOS)
  - OpenMP 병렬 처리
  - SIMD 최적화 (AVX2/SSE4.1)

### SLAM 및 위치 추정

#### **cone_stellation** (Cone-based STructural ELement Layout)
- **기능**: 콘 기반 Graph SLAM
- **입력**: 
  - 콘 탐지 결과
  - IMU/GPS 융합 오도메트리
- **출력**: 
  - 최적화된 맵
  - 드리프트 보정 (map→odom 변환)
- **핵심 기술**: 
  - GTSAM 기반 factor graph
  - Inter-landmark factors
  - Loop closure 검출
- **상태**: 개발 진행 중

#### **INS** (Inertial Navigation System)
- **구성 요소**:
  - **RTK_GPS_NTRIP**: RTK GPS + NTRIP 클라이언트
    - u-blox GPS 드라이버
    - 센티미터급 정밀도
  - **myahrs_ros2_driver**: MYAHRS+ IMU 드라이버
    - 9축 IMU (가속도, 자이로, 지자기)
    - 센서 융합된 방향 데이터
  - **gps_imu_fusion**: EKF 기반 센서 융합
    - robot_localization 패키지 활용
    - 100Hz 융합 출력
  - **dead_reckoning**: IMU 기반 추측 항법

### 유틸리티 패키지

#### **kalman_filters**
- **기능**: C++ Kalman 필터 라이브러리
- **제공 필터**: EKF, UKF
- **용도**: calico의 추적 시스템에서 활용

#### **ros2_camera_lidar_fusion**
- **기능**: 카메라-LiDAR 캘리브레이션 도구
- **특징**: 
  - 내부/외부 파라미터 계산
  - 프로젝션 검증
  - 시각화 도구
- **용도**: 초기 시스템 설정 시 사용

#### **custom_interface**
- **기능**: 커스텀 ROS2 메시지 정의
- **메시지 타입**:
  - `TrackedCone`, `TrackedConeArray`
  - `ModifiedFloat32MultiArray` (deprecated)

## 시스템 실행

### 전체 시스템 시작
```bash
# 1. 카메라 드라이버
ros2 launch usb_cam camera.launch.py

# 2. LiDAR 처리
ros2 run cone_detection cone_detection_node

# 3. YOLO 객체 탐지
ros2 launch yolo_bringup yolov8.launch.py

# 4. GPS/IMU 시스템
ros2 launch ublox_gps ublox_gps_node-launch.py
ros2 run myahrs_ros2_driver myahrs_ros2_node
ros2 launch gps_imu_fusion ekf_launch.py

# 5. 센서 융합
ros2 launch calico calico_full.launch.py

# 6. SLAM (선택적)
ros2 launch cone_stellation cone_slam_launch.py
```

## 주요 토픽 구조

### 입력 토픽
- `/camera_*/image_raw`: 원시 카메라 이미지
- `/velodyne_points`: LiDAR 포인트 클라우드
- `/imu/data`: IMU 측정값
- `/gps/fix`: GPS 위치
- `/yolo/detections_*`: YOLO 탐지 결과

### 출력 토픽
- `/tracked_cones`: 최종 융합된 콘 정보
- `/odometry/filtered`: EKF 융합 오도메트리
- `/cone_slam/optimized_poses`: SLAM 최적화 결과
- `/visualization_marker_array`: RViz 시각화

## 설정 파일

각 패키지는 `config/` 디렉토리에 YAML 설정 파일을 포함:
- 카메라 캘리브레이션 (내부/외부 파라미터)
- 센서 융합 파라미터
- SLAM 설정
- EKF 튜닝 파라미터

## 성능 특징

- **실시간 처리**: 20Hz 이상 센서 융합
- **다중 센서**: 최대 4대 카메라 동시 처리
- **정밀도**: RTK GPS로 센티미터급 위치 정확도
- **최적화**: C++ 포팅으로 5x 성능 향상 (calico)
- **병렬 처리**: OpenMP, SIMD 명령어 활용

## 디버깅 도구

- **RViz 시각화**: 모든 센서 데이터 및 처리 결과 실시간 표시
- **프로젝션 디버그**: 카메라-LiDAR 정합 검증
- **성능 모니터링**: 처리 시간 및 지연 측정

## 개발 현황

- **운용 중**: calico, cone_detection, yolo_ros, INS, usb_cam
- **개발 중**: cone_stellation (SLAM)
- **Deprecated**: hungarian_association, ModifiedFloat32MultiArray

## 시스템 아키텍처

```
카메라 → YOLO → 
                 ↘
                  calico (융합) → TrackedCones → 경로 계획
                 ↗                      ↓
LiDAR → 콘 검출 →                 cone_stellation (SLAM)
                                        ↑
GPS/IMU → EKF 융합 → 오도메트리 ────────┘
```