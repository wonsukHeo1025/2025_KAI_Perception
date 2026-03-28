# 2025 KAI 자율주행자동차 Perception Workspace

## 개요

담당: GPS·IMU 융합 기반 위치 추정 + 카메라-라이다 센서 퓨전 및 객체 추적

RTK-GNSS, IMU, LiDAR, 듀얼 카메라를 센서로 사용하는 자율주행 자작자율차 Perception 워크스페이스입니다.  
저는 차량의 자기 위치 추정 정확도와 객체 인지 성능을 동시에 높이는 역할을 맡았고 EKF 기반 로컬라이제이션, YOLO 기반 콘 검출, 카메라-라이다 투영 및 객체 단위 센서 퓨전, Hungarian + UKF 기반 추적 파이프라인을 중심으로 구현했습니다.


## 담당 범위

| 영역 | 패키지/모듈 | 구현한 내용 |
|---|---|---|
| GPS·IMU Fusion | [`INS/gps_imu_fusion`](./INS/gps_imu_fusion) | EKF 기반 15-state 위치 추정, GNSS position/velocity update, IMU 연속 예측, 가속도계/자이로 바이어스 상태 추정, GPS·IMU lever arm 보정, 정지·전이·주행 3상태 적응형 신뢰도 조절 |
| Camera Cone Detection | [`yolo_ros/yolo_ros/yolo_ros/yolo_dual_camera_node.py`](./yolo_ros/yolo_ros/yolo_ros/yolo_dual_camera_node.py) | 실환경 데이터로 학습한 YOLOv8 콘 검출 모델 적용, 듀얼 카메라 동시 추론, 카메라별 `DetectionArray` 및 디버그 이미지 발행 |
| LiDAR Cone Detection | [`cone_detection`](./cone_detection) | 포인트클라우드 필터링, 클러스터링, 콘 후보 추출, `TrackedConeArray` 및 `BoundingBox3DArray` 기반 LiDAR 콘 출력 구성 |
| Projection / Calibration | [`prism`](./prism) | 다중 카메라 intrinsic/extrinsic calibration 로딩, LiDAR 3D 포인트 및 박스의 2D 이미지 투영, 카메라-라이다 정합 디버그 시각화 |
| Camera-LiDAR Fusion | [`calico/src/nodes/multi_iou_fusion_node.cpp`](./calico/src/nodes/multi_iou_fusion_node.cpp) | 투영된 LiDAR 3D box와 카메라 bbox 간 IoU cost 계산, Hungarian 기반 최적 매칭, 멀티 카메라 결과 병합, `TrackedConeArray` 기반 융합 결과 발행 |
| Multi-Object Tracking | [`calico/src/nodes/ukf_tracking_node.cpp`](./calico/src/nodes/ukf_tracking_node.cpp), [`kalman_filters`](./kalman_filters) | UKF 기반 객체 추적, IMU 기반 ego-motion 보상, matched/unmatched cone 관리, 신규 트랙 생성 및 소실 트랙 제거 |
| ROS 2 Interface | [`custom_interface`](./custom_interface) | `TrackedCone.msg`, `TrackedConeArray.msg`, `ModifiedFloat32MultiArray.msg` 등 Perception 파이프라인 메시지 인터페이스 구성 |

## 주요 기능

GPS + IMU Fusion
- IMU 입력이 들어올 때마다 EKF predict를 수행해 위치, 속도, heading을 연속적으로 추정
- RTK-GNSS position/velocity를 사용해 저주파 보정을 수행하고 `/odometry/filtered`를 발행
- 가속도계 바이어스와 자이로 바이어스를 상태에 포함해 장기 drift를 완화
- GPS position/velocity, IMU acceleration에 대해 lever arm dynamics를 보정해 실차 장착 오차를 줄임
- 정지/전이/주행 3상태에 따라 ZUPT noise, GNSS heading noise, IMU update 빈도를 동적으로 조절

Cone Detection
- 실환경 주행 이미지 기반으로 학습한 YOLOv8 custom cone detector 적용
- 단일 카메라/듀얼 카메라 노드를 분리해 운용하고, 듀얼 카메라로 FOV 한계를 보완
- LiDAR 포인트클라우드에서 콘 클러스터와 3D bounding box를 생성해 카메라-라이다 융합 입력으로 사용

Camera-LiDAR Fusion + Tracking
- calibration YAML을 기반으로 LiDAR 3D point/box를 각 카메라 이미지 평면으로 투영
- 투영 결과와 YOLO bbox 사이의 IoU cost matrix를 계산하고 Hungarian algorithm으로 최적 매칭
- 멀티 카메라에서 들어온 분류 결과를 병합해 각 LiDAR cone의 color/class를 결정
- IMU 보상이 포함된 UKF로 객체를 추적하고, 미매칭 검출은 신규 트랙으로 생성하며 오래 끊긴 트랙은 제거


## 상세 구현

### 1. GPS·IMU 융합 로컬라이제이션 구현

- EKF 상태를 위치, 속도, 자세, accel bias, gyro bias까지 포함한 15-state 구조로 설계
- RTK-GNSS 위치/속도와 IMU를 함께 사용해 저주파 측정 업데이트 + 고주파 예측 구조 구현
- 차량의 평면 주행 가정을 반영해 2D driving oriented odometry를 구성
- GNSS heading 사용 여부와 noise를 속도 구간에 따라 다르게 적용해 저속 및 전이 구간의 yaw 추정을 안정화
- ZUPT hysteresis, stationary IMU decimation 등을 넣어 정지 시 drift를 억제

#### Step 1. IMU 기반 연속 예측

- `imu/processed`가 들어올 때마다 `KaiEkfCore`에서 속도, 위치, heading을 연속적으로 예측
- 예측 단계에서 가속도계/자이로 bias를 함께 상태에 포함해 장기 주행에서 누적 오차를 줄이도록 구성

#### Step 2. GNSS 기반 보정

- `/ublox_gps_node/fix`, `/ublox_gps_node/fix_velocity`를 사용해 EKF의 position/velocity residual을 계산
- GNSS covariance를 받아 측정 신뢰도를 동적으로 반영하고, 칼만 이득으로 상태를 보정

#### Step 3. Lever Arm + Adaptive 신뢰도 조절

- `base_link`와 GPS, IMU 사이의 실제 장착 거리로 인해 생기는 위치/속도/가속도 오차를 lever arm 보정식으로 보완
- 정지, 전이, 주행 상태에 따라 GNSS heading noise와 프로세스 노이즈를 다르게 적용해 adaptive fusion 구조를 구성

### 2. YOLOv8 기반 콘 디텍션 및 듀얼 카메라 파이프라인

- 커스텀 학습한 `best.pt` 모델을 사용해 콘 색상/클래스를 분류하는 YOLO 노드 구성
- 듀얼 카메라 노드에서 `camera_1`, `camera_2` 이미지를 각각 별도 큐와 스레드로 처리해 실시간성을 확보
- 카메라별로 `/camera_1/detections`, `/camera_2/detections`, 디버그 이미지, 콘 정보 토픽을 발행
- 단일 카메라 시야각 한계를 보완하기 위해 듀얼 카메라 구조를 적용해 더 넓은 영역의 콘 인식을 가능하게 함

### 3. 카메라-라이다 투영 및 정합 검증

- [`prism/config/multi_camera_intrinsic_calibration.yaml`](./prism/config/multi_camera_intrinsic_calibration.yaml), [`prism/config/multi_camera_extrinsic_calibration.yaml`](./prism/config/multi_camera_extrinsic_calibration.yaml)을 기반으로 카메라 내부/외부 파라미터를 로딩
- LiDAR 포인트와 3D box를 카메라 좌표계로 변환한 뒤 `cv::projectPoints`를 사용해 2D 이미지 평면으로 투영
- `projection_debug_node`를 통해 각 카메라 영상 위에 투영 결과를 오버레이해 센서 정합 상태를 검증

### 4. 객체 단위 센서 융합 및 UKF 추적

- LiDAR에서 생성한 `/cone/lidar/box`와 카메라 검출 bbox를 같은 이미지 평면으로 맞춘 뒤 IoU 기반 cost matrix 생성
- Hungarian matcher를 사용해 전체 비용이 최소가 되는 최적 매칭 결과를 계산
- 멀티 카메라 입력에서는 카메라별 매칭 결과를 병합해 각 LiDAR cone의 최종 class를 결정하고, 매칭되지 않은 경우 `Unknown`으로 유지
- `/cone/fused` 결과에 대해 UKF 기반 추적을 수행하고, IMU 데이터를 함께 사용해 ego-motion 보상 기반 예측을 적용
- 매칭된 객체는 update, 미매칭 객체는 신규 track 생성 또는 기존 track 제거 방식으로 전체 tracking pipeline을 구성


## 저장소 구조

```text
.
├── INS
│   ├── gps_imu_fusion
│   │   ├── config
│   │   │   └── fusion_params.yaml
│   │   ├── launch
│   │   │   └── ekf_fusion.launch.py
│   │   ├── include/gps_imu_fusion
│   │   │   ├── ekf_fusion_node.hpp
│   │   │   └── kai_ekf_core.hpp
│   │   ├── src
│   │   │   ├── ekf_fusion_node.cpp
│   │   │   └── kai_ekf_core.cpp
│   │   └── test
│   │       └── test_lever_arm.py
│   ├── imu_preprocess
│   ├── myahrs_ros2_driver
│   └── RTK_GPS_NTRIP
├── yolo_ros
│   ├── yolo_bringup
│   ├── yolo_msgs
│   └── yolo_ros
│       ├── models
│       │   └── best.pt
│       └── yolo_ros
│           ├── yolo_single_camera_node.py
│           └── yolo_dual_camera_node.py
├── cone_detection
│   ├── config
│   │   └── cone_detection_config.yaml
│   └── src
│       ├── cone_detection_node.cpp
│       └── dbscan_clusterer.cpp
├── prism
│   ├── config
│   │   ├── multi_camera_intrinsic_calibration.yaml
│   │   └── multi_camera_extrinsic_calibration.yaml
│   └── src
│       ├── projection
│       │   └── multi_camera_fusion.cpp
│       └── nodes
│           └── projection_debug_node.cpp
├── calico
│   ├── config
│   │   └── multi_hungarian_config.yaml
│   └── src
│       ├── fusion
│       │   └── hungarian_matcher.cpp
│       └── nodes
│           ├── multi_iou_fusion_node.cpp
│           └── ukf_tracking_node.cpp
├── kalman_filters
│   ├── include/kalman_filters/tracking
│   └── src/tracking
└── custom_interface
    └── msg
        ├── TrackedCone.msg
        ├── TrackedConeArray.msg
        └── ModifiedFloat32MultiArray.msg
```
