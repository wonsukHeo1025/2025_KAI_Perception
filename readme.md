# ROS2 작업공간 소스 패키지 개요

## 문서 관리 가이드라인

### 목적
이 README는 작업공간의 모든 패키지 개요, 패키지 간 관계, 의존성 문제 및 개선사항을 추적합니다. 개별 패키지의 세부사항은 각 패키지의 README에 문서화되어야 합니다.

### 업데이트 형식
모든 업데이트는 다음 형식을 따라야 합니다:
```
[YYYY-MM-DD HH:MM:SS] [카테고리] 설명
```
카테고리: DEPENDENCY, BUILD, INTEGRATION, REFACTOR, FIX, ANALYSIS

---

## 패키지 개요

### 핵심 인식 패키지

#### 1. **cone_detection**
- **기능**: LiDAR 기반 콘 탐지 및 클러스터링
- **입력**: `/ouster/points` (PointCloud2)
- **출력**: `/cone/lidar`, `/cone/lidar/ukf` (TrackedConeArray)
- **의존성**: PCL, Eigen3, kalman_filters (시스템 라이브러리)

#### 2. **yolo_ros**
- **기능**: YOLO를 사용한 카메라 기반 객체 탐지
- **입력**: 카메라 이미지
- **출력**: `/yolo/detections` (DetectionArray)
- **의존성**: OpenCV, Python YOLO 라이브러리

#### 3. **calico** (C++ 고성능)
- **기능**: 다중 카메라와 LiDAR 센서 융합
- **입력**: YOLO 탐지 + LiDAR 콘
- **출력**: 속성이 포함된 융합 콘 위치
- **의존성**: kalman_filters (시스템 라이브러리), OpenCV, Eigen3

#### 4. **hungarian_association** (Python 레거시)
- **기능**: Hungarian 알고리즘을 사용한 센서 융합
- **입력**: YOLO 탐지 + LiDAR 콘
- **출력**: 연관된 콘 데이터
- **의존성**: numpy, scipy, filterpy

### SLAM 및 위치추정

#### 5. **cone_stellation**
- **기능**: 루프 클로저를 포함한 콘 기반 SLAM
- **입력**: 추적된 콘, IMU, GPS
- **출력**: 차량 포즈, 랜드마크 맵
- **의존성**: GTSAM, Eigen3

#### 6. **INS** (디렉토리)
- **하위 패키지**: RTK_GPS_NTRIP, ublox, myahrs_ros2_driver
- **기능**: GPS/IMU 통합 및 융합
- **출력**: `/odometry/filtered`, `/fix`

### 유틸리티 및 인프라

#### 7. **kalman_filters**
- **기능**: 추적을 위한 UKF/EKF 라이브러리
- **상태**: ROS2 패키지로 작동 중
- **사용처**: cone_detection, calico

#### 8. **custom_interface**
- **기능**: 커스텀 메시지 정의
- **메시지**: TrackedConeArray, ModifiedFloat32MultiArray

#### 9. **usb_cam**
- **기능**: USB 카메라 드라이버
- **출력**: 카메라 이미지 및 정보

#### 10. **filc**
- **기능**: LiDAR 포인트 클라우드 보간 (32→128 채널)
- **입력**: `/ouster/points` (32×1024 PointCloud2)
- **출력**: `/ouster/points/interpolated` (128×1024 PointCloud2)
- **의존성**: PCL, OpenCV, OpenMP
- **성능**: ~50ms/프레임

#### 11. **ros2_camera_lidar_fusion**
- **기능**: 카메라-LiDAR 캘리브레이션 및 투영 유틸리티

---

## 패키지 간 의존성

### 의존성 그래프
```
kalman_filters (ROS2 패키지)
    ├── cone_detection
    └── calico

custom_interface
    ├── cone_detection
    ├── calico
    ├── hungarian_association
    └── cone_stellation

yolo_msgs
    ├── yolo_ros
    ├── calico
    └── hungarian_association

현재 파이프라인:
cone_detection → calico/hungarian_association → cone_stellation
       ↓              ↓                              ↑
  (LiDAR 콘)     (융합된 콘)                   (SLAM 입력)

제안 파이프라인 (FILC 통합):
/ouster/points → FILC → /ouster/enhanced_points → cone_detection
                    ↓                                     ↓
       /ouster/interpolated_points → calico    (향상된 LiDAR 콘)
```

### 중요 의존성
1. **kalman_filters**: ROS2 패키지로 작동 중
2. **custom_interface**: 모든 의존 패키지보다 먼저 빌드되어야 함
3. **yolo_msgs**: 비전 파이프라인에 필요

---

## 개발 로그

### [2025-08-10] [STATUS] cone_detection 현재 상태
- **완료**: DBSCAN 구현, 968라인으로 코드 감소, 시각화 노드 분리, Ouster 포맷 변환 제거

### [2025-08-10] [STATUS] FILC 현재 상태
- **완료**: package.xml 의존성 추가, CMakeLists.txt Eigen3 제거

### [2025-08-10] [STATUS] CALICO 현재 상태
- **완료**: Thread safety (std::atomic 사용), 보안 검증 (YAML 입력, 경로), dlib 의존성 제거, Butterworth 필터 구현

---

*마지막 업데이트: 2025-08-10 (cone_detection 대규모 개선 완료)*