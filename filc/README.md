# Fine-grained Interpolation for LiDAR Continuity (filc)

Ouster OS1-32 LiDAR의 32채널 포인트클라우드를 128채널로 실시간 보간하는 ROS2 패키지

## 개요
- **목표**: 32채널 → 128채널 포인트클라우드 보간
- **입력**: `/ouster/points` (32×1024)
- **출력**: `/ouster/improved_interpolated_points` (128×1024)
- **성능**: 실시간 처리 (>10Hz)

## 빠른 시작

### 1. 빌드
```bash
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select filc
source install/setup.bash
```

### 2. 실행
```bash
# Launch 파일 사용 (권장)
ros2 launch filc interpolation.launch.py

# 파라미터 조정
ros2 launch filc interpolation.launch.py scale_factor:=2.0

# 노드 직접 실행
ros2 run filc improved_interpolation_node
```

### 3. 모니터링
```bash
# 실시간 통계
ros2 run filc visualize_interpolation.py

# 성능 벤치마크
ros2 run filc benchmark_interpolation.py
```

## 주요 기능

### improved_interpolation_node
- **XYZ 직접 보간**: 안정적이고 빠른 처리
- **적응적 보간**: 불연속성 감지 (>0.5m) 시 nearest neighbor
- **병렬 처리**: OpenMP를 통한 멀티코어 활용
- **설정 가능**: scale factor 2.0~4.0 동적 조정

### 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| scale_factor | 4.0 | 보간 배율 (2.0, 3.0, 4.0 권장) |
| interpolation_method | cubic | 보간 방법 (linear/cubic) |
| discontinuity_threshold | 0.5 | 불연속성 임계값 (m) |
| use_image_features | false | 이미지 특징 사용 (개발 중) |

## 설정 파일
`config/interpolation_config.yaml`에서 상세 설정 가능

## 토픽
- **입력**: `/ouster/points` - 원본 32채널 포인트클라우드
- **출력**: `/ouster/improved_interpolated_points` - 보간된 128채널 포인트클라우드

## 시각화
```bash
# RViz2
rviz2
# PointCloud2 추가, Topic: /ouster/improved_interpolated_points
# Fixed Frame: os_sensor
```

## 문제 해결

### QoS 호환성
센서 데이터는 best_effort QoS 사용. 이미 스크립트에 반영됨.

### 성능 최적화
- scale_factor를 낮춰서 테스트 (2.0부터 시작)
- CPU 코어 수에 따라 자동 병렬화

## 라이선스
MIT License