# 문제 해결 가이드

## 현재 발견된 이슈 및 해결 방법

### 1. spherical_interpolated_points 문제
**증상**: 
- 원본 채널의 점들이 yaw가 어긋남
- 보간된 점들의 z가 반전됨

**원인**: 
- 구면 좌표 변환 과정에서 발생하는 오차
- Ouster 좌표 변환 공식의 부정확한 적용

**해결책**:
- `improved_interpolation_node` 사용 권장
- 직접 XYZ 선형 보간으로 더 정확한 결과

### 2. interpolated_points_from_images 미출력
**증상**: 토픽이 발행되지 않음

**원인**:
- intensity_image 토픽이 없음 (signal_image 사용해야 함)
- 이미지 동기화 문제

**해결책**:
- image_interpolation_node.cpp 수정 완료
- `/ouster/signal_image` 구독으로 변경

### 3. 권장 실행 방법

```bash
# 빌드
colcon build --packages-select filc

# 개선된 보간 노드 실행
ros2 run filc improved_interpolation_node

# 파라미터 조정
ros2 param set /improved_interpolation_node use_image_features true
```

## 노드별 특징 비교

| 노드 | 장점 | 단점 | 사용 시기 |
|------|------|------|-----------|
| test_interpolation_node | 가장 빠름, 안정적 | 단순 선형 보간 | 빠른 테스트 |
| image_interpolation_node | 이미지 특징 활용 | 복잡함 | Range/Signal 분석 |
| spherical_interpolation_node | 이론적으로 정확 | 구현 오류 있음 | 디버깅 필요 |
| improved_interpolation_node | test의 안정성 + 이미지 특징 | - | **권장** |

## 디버깅 명령어

```bash
# 토픽 확인
ros2 topic list -v | grep interpolated

# 데이터 확인
ros2 topic echo /ouster/improved_interpolated_points --no-arr --once

# 성능 모니터링
ros2 run filc benchmark_interpolation.py
```