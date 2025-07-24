# Dead Reckoning Package 사용법

## 기본 모드 (캘리브레이션 파일 사용)
기존과 동일하게 config 폴더의 캘리브레이션 파일을 읽어서 적용합니다:
```bash
ros2 launch dead_reckoning dead_reckoning_launch.py
```

## 전처리된 IMU 토픽 사용 모드
imu_preprocess 패키지에서 발행하는 전처리된 토픽을 사용하여 비교 시각화:
```bash
ros2 launch dead_reckoning dead_reckoning_launch.py processed_topic:=/imu/processed
```

## Rotation Only Mode (회전 드리프트만 시각화)
위치 변화를 무시하고 회전(자세) 드리프트만 확인하고 싶을 때 사용합니다:
```bash
# 기본 캘리브레이션 모드에서 회전만 비교
ros2 launch dead_reckoning dead_reckoning_launch.py rotation_only_mode:=true

# 전처리 토픽 모드에서 회전만 비교
ros2 launch dead_reckoning dead_reckoning_launch.py processed_topic:=/imu/processed rotation_only_mode:=true
```

### Rotation Only Mode 특징
- 위치는 고정됨 (왼쪽: 캘리브레이션/전처리, 오른쪽: 원본)
- 10초마다 Roll, Pitch, Yaw 각도와 각도 차이를 로그로 출력
- 종료 시 최종 각도 차이 통계 표시

## 시각화 차이점
### 기본 모드
- **초록색**: 캘리브레이션 적용된 데이터
- **빨간색**: 원본 raw 데이터
- TF frames: `base_link_calibrated`, `base_link_raw`

### 전처리 토픽 모드
- **파란색**: 전처리된 데이터 (imu_preprocess에서 받은 데이터)
- **빨간색**: 원본 raw 데이터
- TF frames: `base_link_processed`, `base_link_raw`

### Rotation Only Mode
- 두 객체가 Y축으로 1미터씩 떨어져서 배치됨
- 위치는 변하지 않고 회전만 변화
- RPY 각도 차이를 실시간으로 확인 가능

## 로그 메시지
- 기본 모드: "캘리브레이션 적용 vs 원본 데이터 비교"
- 전처리 모드: "원본 vs 전처리된 데이터 비교"
- Rotation Only Mode: RPY 각도 및 각도 차이 표시