# Dashboard CLI (ROS2 Terminal Monitor)

경량 터미널 대시보드로 자율주행 파이프라인의 핵심 토픽과 노드 상태를 실시간 모니터링합니다. 실차 기준으로 바뀌지 않는 토픽을 하드코딩해 가독성과 유지보수를 높였습니다.

## Quick Start

- 빌드: 워크스페이스 루트에서
  - `colcon build --packages-select dashboard_cli`
  - `source install/setup.bash`
- 실행:
  - `ros2 run dashboard_cli dashboard`

## 무엇을 보여주나

- PERCEPTION: 카메라/라이다 Hz, 콘 검출/융합(UKF) 개수, 오도메트리 수신 여부
- PLANNING: `/local_planned_path` Hz, `/desired_speed_profile` 길이
- CONTROL: 조향 명령/피드백 Hz, 속도 피드백, RPM 명령/출력
- SAFETY: AEB 이벤트 수신
- NODES: 필수 노드 liveness (UP/DOWN)

각 항목은 GO/NO-GO + 현재 Hz와 간단 요약 값(예: LiDAR points, Path poses, Cone count)을 표시합니다.

## 기본 모니터링 대상 토픽 (요약)

- 카메라/라이다: `/usb_cam_1/image_raw`, `/usb_cam_2/image_raw`, `/ouster/points`
- 콘: `/cone/lidar`, `/cone/fused`, `/cone/fused/ukf`
- 로컬라이제이션: `/odometry/filtered`
- 경로/속도: `/local_planned_path`, `/desired_speed_profile`
- 제어: `/cmd/steer`, `/ctrl/steer`, `/cmd/rpm`, `/target_rpm`, `/ctrl/speed`, `/throttle_data`
- 안전: `/aeb`

임계치(예): 카메라 20Hz, 라이다 10Hz(+ min_points), 로컬 경로 10Hz 등.

## 예상 실행 노드 (요약)

- Perception: `calico_ukf_tracking`, `calico_multi_iou_fusion`, `outlier_filter`, `cone_detection_visualization_node`
- Localization: `ekf_fusion_node`
- Planning/Control: `pure_pursuit_static`, `simple_speed_planner`, `steering_control`, `steering_feedback`, `rpm_mux`, `can_speed_sender`, `can_speed_receiver`, `throttle_serial`

## 아키텍처

- `dashboard_cli/dashboard_cli/spec.py`: 하드코딩된 토픽/노드/임계치 정의
- `dashboard_cli/dashboard_cli/topics.py`: 토픽 구독, Hz/지연 및 요약값 계산
- `dashboard_cli/dashboard_cli/metrics.py`: 슬라이딩 윈도우 기반 Hz/age 계산
- `dashboard_cli/dashboard_cli/nodes.py`: ROS 그래프에서 노드 liveness 확인
- `dashboard_cli/dashboard_cli/ui.py`: 터미널 ASCII UI 렌더링
- `dashboard_cli/dashboard_cli/dashboard_node.py`: 오케스트레이션(타이머, 화면 갱신)

## 커스터마이즈(코드 레벨)

- 토픽/임계치 추가·수정: `Perception/dashboard_cli/dashboard_cli/spec.py`의 `TOPICS`, `EXPECTED_NODES`, `UI_REFRESH_HZ` 수정
- 헬스 체크 추가: `topics.py`의 `get_status()`에서 규칙 확장(예: 조향 명령-피드백 오차)
- UI 변경: `ui.py`에서 섹션/행 구성 변경 또는 색상/서식 추가

## 의존성

- ROS2 rclpy + 메시지 패키지: `std_msgs`, `sensor_msgs`, `nav_msgs`, `visualization_msgs`
- 사용자 메시지: `custom_interface`, `throttle_msgs`, (선택) `yolo_msgs`
  - 특정 메시지가 설치되지 않은 경우 해당 토픽은 건너뛰며 경고만 출력합니다.

## 참고

- 본 대시보드는 HSM 기반 로컬 경로 파이프라인(/cmd/*, /ctrl/* 토픽 체계)을 기준으로 설계되었습니다. 토픽이 변경되면 `spec.py`만 수정하면 됩니다.

