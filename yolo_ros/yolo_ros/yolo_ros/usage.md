# 신호등 감지 및 미션 수행 파이프라인 사용법

## 1. 개요 (Overview)

이 문서는 `detect_traffic_light.py` 노드와 `test_mission_node.py` 노드를 함께 사용하여 신호등을 감지하고, 감지 결과에 따라 특정 미션을 수행하는 전체 파이프라인에 대해 설명합니다.

-   **`detect_traffic_light.py`**: 카메라 이미지로부터 신호등의 색상을 감지하는 핵심 노드입니다. 감지된 색상이 초록색일 경우, `/green` 서비스 호출을 통해 미션 노드에 신호를 보냅니다.

-   **`test_mission_node.py`**: `/green` 서비스 호출을 받아 특정 동작을 수행하는 **제어 노드의 구현 예시**입니다. 이 노드는 서비스 콜백 함수(`green_service_callback`)가 트리거 플래그(`_green_flag`)를 활성화하는 단순한 구조를 가집니다. 사용자는 이 패턴을 참고하여 '초록불일 때 주행 시작'과 같은 실제 제어 로직을 구현할 수 있습니다.

## 2. `detect_traffic_light.py` 상세 설명

### 주요 기능 (Features)

-   **듀얼 모드 ROI 감지**: `yolo` (자동) 또는 `manual` (수동) 모드를 파라미터로 선택할 수 있습니다.
-   **듀얼 카메라 지원**: 2개의 카메라 이미지를 동기적으로 처리하여 더 안정적인 결과를 도출합니다.
-   **실시간 HSV 색상 튜닝**: OpenCV GUI 창의 트랙바를 이용해 실시간으로 빨간색과 초록색의 HSV 범위를 조절하여 환경에 최적화할 수 있습니다.
-   **유연한 시각화**: 카메라, 제어, 마스크 창의 표시 여부를 파라미터로 제어하여 리소스를 효율적으로 사용할 수 있습니다.
-   **서비스 기반 트리거**: 초록불 감지 시, `/green` 서비스를 호출하여 미션 노드에 명확한 신호를 전달합니다.
-   **디버깅용 수동 트리거**: `/virtual_green` 서비스를 통해 실제 감지 없이도 초록불 신호를 강제로 발생시켜 제어 로직을 테스트할 수 있습니다.
-   **실시간 터미널 UI**: 노드의 현재 상태(모드, 검출 상태, 서비스 상태 등)를 터미널에 실시간으로 갱신하여 보여줍니다.

### 터미널 UI 설명 (Terminal UI Guide)

```
--- Traffic Light Detector (Manual) ---
  Detection   : Manual-Based (Detected) | R:15, G:250 / Thr:200
  State       : Green
  Service     : Pending...
  Last Error  : None
-------------------------------------
```

-   **`Detection`**: 현재 ROI 감지 방식, 객체 검출 여부, 검출된 빨강/초록 픽셀 수, 픽셀 임계값을 표시합니다.
-   **`State`**: 검출된 신호등의 최종 색상 상태 (`Green`, `Red`, `Unknown`).
-   **`Service`**: `/green` 서비스 호출 상태를 나타냅니다 (`Idle`, `Pending`, `Calling`, `Failed`, `Succeeded`).
-   **`Last Error`**: 서비스 호출 실패 시의 원인을 표시합니다.

### 파라미터 (Parameters)

-   **`roi_mode`** (string, 기본값: 'manual')
    -   ROI 감지 방식을 선택합니다. `yolo` 또는 `manual` 중 하나를 선택할 수 있습니다.
-   **`target_class_name`** (string, 기본값: 'traffic_light')
    -   `yolo` 모드에서 ROI를 생성할 객체의 클래스 이름입니다.
-   **`pixel_threshold`** (int, 기본값: 200)
    -   색상으로 판단하기 위한 최소 픽셀 수 임계값입니다. GUI의 `Threshold` 트랙바로 실시간 조절이 가능합니다.
-   **`show_camera_windows`** (bool, 기본값: True)
    -   메인 카메라 영상 창의 표시 여부를 설정합니다.
-   **`show_control_windows`** (bool, 기본값: True)
    -   HSV 색상 범위를 조절하는 트랙바 창의 표시 여부를 설정합니다.
-   **`show_mask_windows`** (bool, 기본값: True)
    -   색상이 검출된 영역을 보여주는 마스크 창의 표시 여부를 설정합니다.

## 3. 시스템 파이프라인 및 실행 방법

### 파이프라인 순서

1.  **`test_mission_node` 실행**: `/green` 서비스 서버를 활성화하고 신호를 대기합니다.
2.  **`detect_traffic_light` 실행**: 카메라 영상을 받아 신호등 색상 감지를 시작합니다.
3.  **신호 감지 및 서비스 호출**: `detect_traffic_light`가 초록불을 감지하면 `/green` 서비스를 호출합니다.
4.  **신호 수신 및 미션 수행**: `test_mission_node`가 서비스 호출을 받고 미션을 시작합니다.
5.  **노드 종료**: `detect_traffic_light`는 서비스 호출 성공 후 자동으로 종료됩니다.

### 실행 절차

**1. (터미널 1) 예시 제어 노드 실행**

`test_mission_node`를 먼저 실행하여 `/green` 서비스 요청을 받을 준비를 합니다.

```bash
ros2 run yolo_ros test_mission_node
```

실행 후, 터미널에는 "⏳ 초록불 신호를 대기 중입니다..." 메시지가 주기적으로 출력됩니다.

**2. (터미널 2) 신호등 감지 노드 실행**

`detect_traffic_light` 노드를 실행합니다.

```bash
# Manual 모드 (기본값)
ros2 run yolo_ros detect_traffic_light

# YOLO 모드 (YOLO 노드가 별도로 실행 중이어야 함)
ros2 run yolo_ros detect_traffic_light --ros-args -p roi_mode:=yolo
```

**3. 결과 확인**

`detect_traffic_light` 노드가 초록불을 감지하면, 터미널 2의 서비스 상태가 `Succeeded`로 변경된 후 노드가 종료됩니다. 동시에, 터미널 1의 `test_mission_node`는 "🚀 초록불 확인! 미션을 수행합니다." 메시지를 출력하기 시작합니다.

### 디버깅: `/virtual_green` 서비스 사용하기

카메라나 실제 신호등 없이 제어 로직(e.g., `test_mission_node`)의 동작만 테스트하고 싶을 때, `detect_traffic_light` 노드가 제공하는 `/virtual_green` 서비스를 사용할 수 있습니다. 이 서비스를 호출하면, 실제 초록불이 감지된 것처럼 즉시 `/green` 서비스가 호출되어 파이프라인의 후반부 로직을 독립적으로 테스트할 수 있습니다.

**사용법:** `detect_traffic_light`와 `test_mission_node`가 모두 실행 중인 상태에서, 새 터미널(터미널 3)을 열고 아래 명령어를 입력합니다.

```bash
ros2 service call /virtual_green std_srvs/srv/Trigger
```

호출 즉시 `test_mission_node`가 반응하는 것을 확인할 수 있습니다.

## 4. 인터페이스 요약 (Interfaces)

### `detect_traffic_light.py`

-   **호출하는 서비스**: `/green` (`std_srvs/srv/Trigger`)
-   **제공하는 서비스**: `/virtual_green` (`std_srvs/srv/Trigger`)
-   **구독하는 토픽**:
    -   `/usb_cam_1/image_raw/compressed`, `/usb_cam_2/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)
    -   (YOLO 모드) `/camera_1/detections`, `/camera_2/detections` (`yolo_msgs/msg/DetectionArray`)

### `test_mission_node.py` (예시 제어 노드)

-   **제공하는 서비스**: `/green` (`std_srvs/srv/Trigger`)