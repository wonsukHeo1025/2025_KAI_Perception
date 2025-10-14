# 신호등 감지 및 미션 수행 파이프라인 사용법

## 1. 개요 (Overview)

이 문서는 `detect_traffic_light.py` 노드와 `test_mission_node.py` 노드를 함께 사용하여 신호등을 감지하고, 감지 결과에 따라 특정 미션을 수행하는 전체 파이프라인에 대해 설명합니다.

-   **`detect_traffic_light.py`**: 카메라 이미지로부터 신호등의 색상을 감지하는 핵심 노드입니다. 감지된 색상이 초록색일 경우, `/green` 서비스 호출을 통해 미션 노드에 신호를 보냅니다.

-   **`test_mission_node.py`**: `/green` 서비스 호출을 받아 특정 동작을 수행하는 **제어 노드의 구현 예시**입니다. 이 노드는 서비스 콜백 함수(`green_service_callback`)가 트리거 플래그(`_green_flag`)를 활성화하는 단순한 구조를 가집니다. 사용자는 이 패턴을 참고하여 '초록불일 때 주행 시작'과 같은 실제 제어 로직을 구현할 수 있습니다.

## 2. `detect_traffic_light.py` 상세 설명

### 주요 기능 (Features)

-   **`roi_mode` 기반 모드 전환**: `manual`(기본)과 `yolo` 모드를 ROS 파라미터로 선택해 상황에 맞는 감지 파이프라인을 사용할 수 있습니다.
-   **YOLO-규칙 융합 의사결정**: Ultralytics YOLO 추론과 규칙 기반 색 판별을 함께 수행해 합의·충돌 카운터를 통해 안정적인 최종 신호등 색상을 도출합니다.
-   **수동 ROI 관리 및 영구 저장**: Manual 모드에서 `q` 키로 ROI 편집을 토글하고 마우스로 영역을 지정하면 설정이 `rois.json`에 자동 저장됩니다.
-   **가벼운 시각화 제어**: 카메라 창 표시 여부를 파라미터로 제어하고, 디버그 시 이진화 미리보기·규칙 정보 창을 추가로 확인할 수 있습니다.
-   **서비스 연동과 디버그 모드**: 초록불 확정 시 `/green` 서비스를 호출하며, `debug_mode`가 참이면 호출을 생략하고 GUI 상태만 갱신합니다.
-   **`/virtual_green` 테스트 서비스**: 실제 감지 없이도 초록불 시나리오를 재현해 후속 미션 노드를 단독으로 검증할 수 있습니다.

### 모드 구성 및 전환

-   **Manual (기본값)**: 규칙 기반 색 판별만 수행합니다. ROI는 `rois.json`에서 불러오거나 카메라 창에서 `q` 키로 편집 모드를 켜고 드래그하여 갱신할 수 있습니다.
-   **YOLO**: `yolo_model_path`에 지정된 Ultralytics 모델을 로드해 신호등 후보 영역을 찾고, 규칙 기반 검증으로 최종 색상을 결정합니다. 모델을 찾지 못하면 노드가 종료됩니다.

`roi_mode` 파라미터로 모드를 전환합니다.

```bash
ros2 run yolo_ros detect_traffic_light --ros-args -p roi_mode:=yolo
```

Ultralytics `YOLO` 패키지가 설치되어 있어야 하며, 필요 시 `-p yolo_model_path:=<모델 경로>`로 모델 파일을 지정합니다. 실 환경에서 `/green` 서비스를 실제로 호출하려면 `-p debug_mode:=false`를 추가하세요. `show_control_windows`와 `show_mask_windows` 파라미터는 두 모드 모두에서 코드가 자동으로 `False`로 재설정합니다.

### 터미널 UI 설명 (Terminal UI Guide)

```
--- Traffic Light Detector (Manual) ---
  Detection   : [DEBUG] Manual Mode | ROI1:Green | Consensus pending 1/2
  State       : Green
  Confidence  : RuleBlob:TY/BN (1/0) | EMA R:0.32/G:0.58
  Service     : Debug (skip trigger)
  Last Error  : None
-------------------------------------
```

-   **`Detection`**: 현재 모드와 1·2차 감지 요약 정보를 보여 줍니다. `[DEBUG]` 접두사는 `debug_mode=True`일 때 자동으로 붙습니다.
-   **`State`**: 누적 판별 결과로 확정된 신호등 색상(`Green`, `Red`, `Unknown`).
-   **`Confidence`**: YOLO 신뢰도와 규칙 기반 블롭 검출 요약(Manual 모드에서는 규칙 지표만 표시)을 제공합니다.
-   **`Service`**: `/green` 서비스 호출 상태 (`Idle`, `Pending...`, `Calling...`, `Succeeded`, `Failed`, `Debug (skip trigger)` 등).
-   **`Last Error`**: 서비스 호출 실패 또는 예외 발생 시 마지막 에러 메시지를 보여줍니다.

### 파라미터 (Parameters)

-   **`roi_mode`** (string, 기본값: `'manual'`)
    -   감지 파이프라인을 선택합니다. `manual` 또는 `yolo` 중 하나를 지정합니다.
-   **`pixel_threshold`** (int, 기본값: `200`)
    -   규칙 기반 로직 확장을 위해 유지되는 파라미터입니다. 현재 구현은 이진화 기반 블롭 탐지 임계값을 사용합니다.
-   **`show_camera_windows`** (bool, 기본값: `True`)
    -   카메라 영상 창 표시 여부입니다.
-   **`show_control_windows`**, **`show_mask_windows`** (bool, 기본값: `True`)
    -   두 모드 모두에서 코드가 `False`로 강제 설정합니다. 향후 확장을 위해 파라미터가 남아 있습니다.
-   **`yolo_model_path`** (string, 기본값: `models/yolov10n_lightonly_251002.pt`)
    -   YOLO 모드에서 사용할 모델 파일 경로입니다. 상대 경로는 스크립트 디렉터리를 기준으로 해석합니다.
-   **`yolo_confidence_threshold`** (double, 기본값: `0.5`)
    -   YOLO 탐지를 채택할 최소 신뢰도입니다.
-   **`debug_mode`** (bool, 기본값: `True`)
    -   참이면 `/green` 서비스 호출을 건너뛰고 시각화·로그만 수행합니다. 실제 미션 연동 시 `False`로 설정하세요.

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
# Manual 모드 (기본값, debug_mode=True 상태로 실행)
ros2 run yolo_ros detect_traffic_light

# YOLO 모드 (모델 경로/신뢰도 조정 가능)
ros2 run yolo_ros detect_traffic_light --ros-args -p roi_mode:=yolo -p yolo_model_path:=<모델 경로> -p yolo_confidence_threshold:=0.6

# 실제 /green 서비스 호출 활성화
ros2 run yolo_ros detect_traffic_light --ros-args -p debug_mode:=false
```

YOLO 모드에서는 Ultralytics 모델을 불러오므로 패키지가 설치되어 있어야 합니다. `debug_mode`를 끄지 않으면 초록불 감지 후에도 `/green` 서비스가 호출되지 않습니다.

**3. 결과 확인**

`debug_mode:=false`로 실행한 `detect_traffic_light` 노드가 초록불을 감지하면, 터미널 2의 서비스 상태가 `Succeeded`로 변경된 후 노드가 종료됩니다. 동시에, 터미널 1의 `test_mission_node`는 "🚀 초록불 확인! 미션을 수행합니다." 메시지를 출력하기 시작합니다.

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
    -   `/usb_cam_1/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)

### `test_mission_node.py` (예시 제어 노드)

-   **제공하는 서비스**: `/green` (`std_srvs/srv/Trigger`)
