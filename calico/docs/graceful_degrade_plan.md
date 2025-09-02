# CALICO Graceful Degrade Plan

이 문서는 CALICO가 카메라 결손 상황에서도 “자연스럽게, 당연히” 동작하도록 하는 Graceful Degrade 설계를 정리합니다. 핵심은 LiDAR만으로도 `/cone/fused`를 안정적으로 발행하되, 카메라(YOLO) 정보가 있을 때에만 LiDAR의 Unknown 클래스를 색상 클래스로 덮어씌우는 것입니다.

## 목표
- LiDAR 단독 입력만으로도 `/cone/fused`가 지속 발행되고, 모든 색상은 `Unknown`으로 유지
- 카메라 검출이 일부/전부 도착하면 매칭된 LiDAR 항목의 색상만 업데이트(Unknown → 색상)
- 카메라 입력이 중간에 끊기거나 지연되면 자동으로 Unknown 유지(추가 파라미터 불필요)

## 인터페이스 및 토픽
- LiDAR 입력(두 경로 지원)
  - IoU 경로: `/cone/lidar/box` (`vision_msgs/BoundingBox3DArray`)
    - 현재 런치(`calico_full.launch.py`)에서는 `multi_iou_fusion_node`를 사용하며 코드가 `cones_topic`을 강제로 `/cone/lidar/box`로 오버라이드
  - Array 경로: `/cone/lidar` (`custom_interface/TrackedConeArray`)
    - `multi_camera_fusion_node`에서 사용되는 경로
- 카메라 입력: `/camera_1/detections`, `/camera_2/detections` (`yolo_msgs/DetectionArray`)
- 출력: `/cone/fused` (`custom_interface/TrackedConeArray`)

## 메시지 스키마 요약
- `custom_interface/TrackedConeArray`
  - `std_msgs/Header header`
  - `TrackedCone[] cones`
- `custom_interface/TrackedCone`
  - `int32 track_id`
  - `geometry_msgs/Point position` (os_sensor 좌표계)
  - `string color` (예: `"blue cone"`, `"yellow cone"`, `"red cone"`, 또는 `"Unknown"`)

참고: YOLO 클래스명은 `MessageConverter::mapClassToColor()`로 표준화됩니다. 공백/언더스코어 표기 모두 허용되며 최종 문자열은 Python과 호환되도록 유지합니다. 미매핑 시 `"Unknown"`을 반환합니다.

## 동작 원칙(Degrade 정책)
1) 기본값 Unknown: LiDAR에서 생성된 항목의 `color`는 기본 `"Unknown"`으로 설정
2) 덮어쓰기 정책: 카메라 검출과 매칭된 항목에 한해 `Unknown → {blue,yellow,red,...}`로만 업데이트
   - 이미 색상이 있는 항목을 `Unknown`으로 덮어쓰지 않음(다운그레이드 금지)
3) 입력 가용성에 따른 처리
   - 카메라 둘 다 결손: LiDAR pass-through로 모두 `Unknown`
   - 한 대만 가용: 해당 카메라 검출만 이용해 매칭/업데이트, 나머지는 `Unknown`
   - 실행 중 결손/복귀: 가용 입력만 자동 사용(추가 설정 불필요)
4) 시간 동기화: ApproximateTime 또는 도착 기반 동기화로 “주기적 pass-through” 보장
   - IoU 노드: `time_sync_mode`(header|arrival_ros|arrival_wall)와 `arrival_slop`으로 제어
   - 필요 시 `override_fused_stamp_now`로 타임스탬프 now() 덮어쓰기

## 구현 경로 확인(현 코드 기준)
- IoU 경로(`multi_iou_fusion_node.cpp`)
  - LiDAR: `/cone/lidar/box`를 강제 사용
  - 카메라: `/camera_1/detections`, `/camera_2/detections`
  - 각 카메라로 투영된 박스 간 IoU 기반 Hungarian 매칭 → 카메라별 클래스 리스트 병합 → LiDAR 순서대로 클래스 배열 생성
  - TrackedConeArray로 변환하여 `/cone/fused` 게시(헤더는 LiDAR에서 유지, 옵션으로 now 덮어쓰기)
  - 미매칭/미검출인 경우 해당 인덱스는 `"Unknown"` 유지
  - 단, 현재 구현은 “발행 조건”이 엄격합니다:
    - Header 동기화 모드: 모든 입력(라이다+두 카메라)이 동기화되어야 콜백이 실행 → 카메라 결손 시 `/cone/fused` 미발행
    - Arrival 동기화 모드(기본): 내부 `tryProcessByArrival()`가 모든 카메라의 최신 메시지를 요구 → 한 대라도 결손이면 `/cone/fused` 미발행
- Array 경로(`multi_camera_fusion_node.cpp` + `fusion/multi_camera_fusion.cpp`)
  - LiDAR: `/cone/lidar`(TrackedConeArray)를 내부 `utils::Cone`으로 변환
  - 카메라 검출을 `MessageConverter::fromDetectionArray()`로 내부 표현으로 변환
  - Hungarian 매칭 결과에 따라 `fused_cones[i].color`를 `"Unknown"`에서 매핑된 색상으로만 업데이트
  - 결과를 TrackedConeArray로 변환해 게시
  - 메시지 필터 기반 동기화만 사용하여, 카메라 결손 시 콜백 자체가 발생하지 않음(패스스루 경로 부재)

요약: “Unknown 유지” 자체는 매칭/병합 로직 레벨에서 보장되지만, “카메라 결손 시에도 `/cone/fused`를 발행”하는 패스스루 경로는 현재 미구현입니다. 따라서 라이다만 들어오는 상황에서는 `/cone/fused`가 발행되지 않습니다.

## 상세 설계 포인트
- 클래스 매핑: `MessageConverter::mapClassToColor()` 사용
  - 허용 예: `"blue cone"`, `"Blue_Cone"`, `"blue_cone"` → 표준화 `"blue cone"`
  - 미매핑: `"Unknown"` 반환(대문자 U)
- 병합 규칙(IoU 경로): 카메라별 결과를 간단 투표로 병합(Unknown 제외 최다 득표 우선)
- 시각화: RViz 마커는 `Unknown` 회색, `Yellow/Red/...`는 지정 색으로 렌더링됨
- 성능: Degrade 시에도 추가 연산 없음(Unknown 유지), 안정적인 주기 유지

## 엣지 케이스 및 안전장치
- LiDAR 미발행: 융합/출력 중단(로그 경고). LiDAR는 필수 입력
- 클래스 충돌: 다수 카메라가 상이한 클래스를 제시하면 득표 우선. 전부 Unknown이면 Unknown 유지
- 시간 불일치: `sync_slop`/`arrival_slop`으로 허용 범위 조정. 과도할 경우 Unknown 비율 증가(안전한 폴백)

## 검증 시나리오
1) 정상(완전 입력)
   - 입력: `/cone/lidar[/box]`, `/camera_1/detections`, `/camera_2/detections`
   - 기대: `/cone/fused`에서 Unknown이 색상으로 대체(매칭된 항목)
2) 카메라 1만 활성
   - 입력: `/cone/lidar[/box]`, `/camera_1/detections`(OK), `/camera_2/detections`(결손)
   - 기대: Cam1로 매칭된 항목만 색상, 나머지는 Unknown
3) 전부 결손(카메라)
   - 입력: `/cone/lidar[/box]`만 발행
   - 기대: `/cone/fused`는 LiDAR pass-through로 전부 Unknown
4) 실행 중 드롭/복귀
   - 입력: 카메라 토픽이 중간에 끊겼다가 재개
   - 기대: 끊긴 동안 Unknown 유지, 복귀 즉시 색상 업데이트 재개

간단 점검 명령
```
ros2 topic hz /cone/lidar/box
ros2 topic hz /camera_1/detections
ros2 topic hz /camera_2/detections
ros2 topic hz /cone/fused
ros2 topic echo /cone/fused --once
```

## 변경 필요 여부(결론)
- 코드 수정 필요: 현재는 “카메라 결손 시 `/cone/fused` 미발행” 상태입니다. Graceful Degrade의 핵심인 “LiDAR pass-through(Unknown 유지) 지속 발행”을 위해 아래 변경이 필요합니다.

코드 변경 계획(요약)
- 대상: `calico/src/nodes/multi_iou_fusion_node.cpp`
  - Arrival 모드(`time_sync_mode=arrival_ros|arrival_wall`)
    - `tryProcessByArrival()` 완화: 모든 카메라가 없어도 라이다만으로 처리
      - 로직: LiDAR 수신 시
        1) 카메라별로 최근 메시지가 있으면 사용, 없으면 빈 `DetectionArray`를 대체 투입
        2) 디버그 이미지는 선택 사항. 없으면 `images` 비워서 `processFusion()` 호출(시각화 생략)
        3) 결과 `/cone/fused` 발행(미매칭은 자동 Unknown)
    - 장점: 한 대만 살아있어도 부분 융합 가능, 전부 결손이면 Unknown 패스스루 발행
  - Header 모드(`time_sync_mode=header`)
    - 선택 1: 유지(권장 모드 아님) — 문서에서 Arrival 모드 사용을 권장
    - 선택 2: 라이다 전용 보조 구독 추가 — 동기화 실패 시 타임아웃 기반 Unknown 패스스루 발행
- 대상(옵션): `calico/src/nodes/multi_camera_fusion_node.cpp`
  - 메시지 필터 외에 라이다 단독 경로 추가(구현 복잡도↑). 운영에서는 IoU 노드 사용을 기본으로 권장

파이프라인 영향
- `/cone/fused` 미발행 → UKF/시각화 모두 중단됨
  - UKF(`ukf_tracking_node.cpp`)는 `/cone/fused` 도착 시에만 `/cone/fused/ukf`를 발행
  - 시각화 노드도 `/cone/fused`/`/cone/fused/ukf` 구독 기반으로 동작
- 위 변경으로 라이다만 있어도 `/cone/fused`가 Unknown으로 계속 발행되어 이후 파이프라인이 유지됨

추가 고려사항
- 디버그 이미지 의존성 분리(현재 코드도 이미지 미필수 경로 존재) — 발행 차단 원인이 되지 않도록 유지
- 상태 로그에 “패스스루 발행 중(카메라 결손)” 카운터 추가(운영 가시성)

## 운영 가이드(요약)
- 완전 작동: `/cone/lidar[/box]` + `/camera_1/detections`, `/camera_2/detections` → 색상 업데이트
- 카메라 결손: Arrival 모드에서 Unknown 패스스루 발행(코드 변경 반영 후)
- 런치: 기본 `calico_full.launch.py`는 IoU 경로 사용(박스 기반). Degrade 보장을 위해 `time_sync_mode:=arrival_ros` 권장
