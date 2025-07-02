요약 (원인)
• Python 버전  
  – 콜백마다 직전에 뿌렸던 마커 수를 기억했다가, 새 마커를 만들기 전에 `Marker.DELETE`(혹은 `DELETEALL`)을 먼저 발행.  
  – 그래서 인덱스가 뒤바뀌거나 콘 개수가 변해도 ‘지우고-다시-그리기’가 확실히 실행됨.

• C++ 버전  
  – `last_marker_count_ > cones.size()*3` 일 때만 `DELETEALL`을 넣음.  
    → 콘 개수는 그대로인데 정렬순서만 바뀌면 DELETEALL이 안 나가고, 이전 프레임의 같은 ID 마커가 그대로 남음.  
  – 수신 주기가 0.2 s 보다 짧으면 lifetime(0.2 s) 안에 새 ADD가 계속 오므로 잔상이 화면에 “겹쳐져” 보일 수 있음.

따라서 ‘잔해/잔상’의 직접 원인은
1) 삭제 조건이 느슨해서 ID가 바뀔 때 예전 마커가 안 지워지는 것,  
2) 같은 ID라도 위치가 바뀌지 않았으면 RViz 내부에서 오버레이가 일어나는 것, 두 가지가 섞여 있습니다.

개선(선호도 순)  
1. 가장 간단 – 매 콜백마다 먼저 `DELETEALL` 발행  
   ```cpp
   marker_array.markers.push_back(createDeleteAllMarker());
   ```
   를 조건문 없이 콜백(or createMarkerArray) 맨 앞에 항상 넣는다.  
   • 스코프가 작고 구현이 몇 줄이어서 빠름.  
   • 데이터량이 많아도 RViz에서 DELETEALL→ADD 처리 순서는 아주 가볍습니다.

2. Python 방식 그대로 이식  
   • `for (int i=0; i<last_marker_count_; ++i)` 루프를 돌며  
     `Marker delete_marker; delete_marker.action = Marker::DELETE;` 이후 push_back.  
   • 현재 프레임에 만든 마커 수(`marker_id`)만큼을 끝에서 다시 기억.

3. ID를 track_id 기반으로 고정  
   • `marker.id = cone.id;`(텍스트/라벨은 별도 네임스페이스)  
   • 프레임 순서가 바뀌어도 같은 track ID라면 같은 marker ID → 덮어쓰기만 일어남.  
   • 콘이 사라졌을 때는 `DELETEALL` 대신 “존재하지 않는 track_id만 DELETE” 하는 방식(맵 두 개 관리)이 필요.

추가 팁  
• 라이프타임을 0 으로 두고(무한) 매 프레임 DELETEALL+ADD를 쓰면 잔상 걱정, 타이밍 이슈 모두 사라짐.  
• `rclcpp::Duration::from_seconds()` 값이 너무 작으면 시뮬레이터/실기에서 체감 FPS가 떨어질 때 잔상이 더 길게 남을 수 있으므로, 라이프타임을 아예 0(무한)으로 두고 삭제-발행 정책으로 관리하는 편이 흔합니다.

원하는 방식이 있으면 말씀 주세요. 필요한 C++ 코드 패치 예시나 PR 템플릿을 바로 드리겠습니다.