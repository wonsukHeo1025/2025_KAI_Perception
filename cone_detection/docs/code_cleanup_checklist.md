# Code Cleanup Checklist - 제거 가능한 코드 목록

## 🔴 즉시 제거 가능 (Priority 1)
총 예상 제거 라인: **~700 라인**

### 1. Ouster 포맷 변환 코드
**위치:** `cone_detection_node.cpp` Lines 646-743
```cpp
if (publisher == pub_reconstructed_cones_cloud_) {
    // Ouster 형식으로 변환
    cloud_msg.height = 32;  // 32 channels
    cloud_msg.width = (cloud->size() + 31) / 32;
    // ... 100+ 라인의 복잡한 변환 로직
}
```
**제거 이유:** 
- 불필요한 복잡성
- 표준 `pcl::toROSMsg()`로 충분
- 특정 센서 모델에 과도하게 의존적

**대체 코드:**
```cpp
pcl::toROSMsg(*cloud, cloud_msg);
cloud_msg.header.frame_id = frame_id;
cloud_msg.header.stamp = timestamp;
```

### 2. 레거시 메시지 포맷 관련 코드
**위치:** 
- `cone_detection_node.cpp` Line 101 (퍼블리셔)
- `cone_detection_node.cpp` Lines 598-628 (sortCones 함수)
- `cone_detection_node.cpp` Lines 760-810 (publishArrayWithTimestamp 함수)

```cpp
// 제거할 퍼블리셔
cones_time_pub = this->create_publisher<custom_interface::msg::ModifiedFloat32MultiArray>("/sorted_cones_time", 10);

// 제거할 함수 전체
std::vector<std::vector<double>> OutlierFilter::sortCones(...) { }
void OutlierFilter::publishArrayWithTimestamp(...) { }
```
**제거 이유:** 
- 새로운 TrackedConeArray 포맷으로 대체됨
- 중복된 기능

### 3. 사용하지 않는 visualization 관련 코드
**위치:** `cone_detection_node.cpp` Lines 214-216, 231, 261
```cpp
// Visualization moved to separate node 주석들
// 이미 별도 노드로 분리되어 불필요
```

## 🟡 조건부 제거 가능 (Priority 2)
총 예상 제거 라인: **~300 라인**

### 1. 2단계 검증 시스템 (현재 비활성화)
**위치:**
- `cone_detection_node.cpp` Lines 1089-1246 (validateAndReconstructConesStage2)
- `cone_detection_node.cpp` Lines 1247-1305 (reconstructPointsAroundCones)
- 관련 파라미터 및 멤버 변수

**결정 필요:**
1. 활성화하여 테스트 → 유용하면 유지, 개선
2. 유용하지 않으면 → 완전 제거

**제거 시 영향:**
- 코드베이스 30% 단순화
- 유지보수 부담 감소
- 메모리 사용량 감소

### 2. 지면 평면 계수 관련 미사용 코드
**위치:** `cone_detection_node.cpp` Lines 1007-1028 (validateConesFinalChecks 내부)
```cpp
// 주석 처리된 지면 계수 사용 코드
// Eigen::Vector3f ground_normal(plane_coefs->values[0], ...
```
**제거 이유:** 실제로 사용되지 않음

## 🟢 리팩토링 후 제거 (Priority 3)
총 예상 제거 라인: **~200 라인**

### 1. 중복된 파라미터 로딩 코드
**위치:** `cone_detection_node.cpp` Lines 13-98
```cpp
// 반복적인 파라미터 선언 패턴
for (const auto& [name, value_ptr] : str_params) {
    this->declare_parameter(name, *value_ptr);
    this->get_parameter(name, *value_ptr);
    RCLCPP_INFO(this->get_logger(), "  %s: %s", name.c_str(), value_ptr->c_str());
}
```

**리팩토링 방안:**
```cpp
template<typename T>
void loadParameter(const std::string& name, T& value) {
    this->declare_parameter(name, value);
    this->get_parameter(name, value);
    logParameter(name, value);
}
```

### 2. 하드코딩된 좌표 변환
**위치:** `cone_detection_node.cpp` Lines 508-527
```cpp
void OutlierFilter::lidarToSensorTransform(Cloud::Ptr &cloud) {
    // 하드코딩된 변환
    transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    transform.translation() << 0.0f, 0.0f, 0.038195f;
}
```

**TF2로 대체:**
```cpp
// TF2 사용
geometry_msgs::msg::TransformStamped transform;
transform = tf_buffer_->lookupTransform("os_sensor", "os_lidar", 
                                        tf2::TimePointZero);
```

## 제거 작업 체크리스트

### Phase 1: 즉시 제거 (1-2일)
- [ ] Ouster 포맷 변환 코드 제거 (Lines 646-743)
- [ ] sortCones() 함수 제거
- [ ] publishArrayWithTimestamp() 함수 제거
- [ ] 레거시 퍼블리셔 제거
- [ ] visualization 관련 주석 제거

### Phase 2: 검증 후 제거 (3-5일)
- [ ] 2단계 검증 시스템 테스트
- [ ] 사용 여부 결정
- [ ] 불필요 시 완전 제거
- [ ] 관련 파라미터 정리

### Phase 3: 리팩토링과 함께 제거 (1주)
- [ ] 파라미터 로딩 템플릿화
- [ ] TF2 변환 구현
- [ ] 하드코딩된 값 제거

## 제거 후 검증 항목

### 1. 기능 테스트
```bash
# 빌드 확인
colcon build --packages-select cone_detection

# 런타임 테스트
ros2 launch cone_detection cone_detection_launch.py

# 토픽 확인
ros2 topic list | grep cone
ros2 topic echo /cone/lidar
```

### 2. 성능 비교
```bash
# 제거 전 성능 측정
ros2 topic hz /cone/lidar

# 제거 후 성능 측정
ros2 topic hz /cone/lidar
```

### 3. 메모리 사용량
```bash
# 프로세스 메모리 모니터링
top -p $(pgrep -f cone_detection)
```

## 예상 결과

### 코드 메트릭 개선
| 메트릭 | 현재 | 제거 후 | 개선율 |
|--------|------|---------|--------|
| 총 라인 수 | 1325 | ~525 | -60% |
| 복잡도 | High | Medium | -40% |
| 파일 크기 | 45KB | ~20KB | -55% |

### 유지보수성 개선
- 코드 가독성: **크게 향상**
- 디버깅 용이성: **향상**
- 새 기능 추가: **더 쉬워짐**
- 테스트 작성: **단순화**

## 주의사항

### 제거 전 백업
```bash
# Git 브랜치 생성
git checkout -b cleanup/remove-unused-code

# 태그 생성
git tag before-cleanup
```

### 단계별 커밋
```bash
# 각 제거 작업별 커밋
git commit -m "Remove Ouster format conversion code"
git commit -m "Remove legacy message format and sortCones"
git commit -m "Remove Stage2 validation (unused)"
```

### 롤백 계획
```bash
# 문제 발생 시
git revert HEAD
# 또는
git checkout before-cleanup
```

## 결론

코드 정리를 통해 전체 코드베이스의 60%를 제거할 수 있으며, 이는 유지보수성과 가독성을 크게 향상시킬 것입니다. 우선순위에 따라 단계적으로 진행하되, 각 단계마다 충분한 테스트를 수행해야 합니다.