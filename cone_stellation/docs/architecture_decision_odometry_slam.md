# 아키텍처 결정: 하이브리드 SLAM 시스템의 상세 분석

## 결정 날짜: 2025-07-21 (한국어로 재작성)

## 수정된 전제 조건
- **주행 속도**: 최대 60km/h, 평균 30km/h (레이싱이 아닌 일반 자율주행)
- **센서 주사율**:
  - IMU: 100Hz
  - RTK GPS: 8Hz
  - LiDAR 콘 검출: 19-20Hz

## 핵심 질문과 답변

### 1. RTK GPS를 통한 절대 좌표 표현
**질문**: RTK GPS factor를 추가하면 상대위치 SLAM을 절대좌표(WGS84→UTM)로 표현 가능한가?

**답변**: 네, 가능합니다.
- GPS factor를 추가하면 SLAM의 map 프레임이 글로벌 좌표계(예: UTM)에 고정됩니다
- 이를 통해 모든 콘 랜드마크와 차량 위치를 절대 좌표로 표현 가능
- RTK Fix (status 2)가 아니어도, 일반 GPS (status 0)로도 대략적인 절대 위치 제공 가능

### 2. 하이브리드 접근법의 논리적 갭 해결
**질문**: SLAM이 EKF odometry를 입력으로 받지 않는데 어떻게 drift를 수정하나?

**핵심 이해**: `map->odom` 변환은 정적 오프셋이 아니라 **동적 보정값**입니다!

## map->odom 변환의 실제 작동 원리

### 단계별 설명

1. **EKF의 역할 (100Hz)**
   ```
   입력: IMU + GPS
   출력: odom->base_link 변환 (고주파수, 부드러운 오도메트리)
   특징: 시간이 지나면서 drift 누적
   ```

2. **SLAM의 역할 (19-20Hz)**
   ```
   입력: IMU + GPS + Cones
   출력: map->base_link 변환 (더 정확한 포즈 추정)
   특징: 콘 관측을 통한 drift 보정, 루프 클로저 가능
   ```

3. **map->odom 계산 방법**
   ```cpp
   // SLAM은 직접 계산한다:
   Eigen::Isometry3d map_to_base = slam.getCurrentPose();  // SLAM의 추정
   
   // tf를 통해 현재 EKF의 추정을 얻는다:
   Eigen::Isometry3d odom_to_base = tf_buffer.lookupTransform("base_link", "odom");
   
   // drift 보정 계산:
   Eigen::Isometry3d map_to_odom = map_to_base * odom_to_base.inverse();
   
   // 이를 tf로 broadcast
   tf_broadcaster.sendTransform(map_to_odom);
   ```

### 왜 이 방식이 작동하는가?

**핵심 통찰**: SLAM은 EKF의 출력을 직접 받지 않아도, **tf를 통해 간접적으로 알 수 있습니다**.

1. **EKF는 "낙관적" 추정기**
   - IMU bias와 GPS 노이즈로 인한 오차 누적
   - 시간이 지나면서 odom 프레임에 drift 발생

2. **SLAM은 "ground truth"에 가까운 추정**
   - 콘 관측을 통한 추가 제약
   - 루프 클로저 가능
   - GPS로 글로벌 앵커링

3. **map->odom은 둘 사이의 차이를 보정**
   - SLAM이 자신의 추정(map->base_link)과 EKF의 추정을 비교
   - 그 차이가 바로 drift 보정값

## 실제 데이터 흐름 예시

### 시나리오: 10초간 주행 후
```
시간 t=10초:
- EKF 추정: 차량이 (100, 0)에 있다고 생각 (drift 포함)
- SLAM 추정: 콘 관측 기반으로 실제로는 (98, 0.5)에 있음
- map->odom = (-2, 0.5) 변환으로 보정

제어 시스템:
1. EKF에서 odom 좌표 (100, 0) 수신 (100Hz)
2. tf로 map 좌표로 변환: (100, 0) + (-2, 0.5) = (98, 0.5)
3. 정확한 위치 기반으로 제어 수행
```

## 구체적인 구현 세부사항

### ConeSTELLATION의 하이브리드 구조

```yaml
# 센서 입력
IMU (100Hz) ─┬─→ EKF ─→ odom->base_link (100Hz) ─→ 제어 시스템
             │
GPS (8Hz) ───┤
             │
             └─→ SLAM ─→ map->base_link (20Hz)
                   │
Cones (20Hz) ──────┘
                   ↓
              map->odom 보정값 계산 및 발행
```

### DriftCorrectionManager의 실제 동작
```cpp
void DriftCorrectionManager::update(const EstimationFrame::Ptr& frame) {
    // 1. SLAM의 현재 포즈 (map 프레임)
    Eigen::Isometry3d T_map_base = frame->T_world_sensor;
    
    // 2. 현재 시간의 EKF 오도메트리 조회
    geometry_msgs::msg::TransformStamped odom_to_base;
    try {
        odom_to_base = tf_buffer_->lookupTransform(
            "base_link", "odom", frame->header.stamp);
    } catch (...) {
        return;  // EKF 데이터 없으면 스킵
    }
    
    // 3. map->odom 계산
    Eigen::Isometry3d T_odom_base = tf2::transformToEigen(odom_to_base);
    Eigen::Isometry3d T_map_odom = T_map_base * T_odom_base.inverse();
    
    // 4. 부드러운 보간 (급격한 변화 방지)
    if (!last_map_to_odom_initialized_) {
        last_map_to_odom_ = T_map_odom;
        last_map_to_odom_initialized_ = true;
    } else {
        // 선형 보간으로 부드럽게
        double alpha = 0.1;  // 보간 계수
        T_map_odom = interpolate(last_map_to_odom_, T_map_odom, alpha);
        last_map_to_odom_ = T_map_odom;
    }
    
    // 5. tf 발행
    publishTransform(T_map_odom, "map", "odom", frame->header.stamp);
}
```

## 왜 하이브리드 접근법이 적합한가?

### 30-60km/h 주행에서의 장점

1. **안정적인 제어**
   - 30km/h = 8.3m/s → 10ms 지연 = 8.3cm
   - 60km/h = 16.7m/s → 10ms 지연 = 16.7cm
   - EKF의 일정한 저지연(2-5ms)으로 안정적 제어 가능

2. **계산 효율성**
   - EKF와 SLAM을 분리하여 각각 최적화 가능
   - SLAM은 20Hz로도 충분 (콘 검출 주기와 동일)
   - CPU 자원 효율적 사용

3. **고장 강건성**
   - SLAM 실패 시에도 EKF로 계속 주행 가능
   - GPS 끊김 시 IMU 기반 추정 지속
   - 콘 미검출 시에도 오도메트리 유지

### 오해 해소: "SLAM이 EKF를 모르는데?"

**잘못된 이해**: SLAM이 EKF 출력을 직접 받아야 drift 보정 가능

**올바른 이해**: 
- SLAM은 tf를 통해 현재 `odom->base_link`를 언제든 조회 가능
- 자신의 `map->base_link` 추정과 비교하여 drift 계산
- 이 차이를 `map->odom`으로 발행하여 전체 시스템 보정

## 실제 구현 상태

### 현재 ConeSTELLATION 아키텍처

1. **외부 IMU+GPS EKF** (구현 예정)
   - 100Hz 오도메트리 제공
   - robot_localization 패키지 사용 예정
   - 설정 파일: `config/ekf_config.yaml`

2. **ConeSTELLATION SLAM** (구현 완료)
   - 콘 기반 맵핑에 집중
   - DriftCorrectionManager로 map->odom 발행
   - 20Hz 주기로 동작
   - Inter-landmark factors로 정확도 향상

3. **선택적 강결합 요소** (향후 추가 가능)
   ```yaml
   # slam_config.yaml에 추가 가능
   mapping:
     use_imu_preintegration: true  # IMU factor 추가
     use_gps_factors: true         # GPS 제약 추가
     # 하지만 여전히 실시간 오도메트리는 EKF 사용!
   ```

### 검증된 이점
1. **낮은 지연시간**: 30-60km/h 주행에 충분
2. **모듈성**: 각 컴포넌트 독립적 개발/디버깅
3. **강건성**: 부분 고장에도 주행 지속
4. **GLIM 검증**: 실제 GLIM도 이 구조 사용

## 요약

### 핵심 이해
1. **map->odom은 동적 보정값**: SLAM이 계산한 실제 위치와 EKF의 추정 위치 차이
2. **SLAM은 tf를 통해 EKF 출력 인지**: 직접 입력받지 않아도 됨
3. **하이브리드가 최적**: 낮은 지연시간 + 높은 정확도

### 다음 단계
1. robot_localization 패키지로 EKF 구현
2. 현재 SLAM에 선택적으로 IMU/GPS factor 추가
3. 실차 테스트를 통한 파라미터 튜닝

## 참고 자료
- GLIM 하이브리드 구조: `/home/user1/ROS2_Workspace/GLIM_ws/src/glim/`
- DriftCorrectionManager: `include/cone_stellation/util/drift_correction_manager.hpp`
- ROS2 tf2 문서: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html