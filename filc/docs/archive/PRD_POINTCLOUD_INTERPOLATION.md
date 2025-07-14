# PRD: Ouster OS1-32 포인트클라우드 수직 보간 시스템

## 1. 제품 개요

### 1.1 목적
Ouster OS1-32 LiDAR의 32채널 포인트클라우드를 수직 방향으로 보간하여 128채널의 고밀도 포인트클라우드를 생성하는 ROS2 노드 개발

### 1.2 배경
- **현재 상황**: 32채널 LiDAR의 수직 해상도 한계로 인한 sparse 포인트클라우드
- **문제점**: 기존 구현은 2D 이미지 보간 방식을 사용하여 3D 기하학적 왜곡 발생
- **해결 방안**: 구면 좌표계 기반의 정확한 3D 보간 시스템 구축

### 1.3 핵심 요구사항
- 32×1024 → 128×1024 organized 포인트클라우드 변환
- 실시간 처리 (≥10Hz)
- 기하학적 정확도 유지
- 기존 카메라 융합 시스템과의 호환성

## 2. 기능 요구사항

### 2.1 입력 데이터 처리

#### 2.1.1 포인트클라우드 수신
```
요구사항 ID: FR-001
설명: Ouster OS1-32 organized 포인트클라우드 수신
입력: sensor_msgs/PointCloud2 (32×1024, XYZI)
검증: 
- 입력 포인트클라우드 차원 확인 (height=32, width=1024)
- 포인트 필드 확인 (x, y, z, intensity)
- 타임스탬프 유효성 검증
```

#### 2.1.2 Range/Intensity 이미지 수신 (선택적)
```
요구사항 ID: FR-002
설명: 보조 데이터로 range/intensity 이미지 활용
입력: 
- sensor_msgs/Image (range_image: 32×1024, float32)
- sensor_msgs/Image (intensity_image: 32×1024, float32)
검증: 이미지 차원 및 타입 확인
```

### 2.2 보간 처리

#### 2.2.1 구면 좌표 변환
```
요구사항 ID: FR-003
설명: 직교 좌표(XYZ)를 구면 좌표(range, azimuth, altitude)로 변환
처리:
1. 각 포인트의 range 계산: sqrt(x² + y² + z²)
2. 방위각 계산: atan2(y, x)
3. 고도각 역산: asin(z / range)
출력: 32×1024 구조의 range, azimuth, altitude 배열
```

#### 2.2.2 수직 보간 수행
```
요구사항 ID: FR-004
설명: 각 원본 채널 사이에 3개의 새로운 채널 생성
처리:
1. 원본 채널 i와 i+1 사이에 3개 보간 채널 생성
2. 고도각 보간:
   - 보간 위치: 0.25, 0.5, 0.75
   - 방법: 큐빅 스플라인 보간
3. Range 값 보간:
   - 동일 열의 상하 채널 range 값 사용
   - 불연속성 검출 시 nearest neighbor
   - 연속성 있을 시 가중 평균
4. 방위각: 동일 열이므로 그대로 유지

출력: 128×1024 구조의 보간된 구면 좌표
```

#### 2.2.3 3D 좌표 변환
```
요구사항 ID: FR-005
설명: 보간된 구면 좌표를 Ouster 공식에 따라 3D 직교 좌표로 변환
처리:
1. Ouster 좌표 변환 공식 적용:
   - θ_encoder = 2π × (1 - measurement_id / 1024)
   - beam_origin_offset = 15.806mm 적용
   - x = (range - offset) × cos(θ) × cos(φ)
   - y = (range - offset) × sin(θ) × cos(φ)
   - z = (range - offset) × sin(φ)
2. 유효성 검증:
   - NaN/Inf 체크
   - 최소/최대 거리 필터링

출력: 128×1024 XYZ 포인트클라우드
```

### 2.3 출력 생성

#### 2.3.1 Organized PointCloud2 생성
```
요구사항 ID: FR-006
설명: 보간된 데이터를 ROS2 PointCloud2 메시지로 변환
처리:
1. 헤더 설정:
   - height = 128
   - width = 1024
   - is_dense = false
   - is_bigendian = false
2. 포인트 필드 정의:
   - x, y, z (float32)
   - intensity (float32)
   - ring (uint16) - 채널 번호
   - timestamp (float32) - 선택적
3. 데이터 패킹

출력: sensor_msgs/PointCloud2
```

#### 2.3.2 메타데이터 추가
```
요구사항 ID: FR-007
설명: 보간 정보를 포함한 메타데이터 추가
내용:
- 원본 채널 플래그 (is_original)
- 보간 방법 (interpolation_method)
- 품질 지표 (confidence)
```

## 3. 비기능 요구사항

### 3.1 성능 요구사항
```
요구사항 ID: NFR-001
- 처리 속도: ≥10Hz (100ms 이내 처리)
- 메모리 사용: ≤500MB
- CPU 사용률: ≤80% (4코어 기준)
```

### 3.2 정확도 요구사항
```
요구사항 ID: NFR-002
- 기하학적 오차: ≤1cm @ 10m 거리
- 고도각 정확도: ≤0.1°
- 포인트 위치 일관성: 원본 포인트 위치 불변
```

### 3.3 안정성 요구사항
```
요구사항 ID: NFR-003
- 데이터 손실: 0%
- 예외 처리: 모든 에러 상황 graceful handling
- 복구 시간: ≤1초
```

## 4. 기술 사양

### 4.1 센서 파라미터
```yaml
ouster_os1_32:
  channels: 32
  horizontal_resolution: 1024
  vertical_fov: [-16.611, -0.264]  # degrees
  lidar_origin_to_beam_origin: 15.806  # mm
  beam_altitude_angles:  # 32개 값 (degrees)
    - -16.611
    - -16.084
    # ... (중략)
    - -0.791
    - -0.264
```

### 4.2 보간 알고리즘 상세

#### 4.2.1 고도각 보간 공식
```cpp
// 큐빅 스플라인 보간
double interpolateAltitude(int ch_below, int ch_above, double t) {
    // t: 보간 위치 (0.25, 0.5, 0.75)
    // Catmull-Rom spline 사용
    int ch0 = max(0, ch_below - 1);
    int ch1 = ch_below;
    int ch2 = ch_above;
    int ch3 = min(31, ch_above + 1);
    
    double alt0 = beam_altitude_angles[ch0];
    double alt1 = beam_altitude_angles[ch1];
    double alt2 = beam_altitude_angles[ch2];
    double alt3 = beam_altitude_angles[ch3];
    
    // Catmull-Rom 보간 공식
    double t2 = t * t;
    double t3 = t2 * t;
    
    return 0.5 * ((2 * alt1) +
                  (-alt0 + alt2) * t +
                  (2*alt0 - 5*alt1 + 4*alt2 - alt3) * t2 +
                  (-alt0 + 3*alt1 - 3*alt2 + alt3) * t3);
}
```

#### 4.2.2 Range 보간 전략
```cpp
double interpolateRange(double range_below, double range_above, 
                       double t, double discontinuity_threshold = 0.5) {
    double range_diff = abs(range_above - range_below);
    
    if (range_diff > discontinuity_threshold) {
        // 불연속성 감지: nearest neighbor
        return (t < 0.5) ? range_below : range_above;
    } else {
        // 연속적인 경우: 가중 선형 보간
        return range_below * (1.0 - t) + range_above * t;
    }
}
```

### 4.3 데이터 구조

#### 4.3.1 내부 데이터 구조
```cpp
struct SphericalPoint {
    float range;      // 미터
    float azimuth;    // 라디안
    float altitude;   // 라디안
    float intensity;  // 0-255 정규화
    bool is_valid;    // 유효성 플래그
};

struct InterpolatedCloud {
    std::vector<std::vector<SphericalPoint>> spherical_points;  // [128][1024]
    std::vector<std::vector<geometry_msgs::Point32>> cartesian_points;  // [128][1024]
    std::vector<bool> is_original_channel;  // [128] - true if original
    ros::Time timestamp;
};
```

## 5. 구현 단계

### Phase 1: 기본 구조 구현 (3일)
- [ ] 입력 데이터 파서 구현
- [ ] 구면 좌표 변환 함수 구현
- [ ] 기본 데이터 구조 정의

### Phase 2: 보간 엔진 구현 (5일)
- [ ] 고도각 큐빅 스플라인 보간 구현
- [ ] Range 적응적 보간 구현
- [ ] 보간 검증 로직 구현

### Phase 3: 좌표 변환 구현 (3일)
- [ ] Ouster 공식 좌표 변환 구현
- [ ] 변환 정확도 검증
- [ ] 최적화 (lookup table, 병렬화)

### Phase 4: ROS2 통합 (3일)
- [ ] PointCloud2 메시지 생성
- [ ] 퍼블리셔/서브스크라이버 구현
- [ ] 파라미터 서버 통합

### Phase 5: 테스트 및 검증 (3일)
- [ ] 단위 테스트 작성
- [ ] 통합 테스트
- [ ] 성능 벤치마크
- [ ] 시각화 도구 개발

## 6. 검증 계획

### 6.1 단위 테스트
- 구면 좌표 변환 정확도
- 보간 알고리즘 정확도
- 좌표 변환 정확도

### 6.2 통합 테스트
- End-to-end 처리 파이프라인
- 실시간 성능 측정
- 메모리 사용량 모니터링

### 6.3 검증 메트릭
- 보간된 포인트의 기하학적 일관성
- 원본 포인트 위치 보존
- 처리 지연시간
- 시각적 품질 (아티팩트, 왜곡)

## 7. 리스크 및 완화 방안

### 7.1 기술적 리스크
- **리스크**: 실시간 처리 성능 미달
- **완화**: 병렬화, SIMD 최적화, GPU 가속 검토

### 7.2 정확도 리스크
- **리스크**: 보간된 포인트의 기하학적 오류
- **완화**: 다양한 보간 방법 비교, 적응적 알고리즘 적용

### 7.3 호환성 리스크
- **리스크**: 기존 시스템과의 통합 문제
- **완화**: 하위 호환성 유지, 점진적 마이그레이션 경로 제공

## 8. 성공 기준

1. **기능적 성공**
   - 32×1024 → 128×1024 변환 성공
   - 모든 보간된 포인트가 올바른 3D 위치에 생성

2. **성능적 성공**
   - 10Hz 이상 실시간 처리
   - CPU 사용률 80% 이하

3. **품질적 성공**
   - 시각적 아티팩트 없음
   - 원본 포인트클라우드의 기하학적 특성 유지
   - 카메라 융합 시 정확한 색상 매핑