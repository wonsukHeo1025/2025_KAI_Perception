# filc - 포인트클라우드 보간 전면 재설계 계획

## 핵심 요약

현재 Ouster OS1-32 (32채널, 1024픽셀)의 포인트클라우드를 4배 보간(128채널)하는 과정에서 발생하는 문제점:
- **증상**: 보간된 포인트클라우드가 매우 작게 나오고, 위아래가 뒤바뀌며, 뒤틀려 있음
- **원인**: 2D 이미지 보간 방식을 3D 포인트클라우드에 부적절하게 적용
- **해결**: 구면 좌표계 기반 보간과 Ouster 공식 좌표 변환 적용

## 1. 현재 구현의 문제점 분석

### 1.1 근본적인 문제
현재 구현은 **이미지 보간 방식을 포인트클라우드에 그대로 적용**하여 다음과 같은 문제가 발생:

1. **기하학적 부정확성**
   - 2D 이미지 보간 후 3D 변환은 구면 좌표계의 비선형성을 무시
   - 고도각의 선형 보간이 실제 센서의 빔 패턴과 불일치
   - 4배 스케일(32→128채널)은 너무 높아 기하학적 왜곡 심화

2. **좌표 변환 오류**
   - 간소화된 3D 좌표 계산이 Ouster의 실제 좌표 변환과 불일치
   - 빔 원점 오프셋(15.806mm) 고려 부족
   - 방위각/고도각 매핑의 부정확성

3. **데이터 구조 문제**
   - Organized 포인트클라우드 생성 시 구조적 일관성 부족
   - 보간된 포인트와 원본 포인트 간의 공간적 관계 파괴

## 2. Ouster OS1-32 데이터 구조 이해

### 2.1 센서 특성
- **채널 수**: 32 (수직)
- **해상도**: 1024 픽셀/회전 (수평)
- **고도각 분포**: -16.611° ~ -0.264° (비균등 간격)
- **데이터 형식**: Organized 포인트클라우드 (32×1024 구조)

### 2.2 구면 좌표계 특성
- 포인트클라우드는 구면 투영으로 dense format 생성
- 각 포인트: (range, azimuth, altitude) → (x, y, z)
- 보간 시 구면 좌표계에서 수행해야 정확성 보장

## 3. 새로운 보간 아키텍처 설계

### 3.1 핵심 설계 원칙

1. **구면 좌표계 기반 보간**
   ```cpp
   // 기존: 이미지 보간 → 3D 변환
   // 개선: 3D 구면 좌표 → 보간 → 3D 직교 좌표
   ```

2. **적응적 보간 전략**
   - 근거리: 높은 밀도 보간 (2-3배)
   - 원거리: 낮은 밀도 보간 (1.5-2배)
   - 수평 방향: 각속도 기반 보간

3. **빔 패턴 정확도**
   - 실제 고도각 분포 반영
   - 비선형 보간 또는 스플라인 보간 사용
   - 빔 간격이 좁은 구간과 넓은 구간 구별

### 3.2 구현 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                  입력 처리 단계                       │
├─────────────────────────────────────────────────────┤
│ 1. Organized PointCloud (32×1024) 수신              │
│ 2. 구면 좌표 추출 (range, azimuth, altitude)       │
│ 3. 유효 포인트 마스크 생성                           │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│               구면 좌표계 보간 단계                    │
├─────────────────────────────────────────────────────┤
│ 1. 수직 보간 (고도각 방향)                          │
│    - 센서 빔 패턴 기반 비선형 보간                  │
│    - 새로운 고도각 계산 (스플라인/큐빅)             │
│ 2. 수평 보간 (방위각 방향)                          │
│    - 회전 속도 고려한 시간 보간                     │
│    - 모션 보상 적용                                 │
│ 3. Range 값 보간                                    │
│    - 인접 포인트의 range 가중 평균                  │
│    - 불연속성 검출 및 처리                          │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              3D 좌표 변환 단계                       │
├─────────────────────────────────────────────────────┤
│ 1. Ouster 좌표 변환 공식 적용                       │
│    - 빔 원점 오프셋 고려                            │
│    - 센서 고유 변환 행렬 적용                       │
│ 2. 보간된 구면 좌표 → 직교 좌표 변환               │
│ 3. 포인트 유효성 검증                               │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              출력 생성 단계                          │
├─────────────────────────────────────────────────────┤
│ 1. Organized PointCloud (128×1024) 생성             │
│ 2. 색상 정보 매핑 (카메라 융합)                     │
│ 3. 품질 메트릭 계산 및 모니터링                     │
└─────────────────────────────────────────────────────┘
```

### 3.3 주요 알고리즘 개선사항

#### 3.3.1 고도각 보간 알고리즘
```cpp
// 비선형 고도각 보간 (의사코드)
class AltitudeInterpolator {
    // 실제 빔 고도각 (라디안)
    vector<double> beam_altitudes_rad;
    
    // 스플라인 또는 비선형 보간
    double interpolateAltitude(int lower_beam, int upper_beam, double weight) {
        // 방법 1: 3차 스플라인 보간
        return cubicSpline(beam_altitudes_rad, lower_beam, upper_beam, weight);
        
        // 방법 2: 구면 선형 보간 (SLERP)
        return sphericalInterp(beam_altitudes_rad[lower_beam], 
                              beam_altitudes_rad[upper_beam], weight);
    }
};
```

#### 3.3.2 Range 보간 전략
```cpp
// 적응적 Range 보간
class RangeInterpolator {
    double interpolateRange(const Grid& ranges, int row, int col, 
                           InterpolationMethod method) {
        // 1. 불연속성 검출
        if (detectDiscontinuity(ranges, row, col)) {
            return nearestNeighbor(ranges, row, col);
        }
        
        // 2. 거리 기반 가중치 계산
        auto weights = calculateDistanceWeights(ranges, row, col);
        
        // 3. 가중 보간
        return weightedInterpolation(ranges, weights, method);
    }
};
```

#### 3.3.3 좌표 변환 정확도 개선
```cpp
// Ouster 공식 좌표 변환 (공식 문서 기반)
class OusterCoordinateTransform {
    // beam_to_lidar 변환 행렬의 오프셋 값들
    const double beam_origin_offset = 0.015806; // |n| 값 (미터)
    vector<double> beam_altitude_angles;  // 각 빔의 고도각
    vector<double> beam_azimuth_angles;   // 각 빔의 방위각 오프셋
    Eigen::Matrix4d beam_to_lidar_transform;
    
    Point3D rangeToXYZ(double range, int measurement_id, int beam_idx, 
                       int scan_width = 1024) {
        // 1. 엔코더 각도 계산
        double theta_encoder = 2.0 * M_PI * (1.0 - double(measurement_id) / scan_width);
        
        // 2. 방위각 오프셋 적용
        double theta_azimuth = -2.0 * M_PI * (beam_azimuth_angles[beam_idx] / 360.0);
        
        // 3. 고도각 변환
        double phi = 2.0 * M_PI * (beam_altitude_angles[beam_idx] / 360.0);
        
        // 4. Ouster 공식에 따른 XYZ 계산
        double cos_theta = cos(theta_encoder + theta_azimuth);
        double sin_theta = sin(theta_encoder + theta_azimuth);
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        
        Point3D point;
        point.x = (range - beam_origin_offset) * cos_theta * cos_phi + 
                  beam_to_lidar_transform(0,3) * cos(theta_encoder);
        point.y = (range - beam_origin_offset) * sin_theta * cos_phi + 
                  beam_to_lidar_transform(0,3) * sin(theta_encoder);
        point.z = (range - beam_origin_offset) * sin_phi + 
                  beam_to_lidar_transform(2,3);
        
        return point;
    }
    
    // 보간된 포인트를 위한 변환 (새로운 빔 인덱스)
    Point3D interpolatedRangeToXYZ(double range, int measurement_id, 
                                  double interpolated_altitude, 
                                  double interpolated_azimuth_offset,
                                  int scan_width = 1024) {
        double theta_encoder = 2.0 * M_PI * (1.0 - double(measurement_id) / scan_width);
        double theta_azimuth = -2.0 * M_PI * (interpolated_azimuth_offset / 360.0);
        double phi = 2.0 * M_PI * (interpolated_altitude / 360.0);
        
        // 동일한 변환 공식 적용
        double cos_theta = cos(theta_encoder + theta_azimuth);
        double sin_theta = sin(theta_encoder + theta_azimuth);
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        
        Point3D point;
        point.x = (range - beam_origin_offset) * cos_theta * cos_phi;
        point.y = (range - beam_origin_offset) * sin_theta * cos_phi;
        point.z = (range - beam_origin_offset) * sin_phi;
        
        // 보간된 빔의 경우 beam_to_lidar 오프셋은 0으로 가정
        // (원본 빔 사이에 위치하므로)
        
        return point;
    }
};
```

### 3.4 성능 최적화 전략

1. **병렬 처리 최적화**
   - 각 수직 슬라이스별 독립적 처리
   - SIMD 명령어 활용 (AVX2/SSE)
   - GPU 가속 옵션 (CUDA/OpenCL)

2. **메모리 효율성**
   - 구조화된 메모리 레이아웃
   - 캐시 친화적 데이터 접근 패턴
   - 메모리 풀 사용

3. **실시간 처리**
   - 프레임 드롭 방지를 위한 적응적 품질 조정
   - 파이프라인 병렬화
   - 지연 시간 모니터링

## 4. 구현 로드맵

### Phase 1: 기초 구조 재설계 (1-2주)
- [ ] 구면 좌표계 기반 데이터 구조 설계
- [ ] 비선형 고도각 보간 알고리즘 구현
- [ ] Ouster 공식 좌표 변환 통합

### Phase 2: 핵심 보간 엔진 (2-3주)
- [ ] 적응적 Range 보간 구현
- [ ] 수평(방위각) 보간 최적화
- [ ] 불연속성 검출 및 처리

### Phase 3: 통합 및 최적화 (1-2주)
- [ ] 전체 파이프라인 통합
- [ ] 성능 최적화 (병렬화, SIMD)
- [ ] 품질 검증 및 튜닝

### Phase 4: 카메라 융합 재통합 (1주)
- [ ] 개선된 포인트클라우드와 카메라 융합
- [ ] 색상 매핑 정확도 향상
- [ ] 실시간 모니터링 도구

## 5. 검증 계획

### 5.1 정량적 검증
- 보간된 포인트의 기하학적 정확도 측정
- 원본 대비 포인트 분포 균일성 평가
- 처리 시간 및 메모리 사용량 벤치마크

### 5.2 정성적 검증
- 시각적 품질 평가 (왜곡, 아티팩트)
- 다양한 환경에서의 테스트 (실내/실외)
- 엣지 케이스 처리 확인

## 6. 예상 결과

### 개선 목표
1. **기하학적 정확도**: 포인트클라우드 크기 및 형태 보존
2. **보간 품질**: 자연스러운 포인트 분포, 아티팩트 최소화
3. **처리 성능**: 20Hz 실시간 처리 (4배 보간 기준)
4. **메모리 효율**: 기존 대비 30% 메모리 사용량 감소

### 주요 개선사항
- 정확한 구면 좌표계 보간으로 기하학적 왜곡 해결
- 센서 특성을 반영한 비선형 보간으로 자연스러운 포인트 분포
- 최적화된 알고리즘으로 실시간 처리 보장

---

이 재설계를 통해 Ouster OS1-32의 32채널 포인트클라우드를 128채널로 정확하게 보간하여, 
고품질의 dense 포인트클라우드를 생성할 수 있을 것으로 예상됩니다.

## 7. 즉시 적용 가능한 개선사항

### 7.1 빠른 수정 사항
1. **스케일 팩터 감소**: 4.0 → 2.0 (32채널 → 64채널)
2. **고도각 보간 개선**: 선형 보간 → 큐빅 스플라인
3. **좌표 변환 수정**: Ouster 공식 적용

### 7.2 설정 파일 수정
```yaml
# interpolation_config.yaml
interpolation:
  scale_factor: 2.0  # 4.0에서 감소
  type: "SPHERICAL_CUBIC"  # 새로운 보간 타입
  coordinate_system: "SPHERICAL"  # 구면 좌표계 사용
  
  # 구면 좌표계 보간 설정
  spherical_interpolation:
    altitude_method: "CUBIC_SPLINE"  # 고도각 큐빅 스플라인
    azimuth_method: "LINEAR"         # 방위각 선형
    range_method: "ADAPTIVE"         # 거리 적응적
    
  # Ouster 좌표 변환 설정
  ouster_transform:
    use_official_formula: true
    beam_origin_offset_m: 0.015806
```

### 7.3 핵심 코드 수정 방향
```cpp
// main_fusion_node.cpp의 convertPixelToPoint 함수 대체
Point3D convertSphericalToPoint(double range, int col, int row, 
                               double interpolated_altitude_deg) {
    // Ouster 공식 좌표 변환 적용
    int measurement_id = col;
    double theta_encoder = 2.0 * M_PI * (1.0 - double(measurement_id) / 1024.0);
    double phi = interpolated_altitude_deg * M_PI / 180.0;
    
    double cos_theta = cos(theta_encoder);
    double sin_theta = sin(theta_encoder);
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    
    Point3D point;
    point.x = (range - 0.015806) * cos_theta * cos_phi;
    point.y = (range - 0.015806) * sin_theta * cos_phi;
    point.z = (range - 0.015806) * sin_phi;
    
    return point;
}
```