# 하이브리드 접근법: Range Image + 구면 좌표계 보간

## 개요
lidar-camera-fusion의 Range Image 접근법과 우리의 구면 좌표계 정밀 보간을 결합한 최적화된 방법

## 핵심 아이디어

### 1. Range Image 기반 구조화
```cpp
class OusterRangeImage {
private:
    arma::mat range_;      // 32x1024 거리 값
    arma::mat intensity_;  // 32x1024 강도 값
    arma::mat altitude_;   // 32x1 고도각 (라디안)
    arma::vec azimuth_;    // 1x1024 방위각 (라디안)
    
public:
    // Range Image 생성
    void fromPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud) {
        range_.set_size(32, 1024);
        intensity_.set_size(32, 1024);
        
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 1024; ++col) {
                int idx = row * 1024 + col;
                const auto& pt = cloud.points[idx];
                
                range_(row, col) = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
                intensity_(row, col) = pt.intensity;
            }
        }
    }
    
    // Armadillo 활용 고속 보간
    void interpolateRange() {
        arma::vec rows = arma::regspace(0, 31);  // 원본 행
        arma::vec cols = arma::regspace(0, 1023); // 원본 열
        
        arma::vec rows_interp = arma::regspace(0, 0.25, 31.75); // 128개 행
        
        arma::mat range_interp, intensity_interp;
        
        // 2D 보간 수행
        arma::interp2(cols, rows, range_, cols, rows_interp, range_interp, "cubic");
        arma::interp2(cols, rows, intensity_, cols, rows_interp, intensity_interp, "cubic");
        
        // 고도각은 별도 1D 큐빅 스플라인
        arma::vec altitude_interp = interpolateAltitudes(rows_interp);
        
        // 결과 저장
        range_ = range_interp;
        intensity_ = intensity_interp;
        altitude_ = altitude_interp;
    }
    
    // 보간 품질 검증 (분산 기반)
    void validateInterpolation() {
        for (int row = 1; row < 127; row += 4) {
            for (int col = 0; col < 1024; ++col) {
                // 4개 행의 분산 계산
                arma::vec local_ranges = range_.submat(row-1, col, row+2, col);
                double variance = arma::var(local_ranges);
                
                if (variance > 0.5) { // 임계값
                    // Nearest neighbor로 대체
                    for (int i = 1; i <= 3; ++i) {
                        range_(row + i, col) = range_(row, col);
                    }
                }
            }
        }
    }
    
    // Ouster 공식 활용 3D 변환
    pcl::PointCloud<pcl::PointXYZI> toPointCloud() {
        pcl::PointCloud<pcl::PointXYZI> cloud;
        cloud.width = 1024;
        cloud.height = 128;
        cloud.resize(128 * 1024);
        
        for (int row = 0; row < 128; ++row) {
            for (int col = 0; col < 1024; ++col) {
                // Ouster 좌표 변환
                double theta_encoder = 2.0 * M_PI * (1.0 - double(col) / 1024.0);
                double phi = altitude_(row);
                double r = range_(row, col) - 0.015806; // beam offset
                
                int idx = row * 1024 + col;
                cloud.points[idx].x = r * cos(theta_encoder) * cos(phi);
                cloud.points[idx].y = r * sin(theta_encoder) * cos(phi);
                cloud.points[idx].z = r * sin(phi);
                cloud.points[idx].intensity = intensity_(row, col);
            }
        }
        
        return cloud;
    }
};
```

### 2. 고도각 정밀 보간
```cpp
arma::vec interpolateAltitudes(const arma::vec& rows_interp) {
    // Ouster OS1-32 고도각 (도 → 라디안)
    arma::vec altitudes_deg = {
        -16.611, -16.084, -15.557, -15.029, -14.502, -13.975,
        -13.447, -12.920, -12.393, -11.865, -11.338, -10.811,
        -10.283, -9.756, -9.229, -8.701, -8.174, -7.646,
        -7.119, -6.592, -6.064, -5.537, -5.010, -4.482,
        -3.955, -3.428, -2.900, -2.373, -1.846, -1.318,
        -0.791, -0.264
    };
    
    arma::vec altitudes_rad = altitudes_deg * M_PI / 180.0;
    arma::vec rows_orig = arma::regspace(0, 31);
    
    // Armadillo의 고성능 스플라인 보간
    arma::vec altitude_interp;
    arma::interp1(rows_orig, altitudes_rad, rows_interp, altitude_interp, "spline");
    
    return altitude_interp;
}
```

### 3. 병렬 처리 최적화
```cpp
class ParallelInterpolator {
public:
    void processParallel(OusterRangeImage& range_image) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Range 보간
                range_image.interpolateRange();
            }
            
            #pragma omp section
            {
                // Intensity 보간
                range_image.interpolateIntensity();
            }
            
            #pragma omp section
            {
                // 고도각 계산
                range_image.computeAltitudes();
            }
        }
        
        // 보간 검증은 순차적으로
        range_image.validateInterpolation();
    }
};
```

## 장점

1. **성능**: Armadillo의 최적화된 행렬 연산 활용
2. **정확도**: Ouster 센서 특성 정확히 반영
3. **안정성**: 분산 기반 품질 검증
4. **확장성**: Range Image 구조로 다양한 처리 가능

## 구현 우선순위

1. **Phase 1**: 기본 Range Image 구조 구현
2. **Phase 2**: Armadillo 기반 보간 통합
3. **Phase 3**: 품질 검증 및 필터링
4. **Phase 4**: 병렬화 및 최적화

## 예상 성능

- 처리 속도: 50ms 이내 (20Hz)
- 메모리 사용: 300MB 이하
- 보간 품질: 분산 < 0.5m @ 10m 거리