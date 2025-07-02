#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace filc {

struct SphericalPoint {
    float range;
    float azimuth;
    float altitude;
    float intensity;
    bool valid;
};

class SphericalInterpolator {
public:
    SphericalInterpolator() {
        initializeBeamAltitudes();
    }
    
    // 큐빅 스플라인 보간을 위한 계수 계산
    struct SplineCoefficients {
        std::vector<float> a, b, c, d;
    };
    
    // 고도각 큐빅 스플라인 보간
    std::vector<float> interpolateAltitudesCubicSpline(int target_channels) {
        SplineCoefficients coeffs = computeSplineCoefficients(beam_altitudes_rad_);
        std::vector<float> result(target_channels);
        
        float scale = static_cast<float>(beam_altitudes_rad_.size() - 1) / (target_channels - 1);
        
        for (int i = 0; i < target_channels; ++i) {
            float x = i * scale;
            int j = static_cast<int>(x);
            j = std::min(j, static_cast<int>(beam_altitudes_rad_.size() - 2));
            float dx = x - j;
            
            // 큐빅 스플라인 공식: S(x) = a + b*dx + c*dx^2 + d*dx^3
            result[i] = coeffs.a[j] + coeffs.b[j]*dx + coeffs.c[j]*dx*dx + coeffs.d[j]*dx*dx*dx;
        }
        
        return result;
    }
    
    // 적응적 Range 보간
    float interpolateRangeAdaptive(float r1, float r2, float t, float discontinuity_threshold = 0.5f) {
        float diff = std::abs(r2 - r1);
        
        if (diff > discontinuity_threshold) {
            // 불연속성 감지: Nearest Neighbor
            return (t < 0.5f) ? r1 : r2;
        } else if (diff < 0.1f) {
            // 매우 작은 차이: 선형 보간
            return r1 * (1.0f - t) + r2 * t;
        } else {
            // 중간 차이: Hermite 보간 (부드러운 전환)
            float t2 = t * t;
            float t3 = t2 * t;
            float h1 = 2*t3 - 3*t2 + 1;
            float h2 = -2*t3 + 3*t2;
            return h1 * r1 + h2 * r2;
        }
    }
    
    // XYZ를 구면 좌표로 변환
    SphericalPoint cartesianToSpherical(float x, float y, float z, float intensity) {
        SphericalPoint sp;
        sp.range = std::sqrt(x*x + y*y + z*z);
        sp.azimuth = std::atan2(y, x);
        sp.altitude = std::asin(z / sp.range);
        sp.intensity = intensity;
        sp.valid = std::isfinite(sp.range) && sp.range > 0.0f;
        return sp;
    }
    
    // 구면 좌표를 XYZ로 변환 (Ouster 공식)
    void sphericalToCartesianOuster(const SphericalPoint& sp, int col, int total_cols,
                                   float& x, float& y, float& z) {
        // 엔코더 각도
        float theta_encoder = 2.0f * M_PI * (1.0f - static_cast<float>(col) / total_cols);
        
        // 거리 보정 (beam origin offset)
        float corrected_range = sp.range - beam_origin_offset_;
        
        // XYZ 계산
        float cos_phi = std::cos(sp.altitude);
        x = corrected_range * std::cos(theta_encoder) * cos_phi;
        y = corrected_range * std::sin(theta_encoder) * cos_phi;
        z = corrected_range * std::sin(sp.altitude);
    }
    
    // 보간 품질 검증
    bool validateInterpolation(const std::vector<float>& ranges, float variance_threshold = 0.25f) {
        if (ranges.size() < 2) return true;
        
        // 평균 계산
        float mean = 0.0f;
        int valid_count = 0;
        for (float r : ranges) {
            if (std::isfinite(r) && r > 0) {
                mean += r;
                valid_count++;
            }
        }
        if (valid_count == 0) return false;
        mean /= valid_count;
        
        // 분산 계산
        float variance = 0.0f;
        for (float r : ranges) {
            if (std::isfinite(r) && r > 0) {
                float diff = r - mean;
                variance += diff * diff;
            }
        }
        variance /= valid_count;
        
        return std::sqrt(variance) < variance_threshold;
    }
    
    const std::vector<float>& getBeamAltitudesRad() const { return beam_altitudes_rad_; }
    float getBeamOriginOffset() const { return beam_origin_offset_; }

private:
    void initializeBeamAltitudes() {
        // Ouster OS1-32 빔 고도각 (도)
        std::vector<float> beam_angles_deg = {
            -16.611f, -16.084f, -15.557f, -15.029f, -14.502f, -13.975f,
            -13.447f, -12.920f, -12.393f, -11.865f, -11.338f, -10.811f,
            -10.283f, -9.756f, -9.229f, -8.701f, -8.174f, -7.646f,
            -7.119f, -6.592f, -6.064f, -5.537f, -5.010f, -4.482f,
            -3.955f, -3.428f, -2.900f, -2.373f, -1.846f, -1.318f,
            -0.791f, -0.264f
        };
        
        // 라디안으로 변환
        beam_altitudes_rad_.resize(beam_angles_deg.size());
        for (size_t i = 0; i < beam_angles_deg.size(); ++i) {
            beam_altitudes_rad_[i] = beam_angles_deg[i] * M_PI / 180.0f;
        }
    }
    
    // 자연 큐빅 스플라인 계수 계산
    SplineCoefficients computeSplineCoefficients(const std::vector<float>& y) {
        int n = y.size();
        SplineCoefficients coeffs;
        coeffs.a.resize(n);
        coeffs.b.resize(n-1);
        coeffs.c.resize(n);
        coeffs.d.resize(n-1);
        
        // a 계수는 원본 값
        coeffs.a = y;
        
        // 삼대각 행렬 시스템 설정
        std::vector<float> h(n-1), alpha(n-1), l(n), mu(n-1), z(n);
        
        for (int i = 0; i < n-1; ++i) {
            h[i] = 1.0f; // 균등 간격 가정
        }
        
        for (int i = 1; i < n-1; ++i) {
            alpha[i] = 3.0f * (y[i+1] - y[i]) / h[i] - 3.0f * (y[i] - y[i-1]) / h[i-1];
        }
        
        // 전진 소거
        l[0] = 1.0f;
        mu[0] = 0.0f;
        z[0] = 0.0f;
        
        for (int i = 1; i < n-1; ++i) {
            l[i] = 2.0f * (h[i-1] + h[i]) - h[i-1] * mu[i-1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
        }
        
        l[n-1] = 1.0f;
        z[n-1] = 0.0f;
        coeffs.c[n-1] = 0.0f;
        
        // 후진 대입
        for (int i = n-2; i >= 0; --i) {
            coeffs.c[i] = z[i] - mu[i] * coeffs.c[i+1];
            coeffs.b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (coeffs.c[i+1] + 2.0f * coeffs.c[i]) / 3.0f;
            coeffs.d[i] = (coeffs.c[i+1] - coeffs.c[i]) / (3.0f * h[i]);
        }
        
        return coeffs;
    }
    
    std::vector<float> beam_altitudes_rad_;
    const float beam_origin_offset_ = 0.015806f; // 미터
};

} // namespace filc