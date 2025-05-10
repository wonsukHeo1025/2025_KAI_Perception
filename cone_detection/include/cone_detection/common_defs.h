#pragma once

#include <pcl/common/common_headers.h>
#include <memory>
#include <vector>
#include <cmath>

using Point = pcl::PointXYZI;       // XYZ + Intensity 포인트 타입
using Cloud = pcl::PointCloud<Point>;
using PointC = pcl::PointXYZRGB;   // XYZ + RGB 포인트 타입
using CloudC = pcl::PointCloud<PointC>;

namespace LIDAR {
    // 콘 정보 디스크립터 클래스
    class ConeDescriptor {
    public:
        Cloud::Ptr cloud;         // 클러스터 포인트 클라우드
        Point mean, stddev;       // 평균 및 표준편차
        int count;                // 포인트 개수
        double radius;            // 반경
        bool valid;               // 유효성 여부

        // 기본 생성자
        ConeDescriptor()
            : cloud(new Cloud),
              count(0), radius(0.0), valid(false) {
            // 기본값을 NaN이 아닌 0으로 초기화
            mean.x = mean.y = mean.z = mean.intensity = 0.0f;
            stddev.x = stddev.y = stddev.z = stddev.intensity = 0.0f;
        }

        // 클러스터의 중심 및 유효성 계산
        void calculate() {
            count = cloud->size();
            if (count == 0) {
                valid = false;
                return;
            }

            try {
                Point sum{0, 0, 0, 0}, sum2{0, 0, 0, 0};
                int valid_points = 0;
                
                for (auto &pt : *cloud) {
                    // NaN 체크
                    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
                        continue;
                    }
                    
                    sum.x += pt.x; sum.y += pt.y; sum.z += pt.z;
                    sum.intensity += pt.intensity;
                    sum2.x += pt.x * pt.x; sum2.y += pt.y * pt.y; sum2.z += pt.z * pt.z;
                    sum2.intensity += pt.intensity * pt.intensity;
                    valid_points++;
                }

                // 유효한 점이 너무 적으면 계산하지 않음
                if (valid_points < 3) {  // PCA에 필요한 최소 포인트 수
                    valid = false;
                    return;
                }

                mean.x = sum.x / valid_points;
                mean.y = sum.y / valid_points;
                mean.z = sum.z / valid_points;
                mean.intensity = sum.intensity / valid_points;

                // 분산이 음수가 되지 않도록 오류 체크
                float var_x = sum2.x / valid_points - mean.x * mean.x;
                float var_y = sum2.y / valid_points - mean.y * mean.y;
                float var_z = sum2.z / valid_points - mean.z * mean.z;
                float var_i = sum2.intensity / valid_points - mean.intensity * mean.intensity;

                stddev.x = var_x > 0 ? std::sqrt(var_x) : 0;
                stddev.y = var_y > 0 ? std::sqrt(var_y) : 0;
                stddev.z = var_z > 0 ? std::sqrt(var_z) : 0;
                stddev.intensity = var_i > 0 ? std::sqrt(var_i) : 0;

                radius = std::sqrt(stddev.x * stddev.x + stddev.y * stddev.y + stddev.z * stddev.z);
                
                // 유효성 검사 - radius와 표준편차가 특정 임계값 이하인지
                valid = (radius < 0.3) && (stddev.x < 0.2) && (stddev.y < 0.2) && (stddev.z < 0.2);
                
                // NaN 검사 - 계산 결과가 NaN이면 유효하지 않음
                if (std::isnan(mean.x) || std::isnan(mean.y) || std::isnan(mean.z) ||
                    std::isnan(stddev.x) || std::isnan(stddev.y) || std::isnan(stddev.z) ||
                    std::isnan(radius)) {
                    valid = false;
                }
            } catch (...) {
                // 예외 발생 시 유효하지 않음으로 설정
                valid = false;
            }
        }
    };
}  // namespace LIDAR
