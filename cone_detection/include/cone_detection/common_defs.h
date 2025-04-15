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
              count(0), radius(0.0), valid(false) {}

        // 클러스터의 중심 및 유효성 계산
        void calculate() {
            count = cloud->size();
            if (count == 0) {
                valid = false;
                return;
            }

            Point sum{0, 0, 0, 0}, sum2{0, 0, 0, 0};
            for (auto &pt : *cloud) {
                sum.x += pt.x; sum.y += pt.y; sum.z += pt.z;
                sum.intensity += pt.intensity;
                sum2.x += pt.x * pt.x; sum2.y += pt.y * pt.y; sum2.z += pt.z * pt.z;
                sum2.intensity += pt.intensity * pt.intensity;
            }

            mean.x = sum.x / count;
            mean.y = sum.y / count;
            mean.z = sum.z / count;
            mean.intensity = sum.intensity / count;

            stddev.x = std::sqrt(sum2.x / count - mean.x * mean.x);
            stddev.y = std::sqrt(sum2.y / count - mean.y * mean.y);
            stddev.z = std::sqrt(sum2.z / count - mean.z * mean.z);
            stddev.intensity = std::sqrt(sum2.intensity / count - mean.intensity * mean.intensity);

            radius = std::sqrt(stddev.x * stddev.x + stddev.y * stddev.y + stddev.z * stddev.z);
            valid = (radius < 0.3) && (stddev.x < 0.2) && (stddev.y < 0.2) && (stddev.z < 0.2);
        }
    };
}  // namespace LIDAR
