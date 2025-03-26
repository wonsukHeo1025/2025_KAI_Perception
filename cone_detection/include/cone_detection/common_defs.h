#pragma once

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <memory>
#include <vector>
#include <cmath>

// PCL Viewer 타입 정의
using Viewer = std::shared_ptr<pcl::visualization::PCLVisualizer>;
using Point = pcl::PointXYZI;       // XYZ + Intensity 포인트 타입
using Cloud = pcl::PointCloud<Point>;
using PointC = pcl::PointXYZRGB;   // XYZ + RGB 포인트 타입
using CloudC = pcl::PointCloud<PointC>;

namespace LIDAR {

    // 포인트 클래스 정의
    namespace PointClasses {
        static const int n_classes = 5;  // 클래스 수

        enum PointClass : uint8_t {
            inlier = 0,  // 내부 점
            ground,      // 지면
            too_high,    // 너무 높은 점
            too_far,     // 너무 먼 점
            too_close    // 너무 가까운 점
        };

        // 각 클래스의 색상 정의
        static constexpr uint8_t colors[n_classes][3] = {
            {0, 255, 0},    // inlier - green
            {0, 0, 255},    // ground - blue
            {255, 0, 255},  // too_high - magenta
            {255, 0, 0},    // too_far - red
            {255, 255, 0}   // too_close - yellow
        };
    }

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

    // 세그먼테이션 결과 타입 정의
    using Segmentation = std::vector<PointClasses::PointClass>;

}  // namespace LIDAR
