#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include "common_defs.h"

class Visualizer {
public:
    // PCL Visualizer 및 뷰포트
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int v1 = 0;  // 원시 포인트 클라우드 뷰포트
    int v2 = 0;  // 처리된 포인트 클라우드 뷰포트

    int point_size = 5;  // 포인트 크기

    // 텍스트 레이블
    const char* const c1 = "original";  // 원시 포인트 클라우드
    const char* const c2 = "processed"; // 처리된 포인트 클라우드

    // 생성자
    Visualizer() {
        viewer = std::make_shared<pcl::visualization::PCLVisualizer>("3D Visualizer");
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 20, 20, 0, 0, -2, 0, -1, -1);

        // Viewport 1: Original PointCloud
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer->setBackgroundColor(0, 0, 0, v1);
        viewer->addText(c1, 10, 10, "v1_text", v1);

        // Viewport 2: Processed PointCloud
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer->setBackgroundColor(0, 0, 0, v2);
        viewer->addText(c2, 10, 10, "v2_text", v2);

        // PointCloud 렌더링 속성 설정
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, c1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, c2);
        viewer->addCoordinateSystem(1.0);
    }

    // 3D 시각화 업데이트
    void draw(Cloud::ConstPtr cloud, Cloud::ConstPtr cloud_out, LIDAR::Segmentation& segm, 
              pcl::ModelCoefficients::Ptr plane_coefs, std::vector<LIDAR::ConeDescriptor>& cones,
              const std::vector<std::vector<double>>& sorted_cones) {
        // Processed PointCloud with Colors
        CloudC::Ptr cloud_color(new CloudC);
        cloud_color->resize(cloud_out->size());
        for (size_t n = 0; n < cloud_out->size(); ++n) {
            auto& c = (*cloud_color)[n];
            auto& p = (*cloud_out)[n];
            c.x = p.x;
            c.y = p.y;
            c.z = p.z;
            const uint8_t* rgb = LIDAR::PointClasses::colors[segm[n]];
            c.r = rgb[0];
            c.g = rgb[1];
            c.b = rgb[2];
        }

        // Original PointCloud Intensity
        pcl::visualization::PointCloudColorHandlerGenericField<Point> intensity(cloud, "intensity");

        // Processed PointCloud Segmentation
        pcl::visualization::PointCloudColorHandlerRGBField<PointC> segmentation(cloud_color);

        // Original PointCloud 업데이트 또는 추가
        if (!viewer->updatePointCloud<Point>(cloud, intensity, c1)) {
            viewer->addPointCloud<Point>(cloud, intensity, c1, v1);
        }

        // Processed PointCloud 업데이트 또는 추가
        if (!viewer->updatePointCloud<PointC>(cloud_color, segmentation, c2)) {
            viewer->addPointCloud<PointC>(cloud_color, segmentation, c2, v2);
        }

        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, c1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, c2);

        // Shapes and Annotations
        viewer->removeAllShapes();
        viewer->addPlane(*plane_coefs, "plane", v2);

        int n = 0;
        for (auto& cone : cones) {
            if (cone.valid) {
                std::string name = "sphere_" + std::to_string(n++);
                viewer->addSphere(cone.mean, 0.3, 1.0, 0.0, 0.0, name, v2);
            }
        }

        // Sorted Cones 시각화
        n = 0;
        for (const auto& cone : sorted_cones) {
            double x = cone[0];
            double y = cone[1];

            std::string name = "sorted_sphere_" + std::to_string(n++);
            viewer->addSphere(Point{x, y, 0.3}, 0.3, 0.0, 1.0, 0.0, name, v2); // 초록색 구체
        }

        // Update Viewer
        viewer->spinOnce(1);
    }
};
