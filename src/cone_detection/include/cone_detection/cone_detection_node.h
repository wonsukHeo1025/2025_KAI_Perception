#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <pcl/common/transforms.h>

#include "common_defs.h"
#include "custom_interface/msg/modified_float32_multi_array.hpp"


namespace LIDAR {

class OutlierFilter : public rclcpp::Node {
public:
    struct Params {
        std::string topic_name;         // 토픽 이름
        std::string frame_id_;          // 프레임 ID
        bool x_threshold_enable = false;  // X 필터링 활성화 여부
        bool y_threshold_enable = false;  // Y 필터링 활성화 여부
        bool z_threshold_enable = true;  // Z 필터링 활성화 여부
        float x_threshold_min = -2.0f;   // X 최소값
        float x_threshold_max = 2.0f;    // X 최대값
        float y_threshold_min = -3.0f;   // Y 최소값
        float y_threshold_max = 3.0f;    // Y 최대값
        float z_threshold_min = -5.0f;   // Z 최소값
        float z_threshold_max = 1.0f;    // Z 최대값
        float min_distance = 1.5f;       // 최소 거리
        float max_distance = 70.0f;      // 최대 거리
        float intensity_threshold = 40.0f; // Intensity 기준값
        float plane_distance_threshold = 0.3f; // 평면 세그먼트 거리 허용값
        float roi_angle_min = 35.0f;     // ROI 최소 각도
        float roi_angle_max = 145.0f;    // ROI 최대 각도
        float voxel_leaf_size;        // Voxelization 크기
        float ec_cluster_tolerance;   // 클러스터링 거리 허용치
        int ec_min_cluster_size;      // 클러스터 최소 크기
        int ec_max_cluster_size;      // 클러스터 최대 크기
    };

    OutlierFilter();  // 생성자

protected:
    // 파라미터
    Params params_;
    
    // ROS2 퍼블리셔
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cones_pub_;
    rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr cones_time_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cones_cloud_;

    // ROS2 서브스크라이버
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;

    // 콜백 함수: 포인트 클라우드 데이터 수신 및 처리
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // 포인트 클라우드 필터링
    void filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out);

    // 180도 회전 보정
    void correctYawRotation(Cloud::Ptr &cloud);

    // Voxelization (다운샘플링)
    void voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size);

    // 클러스터링을 통한 콘 추출
    void clusterCones(Cloud::Ptr &cloud_out, std::vector<ConeDescriptor> &cones);

    // 클러스터링된 콘 데이터를 정렬
    std::vector<std::vector<double>> sortCones(const std::vector<ConeDescriptor> &cones);

    // 포인트 클라우드 퍼블리싱
    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
        Cloud::Ptr &cloud,
        const rclcpp::Time &timestamp);

    // 정렬된 콘 데이터를 퍼블리싱
    void publishArray(
        const rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr &publisher,
        const std::vector<std::vector<double>> &array);

    // 정련된 콘 데이터를 타임스탬프와 함께 퍼블리싱
    void publishArrayWithTimestamp(
        const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
        const std::vector<std::vector<double>> &array,
        const rclcpp::Time &timestamp);

    // 클러스터링된 콘 시각화
    void visualizeCones(const std::vector<ConeDescriptor> &cones);
    int previous_marker_count_ = 0;

    // ROI 영역 각도 계산
    float ROI_theta(float x, float y);
    
    void publishSortedConesMarkers(const std::vector<std::vector<double>> &sorted_cones);

};

}  // namespace LIDAR
