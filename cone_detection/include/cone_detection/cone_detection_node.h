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
#include <pcl/ModelCoefficients.h>
#include <pcl/common/pca.h>
#include <pcl/filters/passthrough.h>

#include "common_defs.h"
#include "custom_interface/msg/modified_float32_multi_array.hpp"


namespace LIDAR {

class OutlierFilter : public rclcpp::Node {
public:
    struct Params {
        std::string input_topic_name = "ouster/points"; // 토픽 이름
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
        float voxel_leaf_size = 0.1f;        // Voxelization 크기
        float ec_cluster_tolerance = 0.02f;   // 클러스터링 거리 허용치
        int ec_min_cluster_size = 10;      // 클러스터 최소 크기
        int ec_max_cluster_size = 100;      // 클러스터 최대 크기
        float min_cone_height = 0.0f;        // 최소 콘 높이
        float max_cone_height = 1.0f;        // 최대 콘 높이

        // 2단계 검증 파라미터
        bool enable_stage2_validation = false;
        float s1_ec_cluster_tolerance = 0.45f;
        int s1_ec_min_cluster_size = 3;
        int s1_ec_max_cluster_size = 250;
        float s2_roi_cylinder_radius = 0.25f;
        float s2_roi_cylinder_bottom_offset = -0.1f;
        float s2_roi_cylinder_top_offset = 0.7f;
        int s2_min_points_in_reconstructed_roi = 10;
        int s2_max_points_in_reconstructed_roi = 500;
        
        // 방법론 3: 높이별 포인트 밀도 변화율 분석 파라미터
        int s2_height_histogram_bins = 5; // YAML에서 기본값과 일치시킴
        int s2_max_uphill_transitions_allowed = 1;
        float s2_bottom_heavy_ratio_threshold = 0.5f;
        int s2_bottom_bins_count_for_heavy_check = 2;
        float s2_top_sparse_max_point_ratio_per_bin = 0.25f;
        int s2_num_top_bins_for_sparsity_check = 1;
    };

    OutlierFilter();  // 생성자

protected:
    // 파라미터
    Params params_;

    // 지면 계수 멤버 변수
    pcl::ModelCoefficients::Ptr last_plane_coefs_;
    Cloud::Ptr original_cloud_for_stage2_; // 원본 포인트 클라우드 저장용 (Stage2용)
    
    // ROS2 퍼블리셔
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr cones_time_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cones_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_points_fixed_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_reconstructed_cones_cloud_; // Stage2 재구성 콘 발행용

    // ROS2 서브스크라이버
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;

    int previous_marker_count_ = 0;

    // 콜백 함수
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // 포인트 클라우드 처리 함수들
    void filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out);
    void lidarToSensorTransform(Cloud::Ptr &cloud);
    void voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size);
    void clusterCones(Cloud::Ptr &cloud_in, std::vector<ConeDescriptor> &cones, bool use_s1_params);
    void validateConesFinalChecks(
        const std::vector<ConeDescriptor> &initial_cones,
        std::vector<ConeDescriptor> &validated_cones,
        const pcl::ModelCoefficients::ConstPtr &plane_coefs);
    void validateAndReconstructConesStage2(
        const std::vector<ConeDescriptor>& stage1_cones,
        const Cloud::Ptr& original_cloud,
        std::vector<ConeDescriptor>& out_validated_cones,
        const rclcpp::Time& timestamp);
    void reconstructPointsAroundCones(
        const std::vector<ConeDescriptor>& cones_to_reconstruct,
        const Cloud::Ptr& source_cloud,
        Cloud::Ptr& out_reconstructed_cloud,
        const std::string& context_info);
    std::vector<std::vector<double>> sortCones(const std::vector<ConeDescriptor> &cones);

    // 퍼블리싱 함수들
    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
        Cloud::Ptr &cloud,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");

    void publishArrayWithTimestamp(
        const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
        const std::vector<std::vector<double>> &array,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");

    void visualizeCones(const std::vector<ConeDescriptor> &cones, const std::string& frame_id = "os_sensor");
    
    // 유틸리티 함수
    float ROI_theta(float x, float y);
};

}  // namespace LIDAR
