#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vision_msgs/msg/bounding_box3_d_array.hpp>
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
#include "cone_detection/dbscan_clusterer.h"
// Legacy message type - commented out for migration to TrackedConeArray
// #include "custom_interface/msg/modified_float32_multi_array.hpp"
#include "custom_interface/msg/tracked_cone_array.hpp"
#include <kalman_filters/tracking/multi_tracker.hpp>


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
        bool use_dbscan = true;            // Use DBSCAN clustering (more robust to noise) instead of Euclidean clustering
        float min_cone_height = 0.0f;        // 최소 콘 높이
        float max_cone_height = 1.0f;        // 최대 콘 높이

        
        // Tracking parameters
        bool enable_tracking = true;
        double max_association_distance = 0.7;
        int min_hits_before_confirmation = 2;
        int max_age_before_deletion = 4;
        double ukf_p_initial_pos = 1.0;
        double ukf_p_initial_vel = 10.0;
        double ukf_r_measurement = 0.1;
        double ukf_q_pos = 0.01;
        double ukf_q_vel = 0.1;
    };

    OutlierFilter();  // 생성자

protected:
    // 파라미터
    Params params_;

    // 지면 계수 멤버 변수
    pcl::ModelCoefficients::Ptr last_plane_coefs_;
    
    // KdTree optimization for clustering
    pcl::search::KdTree<Point>::Ptr persistent_tree_;
    size_t last_cloud_size_;
    
    // Tracking
    std::shared_ptr<kalman_filters::tracking::MultiTracker> tracker_;
    
    // ROS2 퍼블리셔
    // Legacy format - commented out for migration to TrackedConeArray
    // rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr cones_time_pub;  // Original format for backward compatibility
    rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr cones_time_v2_pub;       // New TrackedConeArray format
    rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr cones_time_ukf_pub_;     // UKF tracked cones
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cones_cloud_;
    rclcpp::Publisher<vision_msgs::msg::BoundingBox3DArray>::SharedPtr bbox_publisher_;            // Bounding box publisher (data only)

    // ROS2 서브스크라이버
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;

    // 콜백 함수
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // 포인트 클라우드 처리 함수들
    void filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out);
    void lidarToSensorTransform(Cloud::Ptr &cloud);
    void voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size);
    void clusterCones(Cloud::Ptr &cloud_in, std::vector<ConeDescriptor> &cones);
    void validateConesFinalChecks(
        const std::vector<ConeDescriptor> &initial_cones,
        std::vector<ConeDescriptor> &validated_cones,
        const pcl::ModelCoefficients::ConstPtr &plane_coefs);
    std::vector<std::vector<double>> sortCones(const std::vector<ConeDescriptor> &cones);

    // 퍼블리싱 함수들
    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
        Cloud::Ptr &cloud,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");

    // Legacy publish function - commented out for migration to TrackedConeArray
    // void publishArrayWithTimestamp(
    //     const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
    //     const std::vector<std::vector<double>> &array,
    //     const rclcpp::Time &timestamp,
    //     const std::string& frame_id = "os_sensor");
    
    void publishTrackedConeArray(
        const rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr &publisher,
        const std::vector<ConeDescriptor> &cones,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");
    
    void publishTrackedConeArray(
        const rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr &publisher,
        const std::vector<kalman_filters::tracking::TrackedObject> &tracked_objects,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");
    
    void publishBoundingBoxes(
        const std::vector<ConeDescriptor> &cones,
        const rclcpp::Time &timestamp,
        const std::string& frame_id = "os_sensor");
    
    // 유틸리티 함수
    float ROI_theta(float x, float y);
};

}  // namespace LIDAR
