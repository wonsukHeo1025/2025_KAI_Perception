#ifndef CALICO_NODES_PROJECTION_DEBUG_NODE_HPP
#define CALICO_NODES_PROJECTION_DEBUG_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <custom_interface/msg/modified_float32_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "calico/utils/config_loader.hpp"
#include "calico/utils/projection_utils.hpp"
#include "calico/utils/message_converter.hpp"

namespace calico {
namespace nodes {

class ProjectionDebugNode : public rclcpp::Node {
public:
    ProjectionDebugNode();

private:
    void loadCameraConfig();
    void lidarCallback(const custom_interface::msg::ModifiedFloat32MultiArray::SharedPtr msg);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void processAndPublish();
    
    // Publishers and subscribers
    rclcpp::Subscription<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    
    // Camera configuration
    std::string camera_id_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    Eigen::Matrix4d extrinsic_matrix_;  // T_lidar_to_cam
    Eigen::Matrix4d T_sensor_to_lidar_;  // Sensor to LiDAR transform (from Ouster)
    Eigen::Matrix4d T_sensor_to_cam_;    // Combined transform for cone data
    
    // Data storage
    std::vector<utils::Cone> latest_cones_;
    cv::Mat latest_image_;
    rclcpp::Time latest_lidar_time_;
    rclcpp::Time latest_image_time_;
    std::mutex data_mutex_;
    
    // Parameters
    std::string config_file_;
    double sync_tolerance_;
    int circle_radius_;
    cv::Scalar point_color_;
    cv::Scalar text_color_;
};

} // namespace nodes
} // namespace calico

#endif // CALICO_NODES_PROJECTION_DEBUG_NODE_HPP