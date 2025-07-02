#include "calico/nodes/projection_debug_node.hpp"
#include <chrono>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace calico {
namespace nodes {

ProjectionDebugNode::ProjectionDebugNode() 
    : Node("projection_debug_node"),
      circle_radius_(5),
      point_color_(0, 255, 0),  // Green in BGR
      text_color_(0, 255, 255),  // Yellow in BGR
      sync_tolerance_(0.1) {  // 100ms sync tolerance
    
    // Initialize T_sensor_to_lidar from Ouster OS1 documentation
    // This transforms from os_sensor frame to os_lidar frame
    T_sensor_to_lidar_ << -1,  0,  0,  0,
                           0, -1,  0,  0,
                           0,  0,  1, -0.038195,
                           0,  0,  0,  1;
    
    // Declare parameters
    this->declare_parameter("config_file", "");
    this->declare_parameter("camera_id", "camera_1");
    this->declare_parameter("sync_tolerance", 0.1);
    this->declare_parameter("circle_radius", 5);
    
    // Get parameters
    config_file_ = this->get_parameter("config_file").as_string();
    camera_id_ = this->get_parameter("camera_id").as_string();
    sync_tolerance_ = this->get_parameter("sync_tolerance").as_double();
    circle_radius_ = this->get_parameter("circle_radius").as_int();
    
    // Load camera configuration
    loadCameraConfig();
    
    // Create subscribers
    lidar_sub_ = this->create_subscription<custom_interface::msg::ModifiedFloat32MultiArray>(
        "/sorted_cones_time", 10,
        std::bind(&ProjectionDebugNode::lidarCallback, this, std::placeholders::_1));
    
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/" + camera_id_ + "/dbg_image", 10,
        std::bind(&ProjectionDebugNode::imageCallback, this, std::placeholders::_1));
    
    // Create publisher with camera-specific topic
    std::string debug_topic = "/debug/" + camera_id_ + "/projection_overlay";
    debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        debug_topic, 10);
    
    RCLCPP_INFO(this->get_logger(), 
                "Projection debug node initialized for camera: %s", camera_id_.c_str());
    RCLCPP_INFO(this->get_logger(), 
                "Publishing debug images to: %s", debug_topic.c_str());
}

void ProjectionDebugNode::loadCameraConfig() {
    if (config_file_.empty()) {
        // Use default config path
        std::string package_share = ament_index_cpp::get_package_share_directory("calico");
        config_file_ = package_share + "/config/multi_hungarian_config.yaml";
    }
    
    utils::ConfigLoader config_loader;
    auto hungarian_config = config_loader.loadConfig(config_file_);
    
    // Find camera config for our camera_id
    bool found = false;
    for (const auto& cam_config : hungarian_config.cameras) {
        if (cam_config.id == camera_id_) {
            camera_matrix_ = cam_config.camera_matrix;
            dist_coeffs_ = cam_config.dist_coeffs;
            extrinsic_matrix_ = cam_config.extrinsic_matrix;
            found = true;
            break;
        }
    }
    
    if (!found) {
        RCLCPP_ERROR(this->get_logger(), 
                     "Camera %s not found in config file", camera_id_.c_str());
        throw std::runtime_error("Camera configuration not found");
    }
    
    RCLCPP_INFO(this->get_logger(), "Loaded configuration for camera: %s", camera_id_.c_str());
    
    // Compute T_sensor_to_cam = T_lidar_to_cam * T_sensor_to_lidar
    // This is used for cone data which is in os_sensor frame
    T_sensor_to_cam_ = extrinsic_matrix_ * T_sensor_to_lidar_;
    
    RCLCPP_INFO(this->get_logger(), "Computed T_sensor_to_cam for cone projection");
}

void ProjectionDebugNode::lidarCallback(
    const custom_interface::msg::ModifiedFloat32MultiArray::SharedPtr msg) {
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // Convert message to cones
    latest_cones_ = utils::MessageConverter::fromModifiedFloat32MultiArray(*msg);
    latest_lidar_time_ = this->get_clock()->now();
    
    RCLCPP_DEBUG(this->get_logger(), "Received %zu LiDAR cones", latest_cones_.size());
    
    // Try to process if we have both image and LiDAR data
    processAndPublish();
}

void ProjectionDebugNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    try {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        latest_image_ = cv_ptr->image.clone();
        latest_image_time_ = this->get_clock()->now();
        
        RCLCPP_DEBUG(this->get_logger(), "Received image: %dx%d", 
                     latest_image_.cols, latest_image_.rows);
        
        // Try to process if we have both image and LiDAR data
        processAndPublish();
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void ProjectionDebugNode::processAndPublish() {
    // Check if we have both image and LiDAR data
    if (latest_image_.empty() || latest_cones_.empty()) {
        return;
    }
    
    // Check time synchronization
    double time_diff = std::abs((latest_lidar_time_ - latest_image_time_).seconds());
    if (time_diff > sync_tolerance_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Time difference too large: %.3f seconds", time_diff);
        return;
    }
    
    // Create debug image
    cv::Mat debug_image = latest_image_.clone();
    
    // Project each cone individually to maintain index correspondence
    std::vector<cv::Point2f> valid_projections_list;
    int valid_projections = 0;
    
    for (size_t i = 0; i < latest_cones_.size(); ++i) {
        const auto& cone = latest_cones_[i];
        
        // Transform cone from sensor frame to camera frame
        Eigen::Vector4d cone_sensor(cone.x, cone.y, cone.z, 1.0);
        Eigen::Vector4d cone_cam = T_sensor_to_cam_ * cone_sensor;
        
        // Check if cone is in front of camera
        if (cone_cam(2) > 1e-3) {  // 1mm minimum distance
            // Convert to OpenCV point
            cv::Point3f cv_point(cone_cam(0), cone_cam(1), cone_cam(2));
            std::vector<cv::Point3f> single_point = {cv_point};
            std::vector<cv::Point2f> projected;
            
            // Project to image plane
            cv::projectPoints(single_point, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                            camera_matrix_, dist_coeffs_, projected);
            
            if (!projected.empty()) {
                const auto& pt = projected[0];
                
                // Check if point is within image bounds
                if (pt.x >= 0 && pt.x < debug_image.cols && 
                    pt.y >= 0 && pt.y < debug_image.rows) {
                    
                    // Draw circle at projected point
                    cv::circle(debug_image, cv::Point(pt.x, pt.y), 
                              circle_radius_, point_color_, -1);
                    
                    // Draw cone info with correct index and color
                    std::string info = std::to_string(i) + ": " + cone.color;
                    cv::putText(debug_image, info, 
                               cv::Point(pt.x + 10, pt.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color_, 1);
                    
                    valid_projections++;
                    valid_projections_list.push_back(pt);
                }
            }
        }
    }
    
    // Add debug info to image
    std::string debug_text = "LiDAR: " + std::to_string(latest_cones_.size()) + 
                            " cones, Valid: " + std::to_string(valid_projections);
    cv::putText(debug_image, debug_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    // Log detailed projection info
    RCLCPP_INFO(this->get_logger(), 
                "Projection debug: %zu cones -> %d valid projections (in front of camera and in image bounds)",
                latest_cones_.size(), valid_projections);
    
    // Convert back to ROS message and publish
    cv_bridge::CvImage cv_msg;
    cv_msg.header.stamp = this->get_clock()->now();
    cv_msg.header.frame_id = camera_id_;
    cv_msg.encoding = sensor_msgs::image_encodings::BGR8;
    cv_msg.image = debug_image;
    
    debug_image_pub_->publish(*cv_msg.toImageMsg());
}

} // namespace nodes
} // namespace calico

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<calico::nodes::ProjectionDebugNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}