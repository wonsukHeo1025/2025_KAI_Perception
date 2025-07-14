#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.h>

#include "filc/config_loader.hpp"

class VisualizationNode : public rclcpp::Node {
public:
    VisualizationNode() : Node("filc_visualization_node") {
        RCLCPP_INFO(this->get_logger(), "filc Visualization Node starting...");
        
        // Declare parameters
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<bool>("publish_camera_poses", true);
        this->declare_parameter<bool>("publish_stats", true);
        this->declare_parameter<double>("stats_publish_rate", 1.0);
        
        // Get parameters
        std::string config_file_path = this->get_parameter("config_file").as_string();
        publish_camera_poses_ = this->get_parameter("publish_camera_poses").as_bool();
        publish_stats_ = this->get_parameter("publish_stats").as_bool();
        double stats_rate = this->get_parameter("stats_publish_rate").as_double();
        
        // Load configuration
        if (config_file_path.empty()) {
            config_file_path = "/home/user1/ROS2_Workspace/ros2_ws/src/filc/config/multi_general_configuration.yaml";
            RCLCPP_WARN(this->get_logger(), "No config file specified, using default: %s", config_file_path.c_str());
        }
        
        config_loader_ = std::make_shared<filc::ConfigLoader>();
        if (!config_loader_->loadConfiguration(config_file_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load configuration from %s", config_file_path.c_str());
            rclcpp::shutdown();
            return;
        }
        
        const auto& lidar_config = config_loader_->getLidarConfig();
        const auto& camera_configs = config_loader_->getCameraConfigs();
        
        RCLCPP_INFO(this->get_logger(), "Visualization configuration loaded successfully");
        
        // Setup QoS profiles
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
        auto image_qos = rclcpp::QoS(rclcpp::KeepLast(2)).best_effort();
        
        // Subscribe to original data for statistics
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_config.lidar_topic,
            sensor_qos,
            std::bind(&VisualizationNode::lidarCallback, this, std::placeholders::_1)
        );
        
        // Subscribe to projected images for monitoring
        for (const auto& [camera_id, camera_config] : camera_configs) {
            auto projected_sub = this->create_subscription<sensor_msgs::msg::Image>(
                camera_config.projected_topic,
                image_qos,
                [this, camera_id](const sensor_msgs::msg::Image::SharedPtr msg) {
                    this->projectedImageCallback(msg, camera_id);
                }
            );
            projected_image_subs_[camera_id] = projected_sub;
            
            RCLCPP_INFO(this->get_logger(), "Monitoring projected images for camera %s on topic %s", 
                       camera_id.c_str(), camera_config.projected_topic.c_str());
        }
        
        // Publishers for visualization
        if (publish_camera_poses_) {
            camera_poses_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/filc/camera_poses", 
                rclcpp::QoS(rclcpp::KeepLast(1)).transient_local()
            );
            publishCameraPoses();
        }
        
        // Statistics publishing timer
        if (publish_stats_ && stats_rate > 0.0) {
            stats_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(static_cast<int>(1000.0 / stats_rate)),
                std::bind(&VisualizationNode::publishStatistics, this)
            );
        }
        
        RCLCPP_INFO(this->get_logger(), "filc Visualization Node initialized successfully");
    }

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Update LiDAR statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            lidar_stats_.last_message_time = this->now();
            lidar_stats_.total_messages++;
            lidar_stats_.point_count = msg->width * msg->height;
            lidar_stats_.frame_id = msg->header.frame_id;
        }
    }
    
    void projectedImageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& camera_id) {
        // Update camera statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            auto& camera_stat = camera_stats_[camera_id];
            camera_stat.last_message_time = this->now();
            camera_stat.total_messages++;
            camera_stat.image_width = msg->width;
            camera_stat.image_height = msg->height;
            camera_stat.encoding = msg->encoding;
        }
    }
    
    void publishCameraPoses() {
        const auto& camera_configs = config_loader_->getCameraConfigs();
        const auto& lidar_config = config_loader_->getLidarConfig();
        
        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;
        
        for (const auto& [camera_id, camera_config] : camera_configs) {
            // Create camera pose marker
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = lidar_config.frame_id;
            marker.header.stamp = this->now();
            marker.ns = "camera_poses";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Extract position and orientation from extrinsic matrix
            const auto& T = camera_config.extrinsic_matrix;
            
            // Position (translation part)
            marker.pose.position.x = T(0, 3);
            marker.pose.position.y = T(1, 3);
            marker.pose.position.z = T(2, 3);
            
            // Orientation (rotation part converted to quaternion)
            Eigen::Matrix3d rotation_matrix = T.block<3, 3>(0, 0);
            Eigen::Quaterniond q(rotation_matrix);
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();
            
            // Scale and color
            marker.scale.x = 0.1;
            marker.scale.y = 0.05;
            marker.scale.z = 0.03;
            
            // Different colors for different cameras
            if (camera_id == "camera_1") {
                marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;
            } else if (camera_id == "camera_2") {
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
            } else {
                marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0;
            }
            marker.color.a = 0.8;
            
            marker_array.markers.push_back(marker);
            
            // Create text label for camera
            visualization_msgs::msg::Marker text_marker;
            text_marker.header = marker.header;
            text_marker.ns = "camera_labels";
            text_marker.id = marker_id++;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            
            text_marker.pose = marker.pose;
            text_marker.pose.position.z += 0.1; // Offset text above camera
            
            text_marker.scale.z = 0.05;
            text_marker.color = marker.color;
            text_marker.color.a = 1.0;
            text_marker.text = camera_id;
            
            marker_array.markers.push_back(text_marker);
        }
        
        if (camera_poses_pub_) {
            camera_poses_pub_->publish(marker_array);
            RCLCPP_INFO(this->get_logger(), "Published camera poses for %zu cameras", camera_configs.size());
        }
    }
    
    void publishStatistics() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        auto current_time = this->now();
        
        // Log LiDAR statistics
        if (lidar_stats_.total_messages > 0) {
            auto time_since_last = (current_time - lidar_stats_.last_message_time).seconds();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "LiDAR Stats - Frame: %s, Messages: %lu, Points: %u, Last seen: %.2fs ago",
                lidar_stats_.frame_id.c_str(),
                lidar_stats_.total_messages,
                lidar_stats_.point_count,
                time_since_last
            );
        }
        
        // Log camera statistics
        for (const auto& [camera_id, stats] : camera_stats_) {
            if (stats.total_messages > 0) {
                auto time_since_last = (current_time - stats.last_message_time).seconds();
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Camera %s Stats - Messages: %lu, Size: %ux%u, Encoding: %s, Last seen: %.2fs ago",
                    camera_id.c_str(),
                    stats.total_messages,
                    stats.image_width,
                    stats.image_height,
                    stats.encoding.c_str(),
                    time_since_last
                );
            }
        }
        
        // Check for stale data
        const double stale_threshold = 5.0; // seconds
        
        if (lidar_stats_.total_messages > 0) {
            auto time_since_last = (current_time - lidar_stats_.last_message_time).seconds();
            if (time_since_last > stale_threshold) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                    "LiDAR data is stale (%.2fs old)", time_since_last);
            }
        }
        
        for (const auto& [camera_id, stats] : camera_stats_) {
            if (stats.total_messages > 0) {
                auto time_since_last = (current_time - stats.last_message_time).seconds();
                if (time_since_last > stale_threshold) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                        "Camera %s data is stale (%.2fs old)", camera_id.c_str(), time_since_last);
                }
            }
        }
    }

    // Statistics structures
    struct LidarStats {
        rclcpp::Time last_message_time;
        size_t total_messages = 0;
        uint32_t point_count = 0;
        std::string frame_id;
    };
    
    struct CameraStats {
        rclcpp::Time last_message_time;
        size_t total_messages = 0;
        uint32_t image_width = 0;
        uint32_t image_height = 0;
        std::string encoding;
    };

    // Members
    std::shared_ptr<filc::ConfigLoader> config_loader_;
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> projected_image_subs_;
    
    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr camera_poses_pub_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // Statistics
    std::mutex stats_mutex_;
    LidarStats lidar_stats_;
    std::map<std::string, CameraStats> camera_stats_;
    
    // Parameters
    bool publish_camera_poses_;
    bool publish_stats_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<VisualizationNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting filc Visualization Node...");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception in visualization node: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
} 