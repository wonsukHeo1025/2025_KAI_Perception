#include <rclcpp/rclcpp.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace LIDAR {

class ConeVisualizationNode : public rclcpp::Node {
public:
    ConeVisualizationNode() : Node("cone_visualization_node"), 
                              previous_marker_count_(0),
                              previous_tracked_marker_count_(0) {
        RCLCPP_INFO(this->get_logger(), "Initializing Cone Visualization Node");
        
        // Declare parameters
        this->declare_parameter<std::string>("frame_id", "os_sensor");
        this->declare_parameter<bool>("show_track_ids", true);
        this->declare_parameter<bool>("show_velocity_arrows", true);
        this->declare_parameter<double>("cone_scale", 0.2);
        this->declare_parameter<double>("cone_height_offset", 0.3);
        this->declare_parameter<double>("arrow_scale", 1.5);
        this->declare_parameter<double>("min_velocity_threshold", 0.1);
        
        // Get parameters
        frame_id_ = this->get_parameter("frame_id").as_string();
        show_track_ids_ = this->get_parameter("show_track_ids").as_bool();
        show_velocity_arrows_ = this->get_parameter("show_velocity_arrows").as_bool();
        cone_scale_ = this->get_parameter("cone_scale").as_double();
        cone_height_offset_ = this->get_parameter("cone_height_offset").as_double();
        arrow_scale_ = this->get_parameter("arrow_scale").as_double();
        min_velocity_threshold_ = this->get_parameter("min_velocity_threshold").as_double();
        
        // Create subscribers
        tracked_cones_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
            "/cones/lidar/ukf", 10,
            std::bind(&ConeVisualizationNode::trackedConesCallback, this, std::placeholders::_1));
        
        // Create publishers
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/lidar/ukf", 10);
        velocity_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/lidar/velocity", 10);
        text_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/lidar/text", 10);
        
        RCLCPP_INFO(this->get_logger(), "Cone Visualization Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: /cones/lidar/ukf");
        RCLCPP_INFO(this->get_logger(), "Publishing LiDAR UKF cones to: /vis/cone/lidar/ukf");
        RCLCPP_INFO(this->get_logger(), "Publishing velocity arrows to: /vis/cone/lidar/velocity");
        RCLCPP_INFO(this->get_logger(), "Publishing track ID text to: /vis/cone/lidar/text");
    }
    
private:
    void trackedConesCallback(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
        visualization_msgs::msg::MarkerArray cone_markers;
        visualization_msgs::msg::MarkerArray velocity_markers;
        visualization_msgs::msg::MarkerArray text_markers;
        
        const auto current_time = this->now();
        double current_time_sec = current_time.seconds();
        
        // Delete previous markers
        deleteOldMarkers(cone_markers, "lidar_ukf_cones", previous_tracked_marker_count_, msg->header.stamp);
        deleteOldMarkers(velocity_markers, "velocity_arrows", previous_tracked_marker_count_, msg->header.stamp);
        deleteOldMarkers(text_markers, "track_ids", previous_tracked_marker_count_, msg->header.stamp);
        
        // Update velocity estimates
        updateVelocityEstimates(msg->cones, current_time_sec);
        
        // Create markers for each tracked cone
        int marker_id = 0;
        for (const auto& cone : msg->cones) {
            // Create cone marker (red cylinder for tracked cones)
            visualization_msgs::msg::Marker cone_marker;
            cone_marker.header = msg->header;
            cone_marker.header.frame_id = frame_id_;
            cone_marker.ns = "lidar_ukf_cones";
            cone_marker.id = marker_id;
            cone_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            cone_marker.action = visualization_msgs::msg::Marker::ADD;
            
            cone_marker.pose.position.x = cone.position.x;
            cone_marker.pose.position.y = cone.position.y;
            cone_marker.pose.position.z = cone.position.z - cone_height_offset_;
            cone_marker.pose.orientation.w = 1.0;
            
            cone_marker.scale.x = cone_marker.scale.y = cone_scale_;  // Same size as raw cones
            cone_marker.scale.z = 0.5;
            
            // Darker red color for tracked cones with transparency
            cone_marker.color.r = 0.8;
            cone_marker.color.g = 0.0;
            cone_marker.color.b = 0.0;
            cone_marker.color.a = 0.8;
            
            cone_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
            cone_markers.markers.push_back(cone_marker);
            
            // Create velocity arrow if enabled and velocity is significant
            if (show_velocity_arrows_ && velocity_estimates_.count(cone.track_id) > 0) {
                auto velocity = velocity_estimates_[cone.track_id];
                double velocity_magnitude = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
                
                if (velocity_magnitude > min_velocity_threshold_) {
                    visualization_msgs::msg::Marker arrow_marker;
                    arrow_marker.header = msg->header;
                    arrow_marker.header.frame_id = frame_id_;
                    arrow_marker.ns = "velocity_arrows";
                    arrow_marker.id = marker_id;
                    arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
                    arrow_marker.action = visualization_msgs::msg::Marker::ADD;
                    
                    // Arrow points
                    geometry_msgs::msg::Point start, end;
                    start.x = cone.position.x;
                    start.y = cone.position.y;
                    start.z = cone.position.z;
                    
                    // Predict position 1 second ahead (calico style)
                    double prediction_time = 1.0;
                    end.x = cone.position.x + velocity.x * prediction_time;
                    end.y = cone.position.y + velocity.y * prediction_time;
                    end.z = cone.position.z;
                    
                    arrow_marker.points.push_back(start);
                    arrow_marker.points.push_back(end);
                    
                    // Arrow properties - matching calico style
                    arrow_marker.scale.x = 0.05 * (1.0 + velocity_magnitude * 0.167);  // Shaft diameter
                    arrow_marker.scale.y = 0.08 * (1.0 + velocity_magnitude * 0.167);  // Head diameter
                    arrow_marker.scale.z = 0.1;  // Head length
                    
                    // Green color for velocity arrows
                    arrow_marker.color.r = 0.0;
                    arrow_marker.color.g = 1.0;
                    arrow_marker.color.b = 0.0;
                    arrow_marker.color.a = 0.9;
                    
                    arrow_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
                    velocity_markers.markers.push_back(arrow_marker);
                }
            }
            
            // Add track ID text if enabled
            if (show_track_ids_) {
                visualization_msgs::msg::Marker text_marker;
                text_marker.header = msg->header;
                text_marker.header.frame_id = frame_id_;
                text_marker.ns = "track_ids";
                text_marker.id = marker_id;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                
                text_marker.pose.position.x = cone.position.x;
                text_marker.pose.position.y = cone.position.y;
                text_marker.pose.position.z = cone.position.z + 0.5; // Above cone
                
                text_marker.scale.z = 0.3; // Text height
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                text_marker.color.a = 1.0;
                
                text_marker.text = "ID:" + std::to_string(cone.track_id);
                text_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
                
                text_markers.markers.push_back(text_marker);
            }
            
            marker_id++;
        }
        
        previous_tracked_marker_count_ = marker_id;
        
        // Publish markers
        if (!cone_markers.markers.empty()) {
            marker_pub_->publish(cone_markers);
        }
        if (!velocity_markers.markers.empty()) {
            velocity_marker_pub_->publish(velocity_markers);
        }
        if (!text_markers.markers.empty() && show_track_ids_) {
            text_marker_pub_->publish(text_markers);
        }
    }
    
    void deleteOldMarkers(visualization_msgs::msg::MarkerArray& markers,
                         const std::string& ns,
                         int count,
                         const rclcpp::Time& stamp) {
        for (int i = 0; i < count; ++i) {
            visualization_msgs::msg::Marker delete_marker;
            delete_marker.header.frame_id = frame_id_;
            delete_marker.header.stamp = stamp;
            delete_marker.ns = ns;
            delete_marker.id = i;
            delete_marker.action = visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(delete_marker);
        }
    }
    
    void updateVelocityEstimates(const std::vector<custom_interface::msg::TrackedCone>& cones,
                                double current_time) {
        // Update position history and estimate velocities
        for (const auto& cone : cones) {
            if (previous_positions_.count(cone.track_id) > 0) {
                auto& prev = previous_positions_[cone.track_id];
                double dt = current_time - prev.timestamp;
                
                if (dt > 0.001 && dt < 1.0) { // Valid time difference
                    geometry_msgs::msg::Vector3 velocity;
                    velocity.x = (cone.position.x - prev.position.x) / dt;
                    velocity.y = (cone.position.y - prev.position.y) / dt;
                    velocity.z = (cone.position.z - prev.position.z) / dt;
                    velocity_estimates_[cone.track_id] = velocity;
                }
            }
            
            // Update position history
            previous_positions_[cone.track_id] = {
                current_time,
                cone.position
            };
        }
        
        // Clean up old tracks
        std::vector<int> tracks_to_remove;
        for (const auto& [track_id, pos_data] : previous_positions_) {
            if (current_time - pos_data.timestamp > 2.0) { // Remove tracks older than 2 seconds
                tracks_to_remove.push_back(track_id);
            }
        }
        
        for (int id : tracks_to_remove) {
            previous_positions_.erase(id);
            velocity_estimates_.erase(id);
        }
    }
    
private:
    // Subscribers
    rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr tracked_cones_sub_;
    
    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr velocity_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr text_marker_pub_;
    
    // Parameters
    std::string frame_id_;
    bool show_track_ids_;
    bool show_velocity_arrows_;
    double cone_scale_;
    double cone_height_offset_;
    double arrow_scale_;
    double min_velocity_threshold_;
    
    // State
    int previous_marker_count_;
    int previous_tracked_marker_count_;
    
    // Velocity estimation
    struct PositionData {
        double timestamp;
        geometry_msgs::msg::Point position;
    };
    std::unordered_map<int, PositionData> previous_positions_;
    std::unordered_map<int, geometry_msgs::msg::Vector3> velocity_estimates_;
};

} // namespace LIDAR

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LIDAR::ConeVisualizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}