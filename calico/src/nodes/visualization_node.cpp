#include <rclcpp/rclcpp.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <atomic>
#include "calico/visualization/rviz_marker_publisher.hpp"
#include "calico/utils/message_converter.hpp"

namespace calico {

class VisualizationNode : public rclcpp::Node {
public:
    VisualizationNode() : Node("calico_visualization") {
        RCLCPP_INFO(this->get_logger(), "Initializing CALICO Visualization Node");
        
        // Declare parameters
        this->declare_parameter<double>("cone_height", 0.5);
        this->declare_parameter<double>("cone_radius", 0.15);
        this->declare_parameter<bool>("show_track_ids", true);
        this->declare_parameter<bool>("show_color_labels", false);
        this->declare_parameter<std::string>("frame_id", "ouster_lidar");
        
        // Load parameters
        double cone_height = this->get_parameter("cone_height").as_double();
        double cone_radius = this->get_parameter("cone_radius").as_double();
        show_track_ids_ = this->get_parameter("show_track_ids").as_bool();
        show_color_labels_ = this->get_parameter("show_color_labels").as_bool();
        frame_id_ = this->get_parameter("frame_id").as_string();
        
        // Track time for velocity estimation
        last_callback_time_ = this->get_clock()->now();
        
        // Initialize marker publishers (separate instances per stream to avoid shared state)
        marker_publisher_tracked_ = std::make_unique<visualization::RVizMarkerPublisher>();
        marker_publisher_tracked_->setConeParameters(cone_height, cone_radius);
        marker_publisher_tracked_->setShowTrackIds(show_track_ids_);
        marker_publisher_tracked_->setShowColorLabels(show_color_labels_);

        marker_publisher_fused_ = std::make_unique<visualization::RVizMarkerPublisher>();
        marker_publisher_fused_->setConeParameters(cone_height, cone_radius);
        // Fused markers topic publishes only cones (no arrows/text)
        marker_publisher_fused_->setShowTrackIds(false);
        marker_publisher_fused_->setShowColorLabels(false);
        
        // Setup QoS
        rclcpp::QoS qos(10);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        // Create subscribers
        tracked_cones_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
            "/cone/fused/ukf", qos,
            std::bind(&VisualizationNode::trackedConesCallback, this, std::placeholders::_1));
        
        fused_cones_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
            "/cone/fused", qos,
            std::bind(&VisualizationNode::fusedConesCallback, this, std::placeholders::_1));
        
        // Create publishers with organized topic names
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/fused/ukf", qos);
        
        fused_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/fused", qos);
        
        arrow_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/fused/velocity", qos);
        
        text_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/vis/cone/fused/text", qos);
        
        RCLCPP_INFO(this->get_logger(), "Visualization Node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Publishing fused UKF cones to: /vis/cone/fused/ukf");
        RCLCPP_INFO(this->get_logger(), "Publishing fused cones to: /vis/cone/fused");
        RCLCPP_INFO(this->get_logger(), "Publishing velocity arrows to: /vis/cone/fused/velocity");
        RCLCPP_INFO(this->get_logger(), "Publishing track ID text to: /vis/cone/fused/text");
        RCLCPP_INFO(this->get_logger(), "Show track IDs: %s", show_track_ids_ ? "yes" : "no");
        RCLCPP_INFO(this->get_logger(), "Show color labels: %s", show_color_labels_ ? "yes" : "no");
    }
    
private:
    void trackedConesCallback(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
        // Convert to internal representation
        auto cones = utils::MessageConverter::fromTrackedConeArray(*msg);
        
        // Get current time for velocity estimation
        auto current_time = this->get_clock()->now();
        double current_time_sec = current_time.seconds();
        
        // Update velocity estimates (only for UKF/tracked stream)
        marker_publisher_tracked_->updateVelocityEstimates(cones, current_time_sec);
        
        // Set velocity information for cones
        marker_publisher_tracked_->setVelocityForCones(cones);
        
        // Use frame_id from message if available, otherwise use parameter
        std::string frame_id = msg->header.frame_id.empty() ? frame_id_ : msg->header.frame_id;
        
        // Create marker array
        auto marker_array = marker_publisher_tracked_->createMarkerArray(
            cones, frame_id, msg->header.stamp);
        
        // Separate markers by type
        visualization_msgs::msg::MarkerArray main_markers;
        visualization_msgs::msg::MarkerArray arrow_markers;
        visualization_msgs::msg::MarkerArray text_markers;
        
        // Split markers by namespace
        for (const auto& marker : marker_array.markers) {
            if (marker.ns == "velocity_arrows") {
                arrow_markers.markers.push_back(marker);
            } else if (marker.ns == "track_id_text") {
                text_markers.markers.push_back(marker);
            } else {
                main_markers.markers.push_back(marker);
            }
        }
        
        // Publish to separate topics
        if (!main_markers.markers.empty()) {
            marker_pub_->publish(main_markers);
        }
        if (!arrow_markers.markers.empty()) {
            arrow_marker_pub_->publish(arrow_markers);
        }
        if (!text_markers.markers.empty() && show_track_ids_) {
            text_marker_pub_->publish(text_markers);
        }
        
        // Log statistics
        static std::atomic<int> vis_count{0};
        if (vis_count.fetch_add(1) % 100 == 0) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "Visualized %zu cones (%d cycles)",
                        cones.size(), vis_count.load());
        }
    }
    
    void fusedConesCallback(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
        // Convert to internal representation
        auto cones = utils::MessageConverter::fromTrackedConeArray(*msg);
        
        // Use frame_id from message if available, otherwise use parameter
        std::string frame_id = msg->header.frame_id.empty() ? frame_id_ : msg->header.frame_id;
        
        // Create marker array (we'll filter out velocity/text markers)
        auto marker_array = marker_publisher_fused_->createMarkerArray(
            cones, frame_id, msg->header.stamp);
        
        // Filter out velocity arrows and text markers - only keep cone markers
        visualization_msgs::msg::MarkerArray simple_markers;
        for (const auto& marker : marker_array.markers) {
            if (marker.ns != "velocity_arrows" && marker.ns != "track_id_text") {
                simple_markers.markers.push_back(marker);
            }
        }
        
        // Publish simple markers (only cones)
        if (!simple_markers.markers.empty()) {
            fused_marker_pub_->publish(simple_markers);
        }
        
        // Log statistics
        static std::atomic<int> fused_vis_count{0};
        if (fused_vis_count.fetch_add(1) % 100 == 0) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "Visualized %zu fused cones (%d cycles)",
                        cones.size(), fused_vis_count.load());
        }
    }
    
private:
    // Marker publishers (separate to avoid cross-topic delete interference)
    std::unique_ptr<visualization::RVizMarkerPublisher> marker_publisher_tracked_;
    std::unique_ptr<visualization::RVizMarkerPublisher> marker_publisher_fused_;
    
    // Subscribers
    rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr tracked_cones_sub_;
    rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr fused_cones_sub_;
    
    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr fused_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr arrow_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr text_marker_pub_;
    
    // Parameters
    bool show_track_ids_;
    bool show_color_labels_;
    std::string frame_id_;
    
    // Time tracking
    rclcpp::Time last_callback_time_;
};

} // namespace calico

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<calico::VisualizationNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}