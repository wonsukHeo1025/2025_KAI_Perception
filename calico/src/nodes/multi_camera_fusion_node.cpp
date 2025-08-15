#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <atomic>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <yolo_msgs/msg/detection_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "calico/utils/config_loader.hpp"
#include "calico/utils/message_converter.hpp"
#include "calico/fusion/multi_camera_fusion.hpp"

namespace calico {

class MultiCameraFusionNode : public rclcpp::Node {
public:
    MultiCameraFusionNode() : Node("calico_multi_camera_fusion") {
        RCLCPP_INFO(this->get_logger(), "Initializing CALICO Multi-Camera Fusion Node");
        
        // Declare parameters
        this->declare_parameter<std::string>("config_file", "");
        
        // Load configuration
        std::string config_file = this->get_parameter("config_file").as_string();
        if (config_file.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No config file specified! Use --ros-args -p config_file:=path/to/config.yaml");
            throw std::runtime_error("Config file not specified");
        }
        
        try {
            utils::ConfigLoader loader;
            config_ = loader.loadConfig(config_file);
            RCLCPP_INFO(this->get_logger(), "Loaded configuration from: %s", config_file.c_str());
            RCLCPP_INFO(this->get_logger(), "Number of cameras: %zu", config_.cameras.size());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load config: %s", e.what());
            throw;
        }
        
        // Initialize fusion module
        fusion_ = std::make_unique<fusion::MultiCameraFusion>();
        fusion_->initialize(config_.cameras);
        fusion_->setMaxMatchingDistance(config_.max_matching_distance);
        
        // Setup QoS
        rclcpp::QoS qos(config_.history_depth);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        // Create publisher
        fused_cones_pub_ = this->create_publisher<custom_interface::msg::TrackedConeArray>(
            config_.output_topic, qos);
        
        // Setup message filters for time synchronization
        setupMessageFilters();
        
        RCLCPP_INFO(this->get_logger(), "CALICO Multi-Camera Fusion Node initialized successfully");
    }

private:
    void setupMessageFilters() {
        // QoS for message filters
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = config_.history_depth;
        
        // Create cone subscriber
        cones_sub_filter_ = std::make_shared<message_filters::Subscriber<custom_interface::msg::TrackedConeArray>>(
            this, config_.cones_topic, qos_profile);
        
        // Create detection subscribers for each camera
        for (const auto& cam : config_.cameras) {
            auto det_sub = std::make_shared<message_filters::Subscriber<yolo_msgs::msg::DetectionArray>>(
                this, cam.detections_topic, qos_profile);
            detection_sub_filters_.push_back(det_sub);
            
            RCLCPP_INFO(this->get_logger(), "Created synchronized subscriber for camera %s on topic: %s",
                       cam.id.c_str(), cam.detections_topic.c_str());
        }
        
        // Setup synchronizer based on number of cameras
        if (config_.cameras.size() == 2) {
            // 2 cameras + 1 cone topic
            typedef message_filters::sync_policies::ApproximateTime<
                custom_interface::msg::TrackedConeArray,
                yolo_msgs::msg::DetectionArray,
                yolo_msgs::msg::DetectionArray> SyncPolicy2;
            
            sync_2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
                SyncPolicy2(config_.sync_queue_size), *cones_sub_filter_, 
                *detection_sub_filters_[0], *detection_sub_filters_[1]);
            
            sync_2_->registerCallback(
                std::bind(&MultiCameraFusionNode::syncCallback2, this,
                         std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            
            // Set the sync tolerance (slop)
            sync_2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(config_.sync_slop));
            
            RCLCPP_INFO(this->get_logger(), "Setup 2-camera synchronizer with slop: %f seconds, queue: %d", 
                       config_.sync_slop, config_.sync_queue_size);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Only 2-camera configuration is currently supported");
            throw std::runtime_error("Unsupported camera configuration");
        }
    }
    
    void syncCallback2(const custom_interface::msg::TrackedConeArray::ConstSharedPtr& cone_msg,
                      const yolo_msgs::msg::DetectionArray::ConstSharedPtr& det1_msg,
                      const yolo_msgs::msg::DetectionArray::ConstSharedPtr& det2_msg) {
        RCLCPP_DEBUG(this->get_logger(), "Synchronized callback triggered");
        
        // Store messages in order
        std::unordered_map<std::string, yolo_msgs::msg::DetectionArray::ConstSharedPtr> camera_detections;
        camera_detections[config_.cameras[0].id] = det1_msg;
        camera_detections[config_.cameras[1].id] = det2_msg;
        
        performFusion(cone_msg, camera_detections);
    }
    
    void performFusion(const custom_interface::msg::TrackedConeArray::ConstSharedPtr& cone_msg,
                      const std::unordered_map<std::string, yolo_msgs::msg::DetectionArray::ConstSharedPtr>& camera_detections) {
        // Convert messages to internal representation
        auto cones = utils::MessageConverter::fromTrackedConeArray(*cone_msg);
        
        // Convert YOLO detections for each camera
        std::unordered_map<std::string, std::vector<utils::Detection>> detections_internal;
        for (const auto& [cam_id, det_msg] : camera_detections) {
            if (det_msg) {
                detections_internal[cam_id] = utils::MessageConverter::fromDetectionArray(*det_msg);
                RCLCPP_DEBUG(this->get_logger(), "Camera %s: %zu detections", 
                           cam_id.c_str(), det_msg->detections.size());
            }
        }
        
        // Perform fusion
        auto fused_cones = fusion_->fuse(cones, detections_internal);
        
        // Convert result back to ROS message
        auto output_msg = utils::MessageConverter::toTrackedConeArray(fused_cones);
        output_msg.header = cone_msg->header;  // Preserve original header
        
        // Publish result
        fused_cones_pub_->publish(output_msg);
        
        // Log statistics periodically
        static std::atomic<int> fusion_count{0};
        int current_count = fusion_count.fetch_add(1) + 1;
        if (current_count % 100 == 0) {
            int matched_cones = 0;
            for (const auto& cone : fused_cones) {
                if (cone.color != "Unknown") {
                    matched_cones++;
                }
            }
            
            RCLCPP_INFO(this->get_logger(), 
                       "Fusion cycle %d: %d/%zu cones matched with YOLO", 
                       current_count, matched_cones, fused_cones.size());
            
            // Log per-camera results
            auto& cam_results = fusion_->getCameraResults();
            for (const auto& [cam_id, result] : cam_results) {
                int matched = 0;
                for (int idx : result.matched_cone_indices) {
                    if (idx >= 0) matched++;
                }
                RCLCPP_INFO(this->get_logger(), 
                           "  Camera %s: %d matched, %zu unmatched",
                           cam_id.c_str(), matched, result.unmatched_cone_indices.size());
            }
        }
    }
    
private:
    // Configuration
    utils::CalicoConfig config_;
    
    // Fusion module
    std::unique_ptr<fusion::MultiCameraFusion> fusion_;
    
    // Publishers
    rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr fused_cones_pub_;
    
    // Message filters for synchronization
    std::shared_ptr<message_filters::Subscriber<custom_interface::msg::TrackedConeArray>> cones_sub_filter_;
    std::vector<std::shared_ptr<message_filters::Subscriber<yolo_msgs::msg::DetectionArray>>> detection_sub_filters_;
    
    // Synchronizers (only one will be used based on camera count)
    typedef message_filters::sync_policies::ApproximateTime<
        custom_interface::msg::TrackedConeArray,
        yolo_msgs::msg::DetectionArray,
        yolo_msgs::msg::DetectionArray> SyncPolicy2;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync_2_;
};

} // namespace calico

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<calico::MultiCameraFusionNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}