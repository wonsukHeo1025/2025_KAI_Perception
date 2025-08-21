/**
 * @file multi_iou_fusion_node.cpp
 * @brief Multi-camera IoU-based sensor fusion node for ROS2
 * 
 * This node performs sensor fusion between LiDAR 3D bounding boxes and 
 * YOLO detections from multiple cameras using IoU-based Hungarian matching.
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/bounding_box3_d_array.hpp>
#include <yolo_msgs/msg/detection_array.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include "calico/fusion/hungarian_matcher.hpp"
#include "calico/utils/config_loader.hpp"
#include "calico/utils/message_converter.hpp"
#include <std_msgs/msg/multi_array_dimension.hpp>
#include <atomic>
#include <filesystem>

namespace calico {
namespace nodes {

using BoundingBox3DArray = vision_msgs::msg::BoundingBox3DArray;
using DetectionArray = yolo_msgs::msg::DetectionArray;
using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using TrackedConeArray = custom_interface::msg::TrackedConeArray;

class MultiIoUFusionNode : public rclcpp::Node
{
public:
    explicit MultiIoUFusionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("calico_iou_fusion", options)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Multi-Camera IoU Fusion Node");
        
        // Declare parameters
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<double>("iou_threshold", 0.01);
        this->declare_parameter<bool>("enable_debug_viz", true);
        
        // Load configuration
        std::string config_file = this->get_parameter("config_file").as_string();
        if (config_file.empty()) {
            // Use package-relative default config path
            try {
                std::filesystem::path package_share_dir = ament_index_cpp::get_package_share_directory("calico");
                std::filesystem::path config_path = package_share_dir / "config" / "multi_hungarian_config.yaml";
                config_file = config_path.string();
            } catch (const std::exception& e) {
                // Fallback to relative path from executable location
                std::filesystem::path fallback_path = std::filesystem::path("..") / "share" / "calico" / "config" / "multi_hungarian_config.yaml";
                config_file = fallback_path.string();
            }
            RCLCPP_WARN(this->get_logger(), "No config file specified, using default: %s", config_file.c_str());
        }
        
        loadConfiguration(config_file);
        
        // Get runtime parameters
        iou_threshold_ = this->get_parameter("iou_threshold").as_double();
        enable_debug_viz_ = this->get_parameter("enable_debug_viz").as_bool();
        
        RCLCPP_INFO(this->get_logger(), "IoU threshold: %.2f", iou_threshold_);
        RCLCPP_INFO(this->get_logger(), "Debug visualization: %s", enable_debug_viz_ ? "enabled" : "disabled");
        
        // Setup publishers and subscribers
        setupPublishers();
        setupSubscribers();
        
        RCLCPP_INFO(this->get_logger(), "Multi-Camera IoU Fusion Node initialized with %zu cameras", 
                    camera_configs_.size());
        
        // Create status timer
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            [this]() {
                RCLCPP_INFO(this->get_logger(), 
                    "Status: LiDAR msgs: %zu, Cam1 msgs: %zu, Cam2 msgs: %zu | Topics: %s, %s, %s", 
                    lidar_msg_count_.load(), det1_msg_count_.load(), det2_msg_count_.load(),
                    lidar_boxes_topic_.c_str(),
                    camera_configs_[0].detections_topic.c_str(),
                    camera_configs_[1].detections_topic.c_str());
            });
        
        RCLCPP_INFO(this->get_logger(), "Multi-Camera IoU Fusion Node started");
    }

private:
    struct CameraConfig {
        std::string id;
        std::string detections_topic;
        std::string image_topic;
        std::string debug_image_topic;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        Eigen::Matrix4d T_lidar_to_cam;
        image_geometry::PinholeCameraModel cam_model;
    };
    
    // Configuration
    std::vector<CameraConfig> camera_configs_;
    std::string lidar_boxes_topic_;
    std::string fused_output_topic_;
    double iou_threshold_;
    bool enable_debug_viz_;
    int sync_queue_size_;
    double sync_slop_;
    
    // Publishers
    rclcpp::Publisher<TrackedConeArray>::SharedPtr fused_pub_;
    std::vector<rclcpp::Publisher<Image>::SharedPtr> debug_image_pubs_;
    
    // Subscribers and synchronizers
    std::shared_ptr<message_filters::Subscriber<BoundingBox3DArray>> lidar_sub_;
    std::vector<std::shared_ptr<message_filters::Subscriber<DetectionArray>>> detection_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<Image>>> image_subs_;
    
    // For 2 cameras (most common case)
    using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<
        BoundingBox3DArray, DetectionArray, DetectionArray, Image, Image>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync_2_;
    
    // Hungarian matcher
    std::unique_ptr<fusion::HungarianMatcher> matcher_;
    
    // Frame transformation from os_sensor to os_lidar
    Eigen::Matrix4d T_sensor_to_lidar_;
    
    // Status timer
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // Debug: Track message counts
    std::atomic<size_t> lidar_msg_count_{0};
    std::atomic<size_t> det1_msg_count_{0};
    std::atomic<size_t> det2_msg_count_{0};
    
    void loadConfiguration(const std::string& config_file)
    {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto calico_config = config["calico"];
            
            // Get topic names
            lidar_boxes_topic_ = calico_config["cones_topic"].as<std::string>();
            RCLCPP_INFO(this->get_logger(), "Configured LiDAR topic from config: %s", lidar_boxes_topic_.c_str());
            
            // Ensure we're using the BoundingBox3D topic
            if (lidar_boxes_topic_ != "/cone/lidar/box") {
                RCLCPP_WARN(this->get_logger(), "Overriding LiDAR topic from '%s' to '/cone/lidar/box'", 
                           lidar_boxes_topic_.c_str());
                lidar_boxes_topic_ = "/cone/lidar/box";
            }
            fused_output_topic_ = calico_config["output_topic"].as<std::string>();
            
            // Get sync parameters
            auto qos_config = calico_config["qos"];
            sync_queue_size_ = qos_config["sync_queue_size"].as<int>(10);
            sync_slop_ = qos_config["sync_slop"].as<double>(0.1);
            
            // Load camera configurations
            auto cameras = calico_config["cameras"];
            auto calib_config = calico_config["calibration"];
            std::string config_folder = calib_config["config_folder"].as<std::string>();
            
            // Handle relative paths - make them relative to config file directory
            if (config_folder == "./" || config_folder[0] != '/') {
                // Get directory of config file
                size_t last_slash = config_file.find_last_of("/");
                std::string config_dir = config_file.substr(0, last_slash + 1);
                
                if (config_folder == "./") {
                    config_folder = config_dir;
                } else {
                    config_folder = config_dir + config_folder;
                }
            }
            
            // Load intrinsic and extrinsic calibration files
            std::string intrinsic_file = config_folder + calib_config["camera_intrinsic_calibration"].as<std::string>();
            std::string extrinsic_file = config_folder + calib_config["camera_extrinsic_calibration"].as<std::string>();
            
            RCLCPP_INFO(this->get_logger(), "Loading calibration from: %s", config_folder.c_str());
            
            YAML::Node intrinsic_config = YAML::LoadFile(intrinsic_file);
            YAML::Node extrinsic_config = YAML::LoadFile(extrinsic_file);
            
            for (const auto& cam : cameras) {
                CameraConfig camera_cfg;
                camera_cfg.id = cam["id"].as<std::string>();
                camera_cfg.detections_topic = cam["detections_topic"].as<std::string>();
                
                // Get image topic (may not be in config)
                if (cam["image_topic"]) {
                    camera_cfg.image_topic = cam["image_topic"].as<std::string>();
                } else {
                    // Infer from camera ID
                    if (camera_cfg.id == "camera_1") {
                        camera_cfg.image_topic = "/camera_1/dbg_image";
                    } else if (camera_cfg.id == "camera_2") {
                        camera_cfg.image_topic = "/camera_2/dbg_image";
                    }
                }
                
                camera_cfg.debug_image_topic = "/" + camera_cfg.id + "/iou_fusion";
                
                // Load calibration for this camera
                loadCameraCalibration(camera_cfg, intrinsic_config, extrinsic_config);
                
                camera_configs_.push_back(camera_cfg);
                RCLCPP_INFO(this->get_logger(), "Configured camera: %s", camera_cfg.id.c_str());
            }
            
            // Initialize matcher
            matcher_ = std::make_unique<fusion::HungarianMatcher>();
            
            // Initialize os_sensor to os_lidar transform
            // Based on TF: 180 degree rotation + 0.036m Z translation
            T_sensor_to_lidar_ = Eigen::Matrix4d::Identity();
            T_sensor_to_lidar_(0, 0) = -1.0;  // X axis flip
            T_sensor_to_lidar_(1, 1) = -1.0;  // Y axis flip
            T_sensor_to_lidar_(2, 3) = 0.036; // Z translation
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load configuration: %s", e.what());
            throw;
        }
    }
    
    void loadCameraCalibration(CameraConfig& cfg, const YAML::Node& intrinsic, const YAML::Node& extrinsic)
    {
        // Load intrinsic parameters
        auto cam_intrinsic = intrinsic[cfg.id];
        
        // Handle nested array format for camera matrix
        auto K_nested = cam_intrinsic["camera_matrix"]["data"];
        std::vector<double> K;
        if (K_nested.IsSequence() && K_nested.size() > 0) {
            // Check if it's a nested array (2D format)
            if (K_nested[0].IsSequence()) {
                // Flatten the 2D array
                for (const auto& row : K_nested) {
                    for (const auto& val : row) {
                        K.push_back(val.as<double>());
                    }
                }
            } else {
                // Already flat array
                K = K_nested.as<std::vector<double>>();
            }
        }
        
        // Handle distortion coefficients
        auto D_node = cam_intrinsic["distortion_coefficients"]["data"];
        std::vector<double> D;
        if (D_node.IsSequence() && D_node.size() > 0) {
            // Check if it's a nested array
            if (D_node[0].IsSequence()) {
                // Flatten the 2D array
                for (const auto& row : D_node) {
                    for (const auto& val : row) {
                        D.push_back(val.as<double>());
                    }
                }
            } else {
                // Already flat array
                D = D_node.as<std::vector<double>>();
            }
        }
        
        cfg.camera_matrix = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) {
            cfg.camera_matrix.at<double>(i / 3, i % 3) = K[i];
        }
        
        cfg.dist_coeffs = cv::Mat(1, D.size(), CV_64F);
        for (size_t i = 0; i < D.size(); ++i) {
            cfg.dist_coeffs.at<double>(0, i) = D[i];
        }
        
        // Load extrinsic parameters
        auto cam_extrinsic = extrinsic[cfg.id];
        auto T_node = cam_extrinsic["extrinsic_matrix"];
        std::vector<double> T;
        
        if (T_node.IsSequence() && T_node.size() > 0) {
            // Check if it's a nested array (2D format)
            if (T_node[0].IsSequence()) {
                // Flatten the 2D array
                for (const auto& row : T_node) {
                    for (const auto& val : row) {
                        T.push_back(val.as<double>());
                    }
                }
            } else {
                // Already flat array
                T = T_node.as<std::vector<double>>();
            }
        }
        
        cfg.T_lidar_to_cam = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                cfg.T_lidar_to_cam(i, j) = T[i * 4 + j];
            }
        }
    }
    
    void setupPublishers()
    {
        // Main fused output publisher
        fused_pub_ = this->create_publisher<TrackedConeArray>(
            fused_output_topic_, 10);
        
        // Debug image publishers for each camera
        if (enable_debug_viz_) {
            for (const auto& cam : camera_configs_) {
                auto debug_pub = this->create_publisher<Image>(cam.debug_image_topic, 10);
                debug_image_pubs_.push_back(debug_pub);
            }
        }
    }
    
    void setupSubscribers()
    {
        // QoS settings
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = 10;  // Increased from 1 to buffer more messages
        
        // Create raw subscribers first to debug
        auto lidar_raw_sub = this->create_subscription<BoundingBox3DArray>(
            lidar_boxes_topic_, 10,
            [this](const BoundingBox3DArray::ConstSharedPtr& msg) {
                lidar_msg_count_++;
                RCLCPP_DEBUG(this->get_logger(), "Received LiDAR msg with %zu boxes", msg->boxes.size());
            });
        
        auto det1_raw_sub = this->create_subscription<DetectionArray>(
            camera_configs_[0].detections_topic, 10,
            [this](const DetectionArray::ConstSharedPtr& msg) {
                det1_msg_count_++;
                RCLCPP_DEBUG(this->get_logger(), "Received Cam1 detections: %zu", msg->detections.size());
            });
            
        auto det2_raw_sub = this->create_subscription<DetectionArray>(
            camera_configs_[1].detections_topic, 10,
            [this](const DetectionArray::ConstSharedPtr& msg) {
                det2_msg_count_++;
                RCLCPP_DEBUG(this->get_logger(), "Received Cam2 detections: %zu", msg->detections.size());
            });
        
        // LiDAR bounding boxes subscriber for synchronizer
        lidar_sub_ = std::make_shared<message_filters::Subscriber<BoundingBox3DArray>>(
            this, lidar_boxes_topic_, qos_profile);
        
        // Create subscribers for each camera
        for (const auto& cam : camera_configs_) {
            auto det_sub = std::make_shared<message_filters::Subscriber<DetectionArray>>(
                this, cam.detections_topic, qos_profile);
            detection_subs_.push_back(det_sub);
            
            if (enable_debug_viz_) {
                auto img_sub = std::make_shared<message_filters::Subscriber<Image>>(
                    this, cam.image_topic, qos_profile);
                image_subs_.push_back(img_sub);
            }
        }
        
        // Setup synchronizer for 2 cameras (most common case)
        if (camera_configs_.size() == 2 && enable_debug_viz_) {
            sync_2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
                SyncPolicy2(sync_queue_size_),
                *lidar_sub_,
                *detection_subs_[0], *detection_subs_[1],
                *image_subs_[0], *image_subs_[1]);
            
            // Set larger time tolerance for synchronization
            sync_2_->setMaxIntervalDuration(rclcpp::Duration(0, 200000000));  // 200ms
            
            sync_2_->registerCallback(
                std::bind(&MultiIoUFusionNode::syncCallback2WithImages, this,
                         std::placeholders::_1, std::placeholders::_2, 
                         std::placeholders::_3, std::placeholders::_4,
                         std::placeholders::_5));
            
            RCLCPP_INFO(this->get_logger(), "Synchronizer with images configured for topics:");
            RCLCPP_INFO(this->get_logger(), "  LiDAR: %s", lidar_boxes_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam1 Det: %s", camera_configs_[0].detections_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam2 Det: %s", camera_configs_[1].detections_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam1 Img: %s", camera_configs_[0].image_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam2 Img: %s", camera_configs_[1].image_topic.c_str());
                         
        } else if (camera_configs_.size() == 2) {
            // Without images for visualization
            using SyncPolicyNoImg = message_filters::sync_policies::ApproximateTime<
                BoundingBox3DArray, DetectionArray, DetectionArray>;
            auto sync_no_img = std::make_shared<message_filters::Synchronizer<SyncPolicyNoImg>>(
                SyncPolicyNoImg(sync_queue_size_),
                *lidar_sub_, *detection_subs_[0], *detection_subs_[1]);
            
            // Set larger time tolerance for synchronization
            sync_no_img->setMaxIntervalDuration(rclcpp::Duration(0, 200000000));  // 200ms
            
            sync_no_img->registerCallback(
                std::bind(&MultiIoUFusionNode::syncCallback2NoImages, this,
                         std::placeholders::_1, std::placeholders::_2, 
                         std::placeholders::_3));
            
            RCLCPP_INFO(this->get_logger(), "Synchronizer without images configured for topics:");
            RCLCPP_INFO(this->get_logger(), "  LiDAR: %s", lidar_boxes_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam1 Det: %s", camera_configs_[0].detections_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "  Cam2 Det: %s", camera_configs_[1].detections_topic.c_str());
        }
        
        RCLCPP_INFO(this->get_logger(), "Synchronizer setup complete");
    }
    
    void syncCallback2WithImages(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const DetectionArray::ConstSharedPtr& detections1,
        const DetectionArray::ConstSharedPtr& detections2,
        const Image::ConstSharedPtr& image1,
        const Image::ConstSharedPtr& image2)
    {
        RCLCPP_DEBUG(this->get_logger(), "syncCallback2WithImages called!");
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Processing fusion with %zu LiDAR boxes", lidar_boxes->boxes.size());
        
        std::vector<DetectionArray::ConstSharedPtr> detections = {detections1, detections2};
        std::vector<Image::ConstSharedPtr> images = {image1, image2};
        
        processFusion(lidar_boxes, detections, images);
    }
    
    void syncCallback2NoImages(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const DetectionArray::ConstSharedPtr& detections1,
        const DetectionArray::ConstSharedPtr& detections2)
    {
        RCLCPP_DEBUG(this->get_logger(), "syncCallback2NoImages called!");
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Processing fusion (no images) with %zu LiDAR boxes", lidar_boxes->boxes.size());
        
        std::vector<DetectionArray::ConstSharedPtr> detections = {detections1, detections2};
        std::vector<Image::ConstSharedPtr> images;  // Empty
        
        processFusion(lidar_boxes, detections, images);
    }
    
    void processFusion(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const std::vector<DetectionArray::ConstSharedPtr>& detections,
        const std::vector<Image::ConstSharedPtr>& images)
    {
        // Store fusion results
        std::vector<std::vector<std::string>> camera_class_names(camera_configs_.size());
        
        // Process each camera
        for (size_t cam_idx = 0; cam_idx < camera_configs_.size(); ++cam_idx) {
            const auto& cam_cfg = camera_configs_[cam_idx];
            const auto& yolo_detections = detections[cam_idx];
            
            // Project 3D boxes to this camera's 2D plane
            std::vector<cv::Rect2f> projected_boxes;
            for (const auto& box3d : lidar_boxes->boxes) {
                auto box2d = project3DBoxTo2D(box3d, cam_cfg);
                projected_boxes.push_back(box2d);
            }
            
            // Convert YOLO detections to 2D boxes
            std::vector<cv::Rect2f> yolo_boxes;
            for (const auto& det : yolo_detections->detections) {
                cv::Rect2f box(
                    det.bbox.center.position.x - det.bbox.size.x / 2,
                    det.bbox.center.position.y - det.bbox.size.y / 2,
                    det.bbox.size.x,
                    det.bbox.size.y);
                yolo_boxes.push_back(box);
            }
            
            // Compute IoU-based cost matrix
            Eigen::MatrixXd cost_matrix = computeIoUCostMatrix(projected_boxes, yolo_boxes);
            
            // Perform Hungarian matching
            auto match_result = matcher_->match(cost_matrix, 1.0 - iou_threshold_);
            
            // Store matched class names
            std::vector<std::string> class_names(lidar_boxes->boxes.size(), "Unknown");
            for (const auto& [yolo_idx, lidar_idx] : match_result.matches) {
                if (cost_matrix(yolo_idx, lidar_idx) < 1.0 - iou_threshold_) {
                    class_names[lidar_idx] = yolo_detections->detections[yolo_idx].class_name;
                }
            }
            camera_class_names[cam_idx] = class_names;
            
            // Visualize if enabled
            if (enable_debug_viz_ && !images.empty()) {
                visualizeMatching3D(images[cam_idx], lidar_boxes, yolo_boxes, 
                                   match_result, cam_idx, cost_matrix, cam_cfg, class_names);
            }
        }
        
        // Merge results from multiple cameras (voting or priority-based)
        std::vector<std::string> final_class_names = mergeClassifications(camera_class_names);
        
        // Publish fused results
        publishFusedResults(lidar_boxes, final_class_names);
    }
    
    cv::Rect2f project3DBoxTo2D(const vision_msgs::msg::BoundingBox3D& box3d, 
                                 const CameraConfig& cam_cfg)
    {
        // Get 8 corners of AABB (ignoring orientation for cones)
        std::vector<cv::Point3f> corners;
        double cx = box3d.center.position.x;
        double cy = box3d.center.position.y;
        double cz = box3d.center.position.z;
        double dx = box3d.size.x / 2.0;
        double dy = box3d.size.y / 2.0;
        double dz = box3d.size.z / 2.0;
        
        for (int i = 0; i < 8; ++i) {
            corners.emplace_back(
                cx + (i & 1 ? dx : -dx),
                cy + (i & 2 ? dy : -dy),
                cz + (i & 4 ? dz : -dz));
        }
        
        // Transform to camera frame and project
        std::vector<cv::Point2f> projected;
        for (const auto& pt : corners) {
            // First transform from os_sensor to os_lidar frame
            Eigen::Vector4d pt_sensor(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4d pt_lidar = T_sensor_to_lidar_ * pt_sensor;
            
            // Then transform from os_lidar to camera frame
            Eigen::Vector4d pt_cam = cam_cfg.T_lidar_to_cam * pt_lidar;
            
            if (pt_cam(2) > 0.1) {  // In front of camera
                cv::Point3d pt3d(pt_cam(0), pt_cam(1), pt_cam(2));
                std::vector<cv::Point2d> pts_2d;
                cv::projectPoints(std::vector<cv::Point3d>{pt3d},
                                  cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                                  cam_cfg.camera_matrix, cam_cfg.dist_coeffs,
                                  pts_2d);
                if (!pts_2d.empty()) {
                    projected.push_back(cv::Point2f(static_cast<float>(pts_2d[0].x),
                                                    static_cast<float>(pts_2d[0].y)));
                }
            }
        }
        
        // Find bounding rectangle
        if (projected.empty()) {
            return cv::Rect2f(0, 0, 0, 0);
        }
        
        return cv::boundingRect(projected);
    }
    
    Eigen::MatrixXd computeIoUCostMatrix(const std::vector<cv::Rect2f>& boxes1,
                                          const std::vector<cv::Rect2f>& boxes2)
    {
        Eigen::MatrixXd cost_matrix(boxes2.size(), boxes1.size());
        
        for (size_t i = 0; i < boxes2.size(); ++i) {
            for (size_t j = 0; j < boxes1.size(); ++j) {
                float iou = computeIoU(boxes2[i], boxes1[j]);
                cost_matrix(i, j) = 1.0 - iou;  // Cost = 1 - IoU
            }
        }
        
        return cost_matrix;
    }
    
    float computeIoU(const cv::Rect2f& box1, const cv::Rect2f& box2)
    {
        float x1 = std::max(box1.x, box2.x);
        float y1 = std::max(box1.y, box2.y);
        float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
        
        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float union_area = area1 + area2 - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
    
    std::vector<std::string> mergeClassifications(
        const std::vector<std::vector<std::string>>& camera_classes)
    {
        if (camera_classes.empty()) return {};
        
        size_t num_cones = camera_classes[0].size();
        std::vector<std::string> merged(num_cones, "Unknown");
        
        // Simple voting or priority-based merging
        for (size_t i = 0; i < num_cones; ++i) {
            std::map<std::string, int> vote_count;
            for (const auto& cam_classes : camera_classes) {
                if (cam_classes[i] != "Unknown") {
                    vote_count[cam_classes[i]]++;
                }
            }
            
            // Select class with most votes
            int max_votes = 0;
            for (const auto& [class_name, votes] : vote_count) {
                if (votes > max_votes) {
                    max_votes = votes;
                    merged[i] = class_name;
                }
            }
        }
        
        return merged;
    }
    
    void publishFusedResults(const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
                            const std::vector<std::string>& class_names)
    {
        // Convert to internal Cone representation first
        std::vector<calico::utils::Cone> cones;
        for (size_t i = 0; i < lidar_boxes->boxes.size(); ++i) {
            calico::utils::Cone cone;
            cone.x = lidar_boxes->boxes[i].center.position.x;
            cone.y = lidar_boxes->boxes[i].center.position.y;
            cone.z = lidar_boxes->boxes[i].center.position.z;
            cone.color = class_names[i];
            cone.id = static_cast<int>(i);  // Use index as track_id for now
            cone.confidence = 1.0;
            cones.push_back(cone);
        }
        
        // Convert to TrackedConeArray using MessageConverter
        auto msg = calico::utils::MessageConverter::toTrackedConeArray(cones);
        msg.header = lidar_boxes->header;  // Preserve original header
        
        fused_pub_->publish(msg);
        
        RCLCPP_INFO(this->get_logger(), "Published %zu fused cones to %s", 
                    lidar_boxes->boxes.size(), fused_output_topic_.c_str());
    }
    
    std::vector<cv::Point2f> project3DBoxCorners(const vision_msgs::msg::BoundingBox3D& box3d,
                                                  const CameraConfig& cam_cfg)
    {
        // Get 8 corners of AABB
        std::vector<cv::Point3f> corners;
        double cx = box3d.center.position.x;
        double cy = box3d.center.position.y;
        double cz = box3d.center.position.z;
        double dx = box3d.size.x / 2.0;
        double dy = box3d.size.y / 2.0;
        double dz = box3d.size.z / 2.0;
        
        // Define 8 corners in specific order for drawing edges
        corners.push_back(cv::Point3f(cx - dx, cy - dy, cz - dz)); // 0: bottom-front-left
        corners.push_back(cv::Point3f(cx + dx, cy - dy, cz - dz)); // 1: bottom-front-right
        corners.push_back(cv::Point3f(cx + dx, cy + dy, cz - dz)); // 2: bottom-back-right
        corners.push_back(cv::Point3f(cx - dx, cy + dy, cz - dz)); // 3: bottom-back-left
        corners.push_back(cv::Point3f(cx - dx, cy - dy, cz + dz)); // 4: top-front-left
        corners.push_back(cv::Point3f(cx + dx, cy - dy, cz + dz)); // 5: top-front-right
        corners.push_back(cv::Point3f(cx + dx, cy + dy, cz + dz)); // 6: top-back-right
        corners.push_back(cv::Point3f(cx - dx, cy + dy, cz + dz)); // 7: top-back-left
        
        // Transform and project corners
        std::vector<cv::Point2f> projected;
        for (const auto& pt : corners) {
            // Transform from os_sensor to os_lidar frame
            Eigen::Vector4d pt_sensor(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4d pt_lidar = T_sensor_to_lidar_ * pt_sensor;
            
            // Transform from os_lidar to camera frame
            Eigen::Vector4d pt_cam = cam_cfg.T_lidar_to_cam * pt_lidar;
            
            if (pt_cam(2) > 0.1) {  // In front of camera
                cv::Point3d pt3d(pt_cam(0), pt_cam(1), pt_cam(2));
                std::vector<cv::Point3d> pts_3d = {pt3d};
                std::vector<cv::Point2d> pts_2d;
                cv::projectPoints(pts_3d, cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                                 cam_cfg.camera_matrix, cam_cfg.dist_coeffs, pts_2d);
                if (!pts_2d.empty()) {
                    projected.push_back(cv::Point2f(pts_2d[0].x, pts_2d[0].y));
                } else {
                    projected.push_back(cv::Point2f(-1, -1)); // Invalid point
                }
            } else {
                projected.push_back(cv::Point2f(-1, -1)); // Behind camera
            }
        }
        
        return projected;
    }
    
    void draw3DBox(cv::Mat& img, const std::vector<cv::Point2f>& corners, 
                   const cv::Scalar& color, int thickness = 2)
    {
        if (corners.size() != 8) return;
        
        // Check if all corners are valid
        bool all_valid = true;
        for (const auto& pt : corners) {
            if (pt.x < 0 || pt.y < 0) {
                all_valid = false;
                break;
            }
        }
        if (!all_valid) return;
        
        // Draw bottom face (0-1-2-3-0)
        cv::line(img, corners[0], corners[1], color, thickness);
        cv::line(img, corners[1], corners[2], color, thickness);
        cv::line(img, corners[2], corners[3], color, thickness);
        cv::line(img, corners[3], corners[0], color, thickness);
        
        // Draw top face (4-5-6-7-4)
        cv::line(img, corners[4], corners[5], color, thickness);
        cv::line(img, corners[5], corners[6], color, thickness);
        cv::line(img, corners[6], corners[7], color, thickness);
        cv::line(img, corners[7], corners[4], color, thickness);
        
        // Draw vertical edges
        cv::line(img, corners[0], corners[4], color, thickness);
        cv::line(img, corners[1], corners[5], color, thickness);
        cv::line(img, corners[2], corners[6], color, thickness);
        cv::line(img, corners[3], corners[7], color, thickness);
    }
    
    void visualizeMatching3D(const Image::ConstSharedPtr& image,
                            const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
                            const std::vector<cv::Rect2f>& yolo_boxes,
                            const fusion::MatchResult& matches,
                            size_t cam_idx,
                            const Eigen::MatrixXd& cost_matrix,
                            const CameraConfig& cam_cfg,
                            const std::vector<std::string>& class_names)
    {
        if (cam_idx >= debug_image_pubs_.size()) return;
        
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        cv::Mat& img = cv_ptr->image;
        
        // Note: dbg_image already contains YOLO boxes, so we only draw LiDAR boxes
        
        // Draw 3D LiDAR boxes (green with thinner lines)
        for (size_t i = 0; i < lidar_boxes->boxes.size(); ++i) {
            auto corners = project3DBoxCorners(lidar_boxes->boxes[i], cam_cfg);
            draw3DBox(img, corners, cv::Scalar(0, 255, 0), 1);  // Reduced thickness from 2 to 1
        }
        
        // Highlight matched pairs and show IoU
        int matched_count = 0;
        for (const auto& [yolo_idx, lidar_idx] : matches.matches) {
            if (yolo_idx < static_cast<int>(yolo_boxes.size()) && 
                lidar_idx < static_cast<int>(lidar_boxes->boxes.size())) {
                float iou = 1.0 - cost_matrix(yolo_idx, lidar_idx);
                if (iou > iou_threshold_) {
                    matched_count++;
                    
                    // Get color based on cone class
                    cv::Scalar color;
                    const std::string& cone_class = class_names[lidar_idx];
                    if (cone_class == "Blue Cone" || cone_class == "blue cone") {
                        color = cv::Scalar(255, 0, 0);  // Blue in BGR
                    } else if (cone_class == "Yellow Cone" || cone_class == "yellow cone") {
                        color = cv::Scalar(0, 255, 255);  // Yellow in BGR
                    } else if (cone_class == "Red Cone" || cone_class == "red cone") {
                        color = cv::Scalar(0, 0, 255);  // Red in BGR
                    } else if (cone_class == "Orange Cone" || cone_class == "orange cone") {
                        color = cv::Scalar(0, 165, 255);  // Orange in BGR
                    } else {
                        color = cv::Scalar(128, 128, 128);  // Gray for unknown
                    }
                    
                    // Draw matched 3D box in cone color (slightly thicker)
                    auto corners = project3DBoxCorners(lidar_boxes->boxes[lidar_idx], cam_cfg);
                    draw3DBox(img, corners, color, 2);  // Reduced thickness from 3 to 2
                    
                    // Draw IoU score near the matched LiDAR box
                    if (!corners.empty()) {
                        cv::Point text_pos(corners[0].x, corners[0].y - 5);
                        std::string iou_text = "IoU: " + std::to_string(iou).substr(0, 4);
                        cv::putText(img, iou_text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 2);
                    }
                }
            }
        }
        
        // Add statistics
        std::string stats = "LiDAR: " + std::to_string(lidar_boxes->boxes.size()) + 
                           " | YOLO: " + std::to_string(yolo_boxes.size()) + 
                           " | Matched: " + std::to_string(matched_count);
        cv::putText(img, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(255, 255, 255), 2);
        
        // Publish debug image
        debug_image_pubs_[cam_idx]->publish(*cv_ptr->toImageMsg());
    }
    
};

} // namespace nodes
} // namespace calico

RCLCPP_COMPONENTS_REGISTER_NODE(calico::nodes::MultiIoUFusionNode)
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<calico::nodes::MultiIoUFusionNode>();
    
    RCLCPP_INFO(node->get_logger(), "Multi-Camera IoU Fusion Node started");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
