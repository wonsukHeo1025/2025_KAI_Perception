/**
 * @file single_iou_fusion_node.cpp
 * @brief Single-camera IoU-based sensor fusion node for ROS2
 * 
 * This node performs sensor fusion between LiDAR 3D bounding boxes and 
 * YOLO detections from a single camera using IoU-based Hungarian matching.
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

class SingleIoUFusionNode : public rclcpp::Node
{
public:
    explicit SingleIoUFusionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("calico_single_iou_fusion", options)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Single-Camera IoU Fusion Node");
        
        // Declare parameters
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<double>("iou_threshold", 0.01);
        this->declare_parameter<bool>("enable_debug_viz", true);
        // Time sync parameters
        this->declare_parameter<std::string>("time_sync_mode", "arrival_ros"); // header | arrival_ros | arrival_wall
        this->declare_parameter<double>("arrival_slop", 0.2); // seconds
        this->declare_parameter<bool>("override_fused_stamp_now", true);
        
        // Load configuration
        std::string config_file = this->get_parameter("config_file").as_string();
        if (config_file.empty()) {
            // Use package-relative default config path
            try {
                std::filesystem::path package_share_dir = ament_index_cpp::get_package_share_directory("calico");
                std::filesystem::path config_path = package_share_dir / "config" / "single_hungarian_config.yaml";
                config_file = config_path.string();
            } catch (const std::exception& e) {
                // Fallback to relative path from executable location
                std::filesystem::path fallback_path = std::filesystem::path("..") / "share" / "calico" / "config" / "single_hungarian_config.yaml";
                config_file = fallback_path.string();
            }
            RCLCPP_WARN(this->get_logger(), "No config file specified, using default: %s", config_file.c_str());
        }
        
        loadConfiguration(config_file);
        
        // Get runtime parameters
        iou_threshold_ = this->get_parameter("iou_threshold").as_double();
        enable_debug_viz_ = this->get_parameter("enable_debug_viz").as_bool();
        // Parse time sync parameters
        {
            const auto mode = this->get_parameter("time_sync_mode").as_string();
            if (mode == "header") {
                time_sync_mode_ = TimeSyncMode::HEADER;
            } else if (mode == "arrival_wall") {
                time_sync_mode_ = TimeSyncMode::ARRIVAL_WALL;
            } else {
                time_sync_mode_ = TimeSyncMode::ARRIVAL_ROS;
            }
            arrival_slop_ = this->get_parameter("arrival_slop").as_double();
            override_fused_stamp_now_ = this->get_parameter("override_fused_stamp_now").as_bool();
        }
        
        RCLCPP_INFO(this->get_logger(), "IoU threshold: %.2f", iou_threshold_);
        RCLCPP_INFO(this->get_logger(), "Debug visualization: %s", enable_debug_viz_ ? "enabled" : "disabled");
        RCLCPP_INFO(this->get_logger(), "Time sync mode: %s", 
            time_sync_mode_ == TimeSyncMode::HEADER ? "header" :
            (time_sync_mode_ == TimeSyncMode::ARRIVAL_WALL ? "arrival_wall" : "arrival_ros"));
        RCLCPP_INFO(this->get_logger(), "Arrival slop: %.3f s", arrival_slop_);
        
        // Setup publishers and subscribers
        setupPublishers();
        setupSubscribers();
        
        RCLCPP_INFO(this->get_logger(), "Single-Camera IoU Fusion Node initialized with camera: %s", 
                    camera_config_.id.c_str());
        
        // Create status timer
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            [this]() {
                RCLCPP_INFO(this->get_logger(), 
                    "Status: LiDAR msgs: %zu, Camera msgs: %zu | Topics: %s, %s", 
                    lidar_msg_count_.load(), det_msg_count_.load(),
                    lidar_boxes_topic_.c_str(),
                    camera_config_.detections_topic.c_str());
            });
        
        RCLCPP_INFO(this->get_logger(), "Single-Camera IoU Fusion Node started");
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
    enum class TimeSyncMode { HEADER, ARRIVAL_ROS, ARRIVAL_WALL };
    CameraConfig camera_config_;
    std::string lidar_boxes_topic_;
    std::string fused_output_topic_;
    double iou_threshold_;
    bool enable_debug_viz_;
    bool filter_unknown_ = false;
    int sync_queue_size_;
    double sync_slop_;
    TimeSyncMode time_sync_mode_;
    double arrival_slop_;
    bool override_fused_stamp_now_;
    
    // Publishers
    rclcpp::Publisher<TrackedConeArray>::SharedPtr fused_pub_;
    rclcpp::Publisher<Image>::SharedPtr debug_image_pub_;
    
    // Subscribers and synchronizers
    std::shared_ptr<message_filters::Subscriber<BoundingBox3DArray>> lidar_sub_;
    std::shared_ptr<message_filters::Subscriber<DetectionArray>> detection_sub_;
    std::shared_ptr<message_filters::Subscriber<Image>> image_sub_;
    
    // Synchronizer for single camera
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        BoundingBox3DArray, DetectionArray, Image>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    using SyncPolicyNoImg = message_filters::sync_policies::ApproximateTime<
        BoundingBox3DArray, DetectionArray>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicyNoImg>> sync_no_img_;
    
    // Hungarian matcher
    std::unique_ptr<fusion::HungarianMatcher> matcher_;
    
    // Frame transformation from os_sensor to os_lidar
    Eigen::Matrix4d T_sensor_to_lidar_;
    
    // Status timer
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // Debug: Track message counts
    std::atomic<size_t> lidar_msg_count_{0};
    std::atomic<size_t> det_msg_count_{0};

    // Raw subscriptions for counters and arrival-time sync
    rclcpp::Subscription<BoundingBox3DArray>::SharedPtr lidar_raw_sub_;
    rclcpp::Subscription<DetectionArray>::SharedPtr det_raw_sub_;
    rclcpp::Subscription<Image>::SharedPtr img_raw_sub_;

    // Arrival-time caches
    BoundingBox3DArray::ConstSharedPtr last_lidar_boxes_;
    rclcpp::Time last_lidar_recv_time_{0, 0, RCL_ROS_TIME};
    DetectionArray::ConstSharedPtr last_detection_;
    rclcpp::Time last_det_recv_time_{0, 0, RCL_ROS_TIME};
    Image::ConstSharedPtr last_image_;
    rclcpp::Time last_img_recv_time_{0, 0, RCL_ROS_TIME};
    
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
            
            // Load camera configuration (single camera)
            auto cameras = calico_config["cameras"];
            auto calib_config = calico_config["calibration"];
            std::string config_folder = calib_config["config_folder"].as<std::string>();

            // Optional: filter_unknown to drop Unknown cones from output
            if (calico_config["filter_unknown"]) {
                filter_unknown_ = calico_config["filter_unknown"].as<bool>();
            } else {
                filter_unknown_ = false;
            }
            RCLCPP_INFO(this->get_logger(), "Filter unknown cones: %s", filter_unknown_ ? "enabled" : "disabled");
            
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
            
            // Load single camera config
            if (cameras.size() != 1) {
                RCLCPP_ERROR(this->get_logger(), "Single camera config must have exactly 1 camera, found %zu", cameras.size());
                throw std::runtime_error("Invalid camera configuration");
            }
            
            auto cam = cameras[0];
            camera_config_.id = cam["id"].as<std::string>();
            camera_config_.detections_topic = cam["detections_topic"].as<std::string>();
            
            // Get image topic (may not be in config)
            if (cam["image_topic"]) {
                camera_config_.image_topic = cam["image_topic"].as<std::string>();
            } else {
                // Infer from camera ID
                if (camera_config_.id == "camera_1") {
                    camera_config_.image_topic = "/camera_1/dbg_image";
                } else {
                    camera_config_.image_topic = "/" + camera_config_.id + "/dbg_image";
                }
            }
            
            camera_config_.debug_image_topic = "/" + camera_config_.id + "/iou_fusion";
            
            // Load calibration for this camera
            loadCameraCalibration(camera_config_, intrinsic_config, extrinsic_config);
            
            RCLCPP_INFO(this->get_logger(), "Configured camera: %s", camera_config_.id.c_str());
            
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
        
        // Debug image publisher
        if (enable_debug_viz_) {
            debug_image_pub_ = this->create_publisher<Image>(camera_config_.debug_image_topic, 10);
        }
    }
    
    void setupSubscribers()
    {
        // QoS settings (align with RELIABLE publishers)
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = 10;

        if (time_sync_mode_ == TimeSyncMode::HEADER) {
            // LiDAR bounding boxes subscriber for synchronizer
            lidar_sub_ = std::make_shared<message_filters::Subscriber<BoundingBox3DArray>>(
                this, lidar_boxes_topic_, qos_profile);

            // Camera detection subscriber
            detection_sub_ = std::make_shared<message_filters::Subscriber<DetectionArray>>(
                this, camera_config_.detections_topic, qos_profile);
            
            if (enable_debug_viz_) {
                // Image subscriber for visualization
                image_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
                    this, camera_config_.image_topic, qos_profile);
                
                // Setup synchronizer with images
                sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                    SyncPolicy(sync_queue_size_),
                    *lidar_sub_, *detection_sub_, *image_sub_);
                
                sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop_));
                
                sync_->registerCallback(
                    std::bind(&SingleIoUFusionNode::syncCallbackWithImage, this,
                             std::placeholders::_1, std::placeholders::_2, 
                             std::placeholders::_3));
                
                RCLCPP_INFO(this->get_logger(), "Synchronizer (header) with image configured for topics:");
                RCLCPP_INFO(this->get_logger(), "  LiDAR: %s", lidar_boxes_topic_.c_str());
                RCLCPP_INFO(this->get_logger(), "  Camera Det: %s", camera_config_.detections_topic.c_str());
                RCLCPP_INFO(this->get_logger(), "  Camera Img: %s", camera_config_.image_topic.c_str());
            } else {
                // Setup synchronizer without images
                sync_no_img_ = std::make_shared<message_filters::Synchronizer<SyncPolicyNoImg>>(
                    SyncPolicyNoImg(sync_queue_size_),
                    *lidar_sub_, *detection_sub_);
                
                sync_no_img_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop_));
                
                sync_no_img_->registerCallback(
                    std::bind(&SingleIoUFusionNode::syncCallbackNoImage, this,
                             std::placeholders::_1, std::placeholders::_2));
                
                RCLCPP_INFO(this->get_logger(), "Synchronizer (header) without image configured for topics:");
                RCLCPP_INFO(this->get_logger(), "  LiDAR: %s", lidar_boxes_topic_.c_str());
                RCLCPP_INFO(this->get_logger(), "  Camera Det: %s", camera_config_.detections_topic.c_str());
            }
            RCLCPP_INFO(this->get_logger(), "Synchronizer setup complete");
        } else {
            // Arrival-time based synchronization using latest messages within slop
            rclcpp::QoS qos(10);
            qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
            
            // Raw LiDAR subscriber (cache + counter + trigger)
            lidar_raw_sub_ = this->create_subscription<BoundingBox3DArray>(
                lidar_boxes_topic_, qos,
                [this](const BoundingBox3DArray::ConstSharedPtr& msg) {
                    lidar_msg_count_++;
                    last_lidar_boxes_ = msg;
                    last_lidar_recv_time_ = nowForSync();
                    tryProcessByArrival();
                });

            // Raw detection subscriber
            det_raw_sub_ = this->create_subscription<DetectionArray>(
                camera_config_.detections_topic, qos,
                [this](const DetectionArray::ConstSharedPtr& msg) {
                    det_msg_count_++;
                    last_detection_ = msg;
                    last_det_recv_time_ = nowForSync();
                });

            // Optional image subscriber for visualization
            if (enable_debug_viz_) {
                img_raw_sub_ = this->create_subscription<Image>(
                    camera_config_.image_topic, qos,
                    [this](const Image::ConstSharedPtr& msg) {
                        last_image_ = msg;
                        last_img_recv_time_ = nowForSync();
                    });
            }

            RCLCPP_INFO(this->get_logger(), "Arrival-time synchronization enabled (slop=%.3f s)", arrival_slop_);
        }
    }
    
    void syncCallbackWithImage(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const DetectionArray::ConstSharedPtr& detections,
        const Image::ConstSharedPtr& image)
    {
        RCLCPP_DEBUG(this->get_logger(), "syncCallbackWithImage called!");
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Processing fusion with %zu LiDAR boxes", lidar_boxes->boxes.size());
        
        processFusion(lidar_boxes, detections, image);
    }
    
    void syncCallbackNoImage(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const DetectionArray::ConstSharedPtr& detections)
    {
        RCLCPP_DEBUG(this->get_logger(), "syncCallbackNoImage called!");
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Processing fusion (no image) with %zu LiDAR boxes", lidar_boxes->boxes.size());
        
        processFusion(lidar_boxes, detections, nullptr);
    }
    
    void processFusion(
        const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
        const DetectionArray::ConstSharedPtr& detections,
        const Image::ConstSharedPtr& image)
    {
        // Project 3D boxes to camera's 2D plane
        std::vector<cv::Rect2f> projected_boxes;
        for (const auto& box3d : lidar_boxes->boxes) {
            auto box2d = project3DBoxTo2D(box3d, camera_config_);
            projected_boxes.push_back(box2d);
        }
        
        // Convert YOLO detections to 2D boxes
        std::vector<cv::Rect2f> yolo_boxes;
        for (const auto& det : detections->detections) {
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
                class_names[lidar_idx] = detections->detections[yolo_idx].class_name;
            }
        }
        
        // Visualize if enabled
        if (enable_debug_viz_ && image) {
            visualizeMatching3D(image, lidar_boxes, yolo_boxes, 
                               match_result, cost_matrix, camera_config_, class_names);
        }
        
        // Publish fused results
        publishFusedResults(lidar_boxes, class_names);
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
    
    void publishFusedResults(const BoundingBox3DArray::ConstSharedPtr& lidar_boxes,
                            const std::vector<std::string>& class_names)
    {
        // Convert to internal Cone representation first
        std::vector<calico::utils::Cone> cones;
        size_t dropped_unknown = 0;
        cones.reserve(lidar_boxes->boxes.size());
        for (size_t i = 0; i < lidar_boxes->boxes.size(); ++i) {
            const bool is_unknown = (class_names[i] == "Unknown");
            if (filter_unknown_ && is_unknown) {
                dropped_unknown++;
                continue;
            }
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
        msg.header = lidar_boxes->header;  // Preserve original frame and stamp by default
        if (override_fused_stamp_now_) {
            // Override timestamp to now based on selected clock
            const auto t = (time_sync_mode_ == TimeSyncMode::ARRIVAL_WALL)
                ? rclcpp::Clock(RCL_SYSTEM_TIME).now()
                : this->now();
            msg.header.stamp = t;            
        }
        
        fused_pub_->publish(msg);

        if (filter_unknown_) {
            RCLCPP_INFO(this->get_logger(),
                        "Published %zu fused cones to %s (dropped %zu Unknown)",
                        cones.size(), fused_output_topic_.c_str(), dropped_unknown);
        } else {
            RCLCPP_INFO(this->get_logger(), "Published %zu fused cones to %s",
                        cones.size(), fused_output_topic_.c_str());
        }
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
                            const Eigen::MatrixXd& cost_matrix,
                            const CameraConfig& cam_cfg,
                            const std::vector<std::string>& class_names)
    {
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
            draw3DBox(img, corners, cv::Scalar(0, 255, 0), 1);
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
                    
                    // Draw matched 3D box in cone color
                    auto corners = project3DBoxCorners(lidar_boxes->boxes[lidar_idx], cam_cfg);
                    draw3DBox(img, corners, color, 2);
                    
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
        debug_image_pub_->publish(*cv_ptr->toImageMsg());
    }
    
    // Helpers for arrival-based sync
    rclcpp::Time nowForSync() const {
        if (time_sync_mode_ == TimeSyncMode::ARRIVAL_WALL) {
            return rclcpp::Clock(RCL_SYSTEM_TIME).now();
        }
        return this->now();
    }

    bool withinSlop(const rclcpp::Time& a, const rclcpp::Time& b) const {
        const rclcpp::Duration d = (a > b) ? (a - b) : (b - a);
        return d <= rclcpp::Duration::from_seconds(arrival_slop_);
    }

    void tryProcessByArrival() {
        if (time_sync_mode_ == TimeSyncMode::HEADER) return; // Not used in header sync
        if (!last_lidar_boxes_) return;

        // Check if detection is available and fresh
        DetectionArray::ConstSharedPtr det_msg;
        if (last_detection_ && withinSlop(last_lidar_recv_time_, last_det_recv_time_)) {
            det_msg = last_detection_;
        } else {
            // Use empty detection array if camera data is stale/missing
            det_msg = std::make_shared<DetectionArray>();
            det_msg->header = last_lidar_boxes_->header;
        }

        // Check if debug image is available and fresh (optional)
        Image::ConstSharedPtr img_msg = nullptr;
        if (enable_debug_viz_ && last_image_ && withinSlop(last_lidar_recv_time_, last_img_recv_time_)) {
            img_msg = last_image_;
        }

        // Always process with available inputs; unmatched cones remain "Unknown"
        processFusion(last_lidar_boxes_, det_msg, img_msg);
    }
    
};

} // namespace nodes
} // namespace calico

RCLCPP_COMPONENTS_REGISTER_NODE(calico::nodes::SingleIoUFusionNode)

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<calico::nodes::SingleIoUFusionNode>();
    
    RCLCPP_INFO(node->get_logger(), "Single-Camera IoU Fusion Node started");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}

