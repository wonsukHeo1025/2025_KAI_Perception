#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>
#include <thread>
#include <limits>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "filc/config_loader.hpp"
#include "filc/projection_utils.hpp"
#include "filc/interpolation_utils.hpp"

// 성능 최적화를 위한 하드코딩된 설정들 (무조건 적용하면 좋은 최적화)
namespace filc {
namespace Performance {
    // QoS 최적화 - 센서 데이터용 최적 설정
    constexpr size_t SENSOR_QOS_DEPTH = 5;
    constexpr size_t IMAGE_QOS_DEPTH = 2;
    
    // 타이머 주기 최적화 (센서 데이터 처리 주기)
    constexpr std::chrono::milliseconds PROCESSING_INTERVAL{50}; // 20Hz
    
    // 메모리 미리 할당 (포인트 클라우드)
    constexpr size_t EXPECTED_POINTS_PER_SCAN = 32 * 1024; // OS1-32 최대 포인트 수
    
    // CPU 최적화 힌트
    #pragma GCC optimize("O3")
    #pragma GCC optimize("unroll-loops")
    #pragma GCC optimize("fast-math")
    
    // SIMD 자동 벡터화 활성화
    #ifdef __GNUC__
        #pragma GCC target("avx2")
    #endif
    
    // 자주 사용되는 함수들 인라인 힌트
    inline void optimizeMemoryAccess() {
        // 메모리 접근 패턴 최적화를 위한 프리페치 힌트
        __builtin_prefetch(nullptr, 0, 3);
    }
}
}

class MainFusionNode : public rclcpp::Node {
public:
    MainFusionNode() : Node("filc_interpolation_node") {
        RCLCPP_INFO(this->get_logger(), "filc OS1-32 Interpolation Node starting...");
        
        // Load configuration from YAML file
        if (!loadConfiguration()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load configuration");
            rclcpp::shutdown();
            return;
        }
        
        // Initialize sensor parameters
        initializeOusterSensorParams();
        
        RCLCPP_INFO(this->get_logger(), "Configuration Settings:");
        RCLCPP_INFO(this->get_logger(), "  Interpolation enabled: %s", enable_interpolation_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  Scale factor: %.2f", interpolation_scale_factor_);
        RCLCPP_INFO(this->get_logger(), "  Type: %s", interpolation_type_str_.c_str());
        RCLCPP_INFO(this->get_logger(), "  LiDAR mode: %d", lidar_mode_);
        RCLCPP_INFO(this->get_logger(), "  Pixels per column: %d", pixels_per_column_);
        
        // Setup QoS profiles
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
        auto image_qos = rclcpp::QoS(rclcpp::KeepLast(2)).best_effort();
        
        // Subscribe to Ouster range image (필수)
        try {
            range_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                range_image_topic_,
                image_qos,
                std::bind(&MainFusionNode::rangeImageCallback, this, std::placeholders::_1)
            );
            
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s", range_image_topic_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to subscribe to %s: %s", range_image_topic_.c_str(), e.what());
            rclcpp::shutdown();
            return;
        }
        
        // Subscribe to Ouster intensity image (선택적)
        try {
            intensity_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                signal_image_topic_,
                image_qos,
                std::bind(&MainFusionNode::intensityImageCallback, this, std::placeholders::_1)
            );
            
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s", signal_image_topic_.c_str());
            enable_intensity_processing_ = true;
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to subscribe to %s: %s", signal_image_topic_.c_str(), e.what());
            RCLCPP_WARN(this->get_logger(), "Continuing with range image only");
            enable_intensity_processing_ = false;
        }
        
        // Subscribe to original point cloud (새로 추가)
        try {
            original_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                original_points_topic_,
                sensor_qos,
                std::bind(&MainFusionNode::originalPointsCallback, this, std::placeholders::_1)
            );
            
            RCLCPP_INFO(this->get_logger(), "Subscribed to original points: %s", original_points_topic_.c_str());
            enable_point_augmentation_ = true;
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to subscribe to %s: %s", original_points_topic_.c_str(), e.what());
            RCLCPP_WARN(this->get_logger(), "Point augmentation disabled, using full reconstruction mode");
            enable_point_augmentation_ = false;
        }
        
        // Publishers for interpolated data
        if (enable_interpolation_) {
            interpolated_range_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                interpolated_range_topic_, image_qos);
            
            if (enable_intensity_processing_) {
                interpolated_intensity_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                    interpolated_signal_topic_, image_qos);
            }
            
            // Publisher for interpolated point cloud
            interpolated_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                interpolated_points_topic_, sensor_qos);
            
            // Publisher for augmented point cloud (원본 + 새로운 포인트들)
            if (enable_point_augmentation_) {
                augmented_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                    augmented_points_topic_, sensor_qos);
                RCLCPP_INFO(this->get_logger(), "Augmented points publisher: %s", augmented_points_topic_.c_str());
            }
            
            RCLCPP_INFO(this->get_logger(), "Interpolation publishers created");
            RCLCPP_INFO(this->get_logger(), "  Range: %s", interpolated_range_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Signal: %s", interpolated_signal_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Points: %s", interpolated_points_topic_.c_str());
        }
        
        // Initialize data
        range_image_available_ = false;
        intensity_image_available_ = false;
        original_points_available_ = false;
        
        // Timer to periodically process interpolation
        interpolation_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20Hz
            std::bind(&MainFusionNode::processInterpolation, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "filc OS1-32 Interpolation Node initialized successfully");
    }

private:
    bool loadConfiguration() {
        try {
            // Get package share directory
            std::string package_path = ament_index_cpp::get_package_share_directory("filc");
            std::string config_path = package_path + "/config/interpolation_config.yaml";
            
            RCLCPP_INFO(this->get_logger(), "Loading configuration from: %s", config_path.c_str());
            
            YAML::Node config = YAML::LoadFile(config_path);
            
            // Load interpolation settings
            auto interp_config = config["interpolation"];
            enable_interpolation_ = interp_config["enable"].as<bool>();
            interpolation_scale_factor_ = interp_config["scale_factor"].as<double>();
            interpolation_type_str_ = interp_config["type"].as<std::string>();
            preserve_edges_ = interp_config["preserve_edges"].as<bool>();
            edge_threshold_ = interp_config["edge_threshold"].as<double>();
            fill_invalid_pixels_ = interp_config["fill_invalid_pixels"].as<bool>();
            use_multithreading_ = interp_config["use_multithreading"].as<bool>();
            num_threads_ = interp_config["num_threads"].as<int>();
            
            // Load hybrid mode settings
            if (interp_config["hybrid_mode"]) {
                auto hybrid_config = interp_config["hybrid_mode"];
                enable_hybrid_mode_ = hybrid_config["enable"].as<bool>();
                front_start_angle_ = hybrid_config["front_start_angle"].as<double>();
                front_end_angle_ = hybrid_config["front_end_angle"].as<double>();
                rear_upsampling_method_ = hybrid_config["rear_upsampling"].as<std::string>();
                
                RCLCPP_INFO(this->get_logger(), "Hybrid mode enabled: front sector %.1f° - %.1f°", 
                           front_start_angle_, front_end_angle_);
            } else {
                enable_hybrid_mode_ = false;
                front_start_angle_ = 90.0;
                front_end_angle_ = 270.0;
                rear_upsampling_method_ = "NEAREST";
            }
            
            // Set interpolation type
            if (interpolation_type_str_ == "CUBIC") {
                interpolation_type_ = filc::InterpolationUtils::InterpolationType::CUBIC;
            } else if (interpolation_type_str_ == "BICUBIC") {
                interpolation_type_ = filc::InterpolationUtils::InterpolationType::BICUBIC;
            } else if (interpolation_type_str_ == "LANCZOS") {
                interpolation_type_ = filc::InterpolationUtils::InterpolationType::LANCZOS;
            } else if (interpolation_type_str_ == "ADAPTIVE") {
                interpolation_type_ = filc::InterpolationUtils::InterpolationType::ADAPTIVE;
            } else {
                interpolation_type_ = filc::InterpolationUtils::InterpolationType::LINEAR;
            }
            
            // Load Ouster sensor settings
            auto ouster_config = config["ouster_os1_32"];
            lidar_mode_ = ouster_config["lidar_mode"].as<int>();
            pixels_per_column_ = ouster_config["pixels_per_column"].as<int>();
            lidar_origin_to_beam_origin_mm_ = ouster_config["lidar_origin_to_beam_origin_mm"].as<double>();
            
            // Load beam altitude angles
            auto altitude_angles = ouster_config["beam_altitude_angles"];
            beam_altitude_angles_.clear();
            for (const auto& angle : altitude_angles) {
                beam_altitude_angles_.push_back(angle.as<double>());
            }
            
            // Load topic settings
            auto topics = config["topics"];
            auto input_topics = topics["input"];
            auto output_topics = topics["output"];
            
            range_image_topic_ = input_topics["range_image"].as<std::string>();
            signal_image_topic_ = input_topics["signal_image"].as<std::string>();
            original_points_topic_ = input_topics["points"].as<std::string>();
            interpolated_range_topic_ = output_topics["interpolated_range"].as<std::string>();
            interpolated_signal_topic_ = output_topics["interpolated_signal"].as<std::string>();
            interpolated_points_topic_ = output_topics["interpolated_points"].as<std::string>();
            augmented_points_topic_ = output_topics["augmented_points"].as<std::string>();
            
            // Load performance settings
            auto perf_config = config["performance"];
            enable_performance_monitoring_ = perf_config["enable_monitoring"].as<bool>();
            
            // Load coordinate transform optimizations
            if (perf_config["coordinate_transform"]) {
                auto coord_config = perf_config["coordinate_transform"];
                use_lookup_table_ = coord_config["use_lookup_table"].as<bool>();
                precompute_angles_ = coord_config["precompute_angles"].as<bool>();
                cache_beam_vectors_ = coord_config["cache_beam_vectors"].as<bool>();
            } else {
                use_lookup_table_ = false;
                precompute_angles_ = false;
                cache_beam_vectors_ = false;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading configuration: %s", e.what());
            return false;
        }
    }
    
    void initializeOusterSensorParams() {
        // 고도각을 라디안으로 변환
        beam_altitude_angles_rad_.resize(beam_altitude_angles_.size());
        for (size_t i = 0; i < beam_altitude_angles_.size(); ++i) {
            beam_altitude_angles_rad_[i] = beam_altitude_angles_[i] * M_PI / 180.0;
        }
        
        // 방위각 테이블 생성 (0도에서 360도)
        beam_azimuth_angles_rad_.resize(lidar_mode_);
        for (int i = 0; i < lidar_mode_; ++i) {
            beam_azimuth_angles_rad_[i] = 2.0 * M_PI * i / lidar_mode_;
        }
        
        RCLCPP_INFO(this->get_logger(), "Initialized %zu altitude angles and %zu azimuth angles", 
                   beam_altitude_angles_rad_.size(), beam_azimuth_angles_rad_.size());
        
        // 고도각 배열 정보 출력 (디버그용)
        RCLCPP_INFO(this->get_logger(), "Altitude angles (degrees):");
        RCLCPP_INFO(this->get_logger(), "  First: %.3f, Last: %.3f", 
                   beam_altitude_angles_[0], beam_altitude_angles_[beam_altitude_angles_.size()-1]);
        RCLCPP_INFO(this->get_logger(), "  Range: %.3f to %.3f", 
                   *std::min_element(beam_altitude_angles_.begin(), beam_altitude_angles_.end()),
                   *std::max_element(beam_altitude_angles_.begin(), beam_altitude_angles_.end()));
        
        // 고도각이 위에서 아래로 내림차순인지 확인
        bool is_descending = true;
        for (size_t i = 1; i < beam_altitude_angles_.size(); ++i) {
            if (beam_altitude_angles_[i] > beam_altitude_angles_[i-1]) {
                is_descending = false;
                break;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Altitude angles order: %s", 
                   is_descending ? "descending (top to bottom)" : "ascending (bottom to top)");
    }
    
    void rangeImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(range_image_mutex_);
            latest_range_image_ = msg;
            range_image_available_ = true;
        }
        
        if (enable_performance_monitoring_) {
            auto now = this->get_clock()->now();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Range image received: %dx%d, encoding: %s", 
                msg->width, msg->height, msg->encoding.c_str());
        }
    }
    
    void intensityImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(intensity_image_mutex_);
            latest_intensity_image_ = msg;
            intensity_image_available_ = true;
        }
        
        if (enable_performance_monitoring_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Intensity image received: %dx%d, encoding: %s", 
                msg->width, msg->height, msg->encoding.c_str());
        }
    }
    
    void originalPointsCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(original_points_mutex_);
            latest_original_points_ = msg;
            original_points_available_ = true;
        }
        
        if (enable_performance_monitoring_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Original point cloud received: %d points", 
                msg->width * msg->height);
        }
    }
    
    void processInterpolation() {
        if (!enable_interpolation_) {
            return;
        }
        
        sensor_msgs::msg::Image::SharedPtr range_msg, intensity_msg;
        sensor_msgs::msg::PointCloud2::SharedPtr original_points_msg;
        
        {
            std::lock_guard<std::mutex> lock1(range_image_mutex_);
            
            if (!latest_range_image_) {
                return; // Range 이미지가 필수
            }
            
            range_msg = latest_range_image_;
            
            // Intensity 이미지는 선택적
            if (enable_intensity_processing_) {
                std::lock_guard<std::mutex> lock2(intensity_image_mutex_);
                intensity_msg = latest_intensity_image_;
            }
            
            // 원본 포인트 클라우드는 선택적 (포인트 증강 모드용)
            if (enable_point_augmentation_) {
                std::lock_guard<std::mutex> lock3(original_points_mutex_);
                if (latest_original_points_) {
                    original_points_msg = latest_original_points_;
                }
            }
        }
        
        try {
            // Convert ROS range image to OpenCV
            cv_bridge::CvImagePtr range_cv_ptr = cv_bridge::toCvCopy(range_msg, "mono16");
            
            if (enable_hybrid_mode_) {
                // 하이브리드 접근법: 설정에 따른 전방 영역만 보간
                int width = range_cv_ptr->image.cols;
                
                // 각도를 열 인덱스로 변환
                int front_start_col = static_cast<int>((front_start_angle_ / 360.0) * width);
                int front_end_col = static_cast<int>((front_end_angle_ / 360.0) * width);
                
                // 전방 영역 추출
                cv::Mat front_range = range_cv_ptr->image(cv::Range::all(), cv::Range(front_start_col, front_end_col));
                cv::Mat front_intensity;
                
                if (enable_intensity_processing_ && intensity_msg) {
                    cv_bridge::CvImagePtr intensity_cv_ptr = cv_bridge::toCvCopy(intensity_msg, "mono16");
                    front_intensity = intensity_cv_ptr->image(cv::Range::all(), cv::Range(front_start_col, front_end_col));
                }
                
                // Configure interpolation for front region only
                filc::InterpolationUtils::InterpolationConfig config = filc::InterpolationUtils::createDefaultConfig();
                config.type = interpolation_type_;
                config.vertical_scale_factor = interpolation_scale_factor_;
                config.preserve_edges = preserve_edges_;
                config.edge_threshold = edge_threshold_;
                config.fill_invalid_pixels = fill_invalid_pixels_;
                config.num_threads = num_threads_;
                
                // Perform interpolation on front region only
                auto start_time = std::chrono::high_resolution_clock::now();
                
                filc::InterpolationUtils::InterpolationResult front_result;
                
                if (enable_intensity_processing_ && !front_intensity.empty()) {
                    // Range + Intensity 이미지 보간 (전방만)
                    front_result = filc::InterpolationUtils::interpolateOusterImages(
                        front_range, front_intensity, config);
                } else {
                    // Range 이미지만 보간 (전방만)
                    front_result = filc::InterpolationUtils::interpolateRangeImageOnly(
                        front_range, config);
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Hybrid interpolation completed: Front %ld -> %ld rows (%.2fx), %.2f ms",
                    front_result.original_height, front_result.interpolated_height, 
                    front_result.interpolation_factor, processing_time);
                
                // 새로운 방식: 원본 포인트 클라우드에 새로운 포인트만 추가
                if (enable_point_augmentation_ && original_points_msg) {
                    auto augmented_cloud = createAugmentedPointCloud(
                        original_points_msg,
                        front_result.interpolated_range,
                        front_result.interpolated_intensity,
                        front_start_col, front_end_col,
                        range_msg->header);
                    
                    if (augmented_pointcloud_pub_ && augmented_cloud) {
                        augmented_pointcloud_pub_->publish(*augmented_cloud);
                        
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Augmented point cloud published: %d total points", 
                            augmented_cloud->width * augmented_cloud->height);
                    }
                } else {
                    // Fallback: 기존 하이브리드 방식
                    auto hybrid_cloud = createHybridPointCloud(
                        range_cv_ptr->image, 
                        intensity_msg ? cv_bridge::toCvCopy(intensity_msg, "mono16")->image : cv::Mat(),
                        front_result.interpolated_range,
                        front_result.interpolated_intensity,
                        front_start_col, front_end_col,
                        range_msg->header);
                    
                    if (interpolated_pointcloud_pub_ && hybrid_cloud) {
                        interpolated_pointcloud_pub_->publish(*hybrid_cloud);
                        
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Hybrid point cloud published: %d points", 
                            hybrid_cloud->width * hybrid_cloud->height);
                    }
                }
                
                // Optionally publish the interpolated front region images for debugging
                if (interpolated_range_pub_) {
                    std_msgs::msg::Header header = range_msg->header;
                    auto interpolated_range_msg = 
                        cv_bridge::CvImage(header, "mono16", front_result.interpolated_range).toImageMsg();
                    interpolated_range_pub_->publish(*interpolated_range_msg);
                }

                if (interpolated_intensity_pub_ && !front_result.interpolated_intensity.empty()) {
                    std_msgs::msg::Header header = range_msg->header;
                    auto interpolated_intensity_msg = 
                        cv_bridge::CvImage(header, "mono16", front_result.interpolated_intensity).toImageMsg();
                    interpolated_intensity_pub_->publish(*interpolated_intensity_msg);
                }
                
            } else {
                // 기존 전체 영역 보간 방식 (하이브리드 모드가 비활성화된 경우)
                filc::InterpolationUtils::InterpolationConfig config = filc::InterpolationUtils::createDefaultConfig();
                config.type = interpolation_type_;
                config.vertical_scale_factor = interpolation_scale_factor_;
                config.preserve_edges = preserve_edges_;
                config.edge_threshold = edge_threshold_;
                config.fill_invalid_pixels = fill_invalid_pixels_;
                config.num_threads = num_threads_;
                
                auto start_time = std::chrono::high_resolution_clock::now();
                
                filc::InterpolationUtils::InterpolationResult result;
                
                if (enable_intensity_processing_ && intensity_msg) {
                    cv_bridge::CvImagePtr intensity_cv_ptr = cv_bridge::toCvCopy(intensity_msg, "mono16");
                    result = filc::InterpolationUtils::interpolateOusterImages(
                        range_cv_ptr->image, intensity_cv_ptr->image, config);
                } else {
                    result = filc::InterpolationUtils::interpolateRangeImageOnly(
                        range_cv_ptr->image, config);
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Full interpolation completed: %ld -> %ld rows (%.2fx), %.2f ms",
                    result.original_height, result.interpolated_height, 
                    result.interpolation_factor, processing_time);
                
                // Publish full interpolated images
                if (interpolated_range_pub_) {
                    std_msgs::msg::Header header = range_msg->header;
                    auto interpolated_range_msg = 
                        cv_bridge::CvImage(header, "mono16", result.interpolated_range).toImageMsg();
                    interpolated_range_pub_->publish(*interpolated_range_msg);
                }

                if (interpolated_intensity_pub_ && !result.interpolated_intensity.empty()) {
                    std_msgs::msg::Header header = range_msg->header;
                    auto interpolated_intensity_msg = 
                        cv_bridge::CvImage(header, "mono16", result.interpolated_intensity).toImageMsg();
                    interpolated_intensity_pub_->publish(*interpolated_intensity_msg);
                }
                
                // Convert to point cloud and publish
                if (interpolated_pointcloud_pub_) {
                    auto interpolated_cloud = convertRangeImageToPointCloud(
                        result.interpolated_range, 
                        result.interpolated_intensity, 
                        range_msg->header);
                    
                    if (interpolated_cloud) {
                        interpolated_pointcloud_pub_->publish(*interpolated_cloud);
                        
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Full interpolated point cloud published: %d points",
                            interpolated_cloud->width * interpolated_cloud->height);
                    }
                }
            }
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Interpolation error: %s", e.what());
        }
    }
    
    // 새로운 포인트 증강 함수 - 원본 포인트 클라우드에 보간된 새로운 행의 포인트만 추가
    sensor_msgs::msg::PointCloud2::SharedPtr createAugmentedPointCloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& original_cloud,
        const cv::Mat& interpolated_front_range,
        const cv::Mat& interpolated_front_intensity,
        int front_start_col, int front_end_col,
        const std_msgs::msg::Header& header) {
        
        if (!original_cloud || interpolated_front_range.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Invalid inputs for augmented point cloud creation");
            return nullptr;
        }
        
        // Convert original ROS point cloud to PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr original_pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*original_cloud, *original_pcl_cloud);
        
        // Create augmented cloud starting with original points
        auto augmented_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*original_pcl_cloud);
        augmented_cloud->header.frame_id = header.frame_id;
        augmented_cloud->header.stamp = pcl_conversions::toPCL(header.stamp);
        
        // 보간된 이미지의 정보
        int interpolated_height = interpolated_front_range.rows;
        int front_width = interpolated_front_range.cols;
        int original_height = pixels_per_column_; // 32개 원본 행
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Augmenting cloud: Original %zu points, Front interpolated: %dx%d, Scale: %.1fx",
            original_pcl_cloud->points.size(), front_width, interpolated_height, interpolation_scale_factor_);
        
        // 보간으로 새로 생긴 행들만 식별하여 포인트 추가
        int new_points_added = 0;
        
        for (int interpolated_row = 0; interpolated_row < interpolated_height; ++interpolated_row) {
            // 현재 보간된 행이 원본 행 사이에 있는 새로운 행인지 확인
            double original_row_position = static_cast<double>(interpolated_row) / interpolation_scale_factor_;
            int original_row_index = static_cast<int>(std::round(original_row_position));
            
            // 새로운 행인지 확인: 원본 행 위치와 충분히 떨어져 있는가?
            double distance_from_original = std::abs(original_row_position - original_row_index);
            
            if (distance_from_original > 0.25) { // 원본 행 사이의 새로운 행
                // 전방 180도 영역의 각 열에 대해 새로운 포인트 생성
                for (int front_col = 0; front_col < front_width; ++front_col) {
                    int original_col = front_start_col + front_col;
                    
                    pcl::PointXYZI new_point;
                    convertPixelToPoint(interpolated_front_range, interpolated_front_intensity,
                                      interpolated_row, front_col, original_col, new_point, true);
                    
                    // 유효한 포인트만 추가
                    if (std::isfinite(new_point.x) && std::isfinite(new_point.y) && std::isfinite(new_point.z)) {
                        augmented_cloud->points.push_back(new_point);
                        new_points_added++;
                    }
                }
            }
        }
        
        // PCL 클라우드 속성 업데이트
        augmented_cloud->width = augmented_cloud->points.size();
        augmented_cloud->height = 1; // 구조화되지 않은 포인트 클라우드
        augmented_cloud->is_dense = false;
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Point augmentation: Added %d new interpolated points to %zu original points (Total: %zu)",
            new_points_added, original_pcl_cloud->points.size(), augmented_cloud->points.size());
        
        // Convert back to ROS message
        auto augmented_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*augmented_cloud, *augmented_msg);
        augmented_msg->header = header;
        
        return augmented_msg;
    }
    
    // 새로운 하이브리드 포인트 클라우드 생성 함수
    sensor_msgs::msg::PointCloud2::SharedPtr createHybridPointCloud(
        const cv::Mat& original_range,
        const cv::Mat& original_intensity, 
        const cv::Mat& interpolated_front_range,
        const cv::Mat& interpolated_front_intensity,
        int front_start_col, int front_end_col,
        const std_msgs::msg::Header& header) {
        
        if (original_range.empty() || interpolated_front_range.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty images for hybrid point cloud creation");
            return nullptr;
        }
        
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->header.frame_id = header.frame_id;
        cloud->header.stamp = pcl_conversions::toPCL(header.stamp);
        
        int original_height = original_range.rows;
        int original_width = original_range.cols;
        int interpolated_height = interpolated_front_range.rows;
        int front_width = interpolated_front_range.cols;
        
        // 최종 포인트 클라우드는 보간된 높이를 기준으로 함
        cloud->width = original_width;
        cloud->height = interpolated_height;
        cloud->is_dense = false;
        cloud->points.resize(original_width * interpolated_height);
        
        // 후방 영역 (원본 해상도 사용, 보간된 높이만큼 확장)
        for (int row = 0; row < interpolated_height; ++row) {
            for (int col = 0; col < original_width; ++col) {
                size_t point_idx = row * original_width + col;
                auto& point = cloud->points[point_idx];
                
                // 전방 영역인지 확인
                if (col >= front_start_col && col < front_end_col) {
                    // 전방 영역: 보간된 데이터 사용
                    int front_col = col - front_start_col;
                    if (front_col < front_width && row < interpolated_height) {
                        convertPixelToPoint(interpolated_front_range, interpolated_front_intensity,
                                          row, front_col, col, point, true);
                    } else {
                        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
                        point.intensity = 0.0f;
                    }
                } else {
                    // 후방 영역: 원본 데이터 사용 (상하 확장)
                    int original_row = static_cast<int>(static_cast<double>(row) / interpolation_scale_factor_);
                    original_row = std::min(original_row, original_height - 1);
                    
                    convertPixelToPoint(original_range, original_intensity,
                                      original_row, col, col, point, false);
                }
            }
        }
        
        // Convert PCL cloud to ROS message
        auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *cloud_msg);
        cloud_msg->header = header;
        
        return cloud_msg;
    }
    
    // 픽셀을 3D 포인트로 변환하는 헬퍼 함수 (수정된 좌표 변환 로직)
    void convertPixelToPoint(const cv::Mat& range_image, const cv::Mat& intensity_image,
                           int row, int col, int original_col, pcl::PointXYZI& point, bool is_interpolated) {
        
        // Get range value (in mm)
        float range_mm = 0.0f;
        if (range_image.type() == CV_16U) {
            range_mm = static_cast<float>(range_image.at<uint16_t>(row, col));
        } else if (range_image.type() == CV_32F) {
            range_mm = range_image.at<float>(row, col);
        }
        
        float range_m = range_mm / 1000.0f; // Convert mm to meters
        
        // Skip invalid points
        if (range_m < 0.1f || range_m > 120.0f) {
            point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
            point.intensity = 0.0f;
            return;
        }
        
        // 수정된 Ouster 좌표 변환 로직
        
        // 1. 방위각 계산 (original_col 사용하여 올바른 각도 매핑)
        double azimuth_step = 2.0 * M_PI / static_cast<double>(lidar_mode_);
        double theta_encoder = static_cast<double>(original_col) * azimuth_step;  // 0 to 2π
        
        // 2. 고도각 계산
        double phi; // altitude angle in radians
        if (is_interpolated) {
            // 보간된 이미지에서 원본 행 위치를 역산
            double original_row_position = static_cast<double>(row) / interpolation_scale_factor_;
            
            // 고도각 배열에서 선형 보간
            int lower_beam = static_cast<int>(std::floor(original_row_position));
            int upper_beam = std::min(lower_beam + 1, static_cast<int>(beam_altitude_angles_rad_.size()) - 1);
            
            // 경계 체크
            lower_beam = std::max(0, std::min(lower_beam, static_cast<int>(beam_altitude_angles_rad_.size()) - 1));
            
            if (lower_beam == upper_beam) {
                phi = beam_altitude_angles_rad_[lower_beam];
            } else {
                double weight = original_row_position - lower_beam;
                phi = beam_altitude_angles_rad_[lower_beam] * (1.0 - weight) + 
                      beam_altitude_angles_rad_[upper_beam] * weight;
            }
        } else {
            // 원본 이미지: 직접 beam 인덱스 사용
            int beam_idx = std::max(0, std::min(row, static_cast<int>(beam_altitude_angles_rad_.size()) - 1));
            phi = beam_altitude_angles_rad_[beam_idx];
        }
        
        // 3. 3D 좌표 계산 (간소화된 변환)
        double cos_phi = std::cos(phi);
        double sin_phi = std::sin(phi);
        double cos_theta = std::cos(theta_encoder);
        double sin_theta = std::sin(theta_encoder);
        
        // Ouster LiDAR 좌표계: X축이 앞쪽, Y축이 왼쪽, Z축이 위쪽
        point.x = range_m * cos_phi * cos_theta;   // 전방이 양수
        point.y = range_m * cos_phi * sin_theta;   // 왼쪽이 양수  
        point.z = range_m * sin_phi;               // 위쪽이 양수
        
        // Get intensity value
        if (!intensity_image.empty() && row < intensity_image.rows && col < intensity_image.cols) {
            if (intensity_image.type() == CV_16U) {
                point.intensity = static_cast<float>(intensity_image.at<uint16_t>(row, col));
            } else if (intensity_image.type() == CV_32F) {
                point.intensity = intensity_image.at<float>(row, col);
            }
        } else {
            // Use range as intensity if no intensity image
            point.intensity = range_mm / 100.0f; // Normalize
        }
    }

    // 기존 convertRangeImageToPointCloud 함수도 수정된 로직으로 업데이트
    sensor_msgs::msg::PointCloud2::SharedPtr convertRangeImageToPointCloud(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        const std_msgs::msg::Header& header) {
        
        if (range_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty range image for point cloud conversion");
            return nullptr;
        }
        
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->header.frame_id = header.frame_id;
        cloud->header.stamp = pcl_conversions::toPCL(header.stamp);
        
        int height = range_image.rows;
        int width = range_image.cols;
        
        cloud->width = width;
        cloud->height = height;
        cloud->is_dense = false;
        cloud->points.resize(width * height);
        
        // Convert each pixel to 3D point using updated logic
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                size_t point_idx = row * width + col;
                auto& point = cloud->points[point_idx];
                
                convertPixelToPoint(range_image, intensity_image, row, col, col, point, enable_interpolation_);
            }
        }
        
        // Convert PCL cloud to ROS message
        auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *cloud_msg);
        cloud_msg->header = header;
        
        return cloud_msg;
    }

private:
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr range_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr intensity_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr original_points_sub_;
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr interpolated_range_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr interpolated_intensity_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr interpolated_pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr augmented_pointcloud_pub_;
    
    // Mutexes
    std::mutex range_image_mutex_;
    std::mutex intensity_image_mutex_;
    std::mutex original_points_mutex_;
    
    // Latest data
    sensor_msgs::msg::Image::SharedPtr latest_range_image_;
    sensor_msgs::msg::Image::SharedPtr latest_intensity_image_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_original_points_;
    
    // Status flags
    bool range_image_available_;
    bool intensity_image_available_;
    bool original_points_available_;
    bool enable_intensity_processing_;
    bool enable_point_augmentation_;
    
    // Configuration
    bool enable_interpolation_;
    double interpolation_scale_factor_;
    std::string interpolation_type_str_;
    filc::InterpolationUtils::InterpolationType interpolation_type_;
    bool preserve_edges_;
    double edge_threshold_;
    bool fill_invalid_pixels_;
    
    // Hybrid mode settings
    bool enable_hybrid_mode_;
    double front_start_angle_;
    double front_end_angle_;
    std::string rear_upsampling_method_;
    
    // Ouster sensor parameters
    int lidar_mode_;
    int pixels_per_column_;
    double lidar_origin_to_beam_origin_mm_;
    std::vector<double> beam_altitude_angles_;
    std::vector<double> beam_altitude_angles_rad_;
    std::vector<double> beam_azimuth_angles_rad_;
    
    // Performance parameters
    bool use_multithreading_;
    int num_threads_;
    bool enable_performance_monitoring_;
    
    // Coordinate transform optimizations
    bool use_lookup_table_;
    bool precompute_angles_;
    bool cache_beam_vectors_;
    
    // Topic names
    std::string range_image_topic_;
    std::string signal_image_topic_;
    std::string original_points_topic_;
    std::string interpolated_range_topic_;
    std::string interpolated_signal_topic_;
    std::string interpolated_points_topic_;
    std::string augmented_points_topic_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr interpolation_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MainFusionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 