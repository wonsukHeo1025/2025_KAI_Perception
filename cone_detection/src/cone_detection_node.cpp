#include "cone_detection/cone_detection_node.h"
#include <memory>
#include <limits>
#include <cstring>
#include <cmath>

namespace LIDAR {

// OutlierFilter 클래스 생성자: ROS2 노드 초기화 및 설정
OutlierFilter::OutlierFilter()
    : Node("outlier_filter"), last_plane_coefs_(new pcl::ModelCoefficients), 
      persistent_tree_(nullptr), last_cloud_size_(0) {
    
    std::vector<std::pair<std::string, std::string*>> str_params = {
        {"input_topic_name", &params_.input_topic_name}
    };
    std::vector<std::pair<std::string, bool*>> bool_params = {
        {"x_threshold_enable", &params_.x_threshold_enable},
        {"y_threshold_enable", &params_.y_threshold_enable},
        {"z_threshold_enable", &params_.z_threshold_enable},
        {"enable_tracking", &params_.enable_tracking},
        {"use_dbscan", &params_.use_dbscan}
    };
    std::vector<std::pair<std::string, int*>> int_params = {
        {"ec_min_cluster_size", &params_.ec_min_cluster_size},
        {"ec_max_cluster_size", &params_.ec_max_cluster_size},
        {"min_hits_before_confirmation", &params_.min_hits_before_confirmation},
        {"max_age_before_deletion", &params_.max_age_before_deletion}
    };
    std::vector<std::pair<std::string, float*>> float_params = {
        {"x_threshold_min", &params_.x_threshold_min},
        {"x_threshold_max", &params_.x_threshold_max},
        {"y_threshold_min", &params_.y_threshold_min},
        {"y_threshold_max", &params_.y_threshold_max},
        {"z_threshold_min", &params_.z_threshold_min},
        {"z_threshold_max", &params_.z_threshold_max},
        {"min_distance", &params_.min_distance},
        {"max_distance", &params_.max_distance},
        {"intensity_threshold", &params_.intensity_threshold},
        {"plane_distance_threshold", &params_.plane_distance_threshold},
        {"roi_angle_min", &params_.roi_angle_min},
        {"roi_angle_max", &params_.roi_angle_max},
        {"voxel_leaf_size", &params_.voxel_leaf_size},
        {"ec_cluster_tolerance", &params_.ec_cluster_tolerance},
        {"min_cone_height", &params_.min_cone_height},
        {"max_cone_height", &params_.max_cone_height},
    };
    std::vector<std::pair<std::string, double*>> double_params = {
        {"max_association_distance", &params_.max_association_distance},
        {"ukf_p_initial_pos", &params_.ukf_p_initial_pos},
        {"ukf_p_initial_vel", &params_.ukf_p_initial_vel},
        {"ukf_r_measurement", &params_.ukf_r_measurement},
        {"ukf_q_pos", &params_.ukf_q_pos},
        {"ukf_q_vel", &params_.ukf_q_vel}
    };

    for (const auto& [name, value_ptr] : str_params) {
        this->declare_parameter(name, *value_ptr);
        this->get_parameter(name, *value_ptr);
        RCLCPP_INFO(this->get_logger(), "  %s: %s", name.c_str(), value_ptr->c_str());
    }
    
    for (const auto& [name, value_ptr] : bool_params) {
        this->declare_parameter(name, *value_ptr);
        this->get_parameter(name, *value_ptr);
        RCLCPP_INFO(this->get_logger(), "  %s: %s", name.c_str(), *value_ptr ? "true" : "false");
    }
    
    for (const auto& [name, value_ptr] : int_params) {
        this->declare_parameter(name, *value_ptr);
        this->get_parameter(name, *value_ptr);
        RCLCPP_INFO(this->get_logger(), "  %s: %d", name.c_str(), *value_ptr);
    }
    
    for (const auto& [name, value_ptr] : float_params) {
        this->declare_parameter(name, *value_ptr);
        this->get_parameter(name, *value_ptr);
        RCLCPP_INFO(this->get_logger(), "  %s: %.2f", name.c_str(), *value_ptr);
    }
    
    for (const auto& [name, value_ptr] : double_params) {
        this->declare_parameter(name, *value_ptr);
        this->get_parameter(name, *value_ptr);
        RCLCPP_INFO(this->get_logger(), "  %s: %.2f", name.c_str(), *value_ptr);
    }

    // 퍼블리셔 초기화
    // Legacy format - commented out for migration to TrackedConeArray
    // cones_time_pub = this->create_publisher<custom_interface::msg::ModifiedFloat32MultiArray>("/sorted_cones_time", 10);      // Original format for backward compatibility
    cones_time_v2_pub = this->create_publisher<custom_interface::msg::TrackedConeArray>("/cone/lidar", 10);         // New TrackedConeArray format
    cones_time_ukf_pub_ = this->create_publisher<custom_interface::msg::TrackedConeArray>("/cone/lidar/ukf", 10);
    pub_cones_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ouster/points/preprocessed", 10);
    bbox_publisher_ = this->create_publisher<vision_msgs::msg::BoundingBox3DArray>("/cone/lidar/box", 10);

    // 서브스크라이버 초기화 (포인트 클라우드 데이터 수신)
    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params_.input_topic_name, rclcpp::SensorDataQoS(),
        std::bind(&OutlierFilter::callback, this, std::placeholders::_1));

    // Initialize tracker if enabled
    if (params_.enable_tracking) {
        kalman_filters::tracking::TrackingConfig tracker_config;
        tracker_config.max_association_dist = params_.max_association_distance;
        tracker_config.min_hits = params_.min_hits_before_confirmation;
        tracker_config.max_age = params_.max_age_before_deletion;
        tracker_config.p_initial_pos = params_.ukf_p_initial_pos;
        tracker_config.p_initial_vel = params_.ukf_p_initial_vel;
        tracker_config.r_pos = params_.ukf_r_measurement;
        tracker_config.q_pos = params_.ukf_q_pos;
        tracker_config.q_vel = params_.ukf_q_vel;
        
        tracker_ = std::make_shared<kalman_filters::tracking::MultiTracker>(tracker_config);
        RCLCPP_INFO(this->get_logger(), "UKF tracking enabled");
    }

    RCLCPP_INFO(this->get_logger(), "Cone_detection_node has been started!");
}

// 콜백 함수: 수신된 포인트 클라우드 데이터를 처리
void OutlierFilter::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!msg) {
        RCLCPP_ERROR(this->get_logger(), "Received null point cloud message");
        return;
    }

    try {
        // 충분한 포인트가 있는지 검사
        if (msg->width * msg->height < 10) {  // 최소 10개 포인트 필요
            RCLCPP_WARN(this->get_logger(), "Received point cloud has too few points (%d)", msg->width * msg->height);
            return;
        }

        Cloud::Ptr cloud_in(new Cloud), cloud_filtered(new Cloud);

        // ROS 메시지를 PCL 포인트 클라우드로 변환
        pcl::fromROSMsg(*msg, *cloud_in);
        
        if (cloud_in->empty()) {
            RCLCPP_WARN(this->get_logger(), "Converted cloud is empty");
            return;
        }

        // LiDAR 좌표계를 센서 좌표계로 변환
        lidarToSensorTransform(cloud_in);


        // X축 필터링: x >= 0 인 포인트만 남김
        Cloud::Ptr cloud_positive_x(new Cloud);
        try {
            pcl::PassThrough<Point> pass_x;
            pass_x.setInputCloud(cloud_in);
            pass_x.setFilterFieldName("x");
            pass_x.setFilterLimits(0.0, std::numeric_limits<float>::max());
            pass_x.filter(*cloud_positive_x);
            
            if (cloud_positive_x->empty()) {
                RCLCPP_WARN(this->get_logger(), "Positive X filtering removed all points");
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in X filtering: %s", e.what());
        }

        // 이상점 제거 및 필터링 수행
        try {
            filterPointCloud(cloud_in, cloud_filtered);
            
            if (cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty, skipping further processing");
                return;
            }

            // 필터링된 포인트 클라우드를 퍼블리싱
            publishCloud(pub_cones_cloud_, cloud_filtered, msg->header.stamp, "os_sensor");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in filtering: %s", e.what());
            return;
        }

        // 초기 클러스터링 수행
        std::vector<ConeDescriptor> stage1_candidate_cones;
        try {
            if (cloud_filtered->size() < static_cast<size_t>(params_.ec_min_cluster_size)) {
                RCLCPP_WARN(this->get_logger(), "Too few points for clustering: %zu", cloud_filtered->size());
                return;
            }
            clusterCones(cloud_filtered, stage1_candidate_cones);
            
            if (stage1_candidate_cones.empty()) {
                RCLCPP_INFO(this->get_logger(), "No cones detected after stage 1 clustering");
                // Visualization moved to separate node
                return;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in clustering: %s", e.what());
            return;
        }

        // Use stage 1 candidates directly
        std::vector<ConeDescriptor> intermediate_cones = stage1_candidate_cones;

        // 최종 검증 단계 수행 (예: 높이, 지면과의 관계 등)
        std::vector<ConeDescriptor> final_validated_cones;
        Cloud::Ptr final_reconstructed_points_for_pub(new Cloud); // 최종 재구성된 포인트를 담을 클라우드

        try {
            if (last_plane_coefs_ && !last_plane_coefs_->values.empty()) {
                validateConesFinalChecks(intermediate_cones, final_validated_cones, 
                            pcl::ModelCoefficients::ConstPtr(last_plane_coefs_));
            } else {
                RCLCPP_WARN_ONCE(this->get_logger(), "Ground plane coefficients not valid, skipping final validation checks");
                final_validated_cones = intermediate_cones; // 지면 정보 없으면 중간 결과 그대로 사용
            }
            
            if (final_validated_cones.empty()) {
                RCLCPP_INFO(this->get_logger(), "No cones passed final validation checks");
                // Visualization moved to separate node
                return;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in final cone validation: %s", e.what());
            return;
        }

        // Removed point reconstruction logic as Stage 2 validation is disabled


        // 검증된 콘 정렬 및 결과 퍼블리싱
        try {
            // Dual publishing strategy: publish to both old and new formats
            auto sorted_cones = sortCones(final_validated_cones);
            // Legacy format - commented out for migration to TrackedConeArray
            // publishArrayWithTimestamp(cones_time_pub, sorted_cones, msg->header.stamp, "os_sensor");        // Old format
            publishTrackedConeArray(cones_time_v2_pub, final_validated_cones, msg->header.stamp, "os_sensor"); // New format
            
            // Publish bounding boxes
            publishBoundingBoxes(final_validated_cones, msg->header.stamp, "os_sensor");

            // Apply UKF tracking if enabled
            if (params_.enable_tracking && tracker_) {
                // Convert timestamp to seconds
                double timestamp_sec = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
                
                // Convert ConeDescriptor to Detection
                std::vector<kalman_filters::tracking::Detection> detections;
                for (const auto& cone : final_validated_cones) {
                    detections.emplace_back(cone.mean.x, cone.mean.y, cone.mean.z, "");
                }
                
                // Update tracker with new detections
                tracker_->update(detections, timestamp_sec);
                
                // Get tracked objects and convert back to ConeDescriptor
                auto tracked_objects = tracker_->getTrackedObjects();
                std::vector<ConeDescriptor> tracked_cones;
                
                for (const auto& obj : tracked_objects) {
                    ConeDescriptor cone;
                    cone.mean.x = obj.x;
                    cone.mean.y = obj.y;
                    cone.mean.z = obj.z;
                    // color/label not used in lidar clustering
                    // track_id managed separately if needed
                    cone.calculate();  // Update valid flag and other properties
                    tracked_cones.push_back(cone);
                }
                
                // Publish tracked cones
                if (!tracked_objects.empty()) {
                    publishTrackedConeArray(cones_time_ukf_pub_, tracked_objects, msg->header.stamp, "os_sensor");
                    
                    RCLCPP_DEBUG(this->get_logger(), "Published %zu tracked cones from %zu tracks", 
                                tracked_objects.size(), tracker_->getNumTracks());
                }
            }

            // Visualization moved to separate node
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in result publishing: %s", e.what());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Unhandled exception in callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in callback");
    }
}

void OutlierFilter::voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size) {
    if (!cloud_in || cloud_in->empty()) {
        cloud_out->clear();
        return;
    }
    
    try {
        pcl::VoxelGrid<Point> voxel_filter;
        voxel_filter.setInputCloud(cloud_in);
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_filter.filter(*cloud_out);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in voxelizeCloud: %s", e.what());
        cloud_out->clear();
    }
}

// 포인트 클라우드 필터링 함수
void OutlierFilter::filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out) {
    // 입력 cloud_in이 비어있으면 바로 리턴 (Early exit)
    if (!cloud_in || cloud_in->points.empty()) {
        RCLCPP_WARN_ONCE(this->get_logger(), "Input cloud is empty or null.");
        cloud_out->points.clear();
        cloud_out->width = 0;
        cloud_out->height = 1;
        cloud_out->is_dense = true;
        return;
    }

    try {
        // Voxelization (downsampling)
        Cloud::Ptr downsampled_cloud(new Cloud);
        voxelizeCloud(cloud_in, downsampled_cloud, params_.voxel_leaf_size);

        // Voxelization 후 비어있으면 리턴
        if (downsampled_cloud->points.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Downsampled cloud is empty.");
            cloud_out->points.clear();
            cloud_out->width = 0;
            cloud_out->height = 1;
            cloud_out->is_dense = true;
            return;
        }

        // 1. ROI 각도, 거리, 강도 필터링
        Cloud::Ptr roi_filtered_cloud(new Cloud);
        roi_filtered_cloud->points.reserve(downsampled_cloud->points.size() / 2);  // 예상 크기 할당

        for (const auto& point : downsampled_cloud->points) {
            if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
                continue;  // NaN 무시
            }
            
            float angle = ROI_theta(point.y, point.x);
            float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

            if ((params_.roi_angle_min <= angle && angle <= params_.roi_angle_max) &&
                (params_.min_distance <= distance && distance <= params_.max_distance) &&
                (params_.intensity_threshold <= point.intensity)) {
                roi_filtered_cloud->points.push_back(point);
            }
        }
        roi_filtered_cloud->width = roi_filtered_cloud->points.size();
        roi_filtered_cloud->height = 1;
        roi_filtered_cloud->is_dense = false;  // NaN 값이 있을 수 있음

        // ROI 필터링 후 비어있으면 리턴
        if (roi_filtered_cloud->points.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "ROI filtered cloud is empty.");
            cloud_out->points.clear();
            cloud_out->width = 0;
            cloud_out->height = 1;
            cloud_out->is_dense = true;
            return;
        }

        // 2. X, Y, Z 축 필터링 (PassThrough 필터 체인 사용)
        Cloud::Ptr current_filtered_cloud = roi_filtered_cloud;
        
        auto applyPassthroughFilter = [&](const std::string& field_name, bool enabled, float min_val, float max_val) -> bool {
            if (!enabled) return true;
            
            try {
                Cloud::Ptr temp_cloud(new Cloud);
                pcl::PassThrough<Point> pass;
                pass.setInputCloud(current_filtered_cloud);
                pass.setFilterFieldName(field_name);
                pass.setFilterLimits(min_val, max_val);
                pass.filter(*temp_cloud);
                
                if (temp_cloud->points.empty()) {
                    RCLCPP_WARN_ONCE(this->get_logger(), "Cloud empty after %s PassThrough filter.", field_name.c_str());
                    *cloud_out = *temp_cloud;
                    return false;
                }
                
                current_filtered_cloud = temp_cloud;
                return true;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Exception in %s PassThrough filter: %s", field_name.c_str(), e.what());
                return false;
            }
        };
        
        if (!applyPassthroughFilter("x", params_.x_threshold_enable, params_.x_threshold_min, params_.x_threshold_max) ||
            !applyPassthroughFilter("y", params_.y_threshold_enable, params_.y_threshold_min, params_.y_threshold_max) ||
            !applyPassthroughFilter("z", params_.z_threshold_enable, params_.z_threshold_min, params_.z_threshold_max)) {
            return;
        }

        // 필터링된 포인트 클라우드를 cloud_out에 복사
        *cloud_out = *current_filtered_cloud;

        // 최소 포인트 수 체크 - RANSAC에는 최소한의 포인트가 필요
        const size_t MIN_POINTS_FOR_RANSAC = 10;  // 임의의 값, 상황에 맞게 조정
        if (cloud_out->points.size() < MIN_POINTS_FOR_RANSAC) {
            RCLCPP_WARN(this->get_logger(), "Not enough points for RANSAC plane segmentation: %zu", cloud_out->points.size());
            last_plane_coefs_->values.clear();
            return;
        }

        // 3. 평면 제거를 위한 RANSAC 세그먼테이션
        try {
            pcl::ModelCoefficients::Ptr current_plane_coefs(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<Point> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(params_.plane_distance_threshold);
            seg.setInputCloud(cloud_out);
            seg.segment(*inliers, *current_plane_coefs);

            // 평면 포인트 제거
            if (!inliers->indices.empty()) {
                pcl::ExtractIndices<Point> extract_plane;
                extract_plane.setInputCloud(cloud_out);
                extract_plane.setIndices(inliers);
                extract_plane.setNegative(true);
                extract_plane.filter(*cloud_out);

                // 지면 계수 멤버 변수에 저장
                *last_plane_coefs_ = *current_plane_coefs;
            } else {
                RCLCPP_WARN_ONCE(this->get_logger(), "No ground plane found using RANSAC.");
                last_plane_coefs_->values.clear();
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in RANSAC plane segmentation: %s", e.what());
            last_plane_coefs_->values.clear();
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in filterPointCloud: %s", e.what());
        cloud_out->points.clear();
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in filterPointCloud");
        cloud_out->points.clear();
    }
}

// LiDAR 좌표계를 센서 좌표계로 변환 (os_lidar to os_sensor)
void OutlierFilter::lidarToSensorTransform(Cloud::Ptr &cloud) {
    if (!cloud || cloud->empty()) {
        return;
    }
    
    try {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        
        // 회전 부분 설정 (X와 Y 축 반전)
        transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
        
        // 이동 부분 설정 (Z축 오프셋)
        transform.translation() << 0.0f, 0.0f, 0.038195f; // 38.195mm -> 0.038195m
        
        // 포인트 클라우드에 변환 적용
        pcl::transformPointCloud(*cloud, *cloud, transform);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in lidarToSensorTransform: %s", e.what());
    }
}

// 클러스터링 수행 (콘 클러스터 식별)
void OutlierFilter::clusterCones(Cloud::Ptr &cloud_in, std::vector<ConeDescriptor> &cones) {
    cones.clear();
    
    if (!cloud_in || cloud_in->empty()) {
        return;
    }

    try {
        // Clustering parameters
        float cluster_tolerance = params_.ec_cluster_tolerance;
        int min_cluster_size = params_.ec_min_cluster_size;
        int max_cluster_size = params_.ec_max_cluster_size;

        // 최소 포인트 수 검사
        if (cloud_in->points.size() < static_cast<size_t>(min_cluster_size)) {
            RCLCPP_WARN(this->get_logger(), "Too few points for clustering: %zu (min_required: %d)", cloud_in->points.size(), min_cluster_size);
            return;
        }
        
        // KD-Tree 생성 with reuse optimization
        // Only recreate tree if needed (cloud size changed significantly)
        if (!persistent_tree_ || 
            std::abs((int)cloud_in->size() - (int)last_cloud_size_) > cloud_in->size() * 0.1) {
            persistent_tree_ = std::make_shared<pcl::search::KdTree<Point>>();
            last_cloud_size_ = cloud_in->size();
        }
        persistent_tree_->setInputCloud(cloud_in);

        std::vector<pcl::PointIndices> cluster_indices;
        
        if (params_.use_dbscan) {
            // Use DBSCAN clustering (more robust to noise)
            DBSCANClusterer dbscan(cluster_tolerance, min_cluster_size);  // eps = cluster_tolerance, min_points = min_cluster_size
            dbscan.setInputCloud(cloud_in);
            dbscan.setSearchMethod(persistent_tree_);
            dbscan.extract(cluster_indices);
        } else {
            // Use traditional Euclidean clustering
            pcl::EuclideanClusterExtraction<Point> ec;
            ec.setClusterTolerance(cluster_tolerance);
            ec.setMinClusterSize(min_cluster_size);
            ec.setMaxClusterSize(max_cluster_size);
            ec.setSearchMethod(persistent_tree_);
            ec.setInputCloud(cloud_in);
            ec.extract(cluster_indices);
        }

        if (cluster_indices.empty()) {
            RCLCPP_INFO(this->get_logger(), "No clusters found");
            return;
        }

        cones.reserve(cluster_indices.size());
        pcl::ExtractIndices<Point> extract;
        extract.setInputCloud(cloud_in);

        for (const auto &indices : cluster_indices) {
            try {
                // 최소 포인트 수 체크
                if (indices.indices.size() < 3) {  // PCA를 위한 최소 포인트 수
                    continue;
                }
                
                ConeDescriptor cone;
                pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
                extract.setIndices(indices_ptr);
                extract.filter(*cone.cloud);
                
                if (!cone.cloud->empty()) {
                    cone.calculate();
                    cones.push_back(cone);
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Exception processing cluster: %s", e.what());
                // 에러가 발생해도 다음 클러스터 계속 처리
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in clusterCones: %s", e.what());
    }
}

// 클러스터된 콘을 정렬
std::vector<std::vector<double>> OutlierFilter::sortCones(const std::vector<ConeDescriptor> &cones) {
    std::vector<std::vector<double>> sorted_cones;
    
    if (cones.empty()) {
        return sorted_cones;
    }
    
    try {
        sorted_cones.reserve(cones.size());
        
        // 각 클러스터의 무게중심 좌표(X, Y, Z) 추출
        for (const auto &cone : cones) {
            if (cone.valid) {  // 유효한 콘만 포함
                sorted_cones.push_back({cone.mean.x, cone.mean.y, cone.mean.z});
            }
        }

        // 정렬할 요소가 있는지 확인
        if (!sorted_cones.empty()) {
            // x축을 기준으로 정렬
            std::sort(sorted_cones.begin(), sorted_cones.end(),
                      [](const std::vector<double> &a, const std::vector<double> &b) {
                          return a[0] < b[0];
                      });
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in sortCones: %s", e.what());
    }

    return sorted_cones;
}

// 포인트 클라우드 퍼블리싱
void OutlierFilter::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
    Cloud::Ptr &cloud,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    
    if (!publisher || !cloud) {
        return;
    }
    
    try {
        if (publisher->get_subscription_count() > 0) {
            sensor_msgs::msg::PointCloud2 cloud_msg;
            
            {
                // Standard PCL to ROS conversion
                pcl::toROSMsg(*cloud, cloud_msg);
                cloud_msg.header.frame_id = frame_id;
                cloud_msg.header.stamp = timestamp;
            }
            
            publisher->publish(cloud_msg);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishCloud: %s", e.what());
    }
}

// Legacy publish function - commented out for migration to TrackedConeArray
// 정렬된 콘 데이터를 타임스탬프와 함께 퍼블리싱
// void OutlierFilter::publishArrayWithTimestamp(
//     const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
//     const std::vector<std::vector<double>> &array,
//     const rclcpp::Time &timestamp,
//     const std::string& frame_id) {
//     
//     if (!publisher) {
//         return;
//     }
//     
//     try {
//         if (publisher->get_subscription_count() > 0) {
//             custom_interface::msg::ModifiedFloat32MultiArray msg;
// 
//             msg.header.stamp = timestamp;
//             msg.header.frame_id = frame_id;
//             
//             // 메시지 레이아웃 설정
//             msg.layout.dim.resize(2);
//             if (!array.empty()) {
//                 msg.layout.dim[0].size = array.size();
//                 msg.layout.dim[1].size = array[0].size();
//                 msg.layout.dim[0].stride = array.size() * array[0].size();
//                 msg.layout.dim[1].stride = array[0].size();
//                 
//                 // 기본값으로 "Unknown" 설정
//                 msg.class_names.resize(array.size(), "Unknown");
//                 
//                 // 데이터 추가
//                 msg.data.reserve(array.size() * array[0].size());  // 메모리 미리 할당
//                 for (const auto &row : array) {
//                     for (const auto &val : row) {
//                         // Enhanced NaN/Inf validation (similar to TrackedConeArray validation)
//                         if (std::isnan(val) || std::isinf(val)) {
//                             msg.data.push_back(0.0);  // NaN/Inf replaced with 0
//                         } else {
//                             msg.data.push_back(val);
//                         }
//                     }
//                 }
//                 
//                 publisher->publish(msg);
//             }
//         }
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(this->get_logger(), "Exception in publishArrayWithTimestamp: %s", e.what());
//     }
// }

// 검증된 콘 데이터를 TrackedConeArray로 퍼블리싱
void OutlierFilter::publishTrackedConeArray(
    const rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr &publisher,
    const std::vector<ConeDescriptor> &cones,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    
    if (!publisher) {
        return;
    }
    
    try {
        if (publisher->get_subscription_count() > 0) {
            custom_interface::msg::TrackedConeArray msg;

            msg.header.stamp = timestamp;
            msg.header.frame_id = frame_id;
            
            // ConeDescriptor를 TrackedCone으로 변환
            msg.cones.reserve(cones.size());
            for (size_t i = 0; i < cones.size(); ++i) {
                const auto& cone = cones[i];
                custom_interface::msg::TrackedCone tracked_cone;
                
                // 위치 설정 (ConeDescriptor의 mean 값 사용) with NaN/Inf validation
                if (std::isnan(cone.mean.x) || std::isinf(cone.mean.x)) {
                    tracked_cone.position.x = 0.0;
                } else {
                    tracked_cone.position.x = cone.mean.x;
                }
                
                if (std::isnan(cone.mean.y) || std::isinf(cone.mean.y)) {
                    tracked_cone.position.y = 0.0;
                } else {
                    tracked_cone.position.y = cone.mean.y;
                }
                
                if (std::isnan(cone.mean.z) || std::isinf(cone.mean.z)) {
                    tracked_cone.position.z = 0.0;
                } else {
                    tracked_cone.position.z = cone.mean.z;
                }
                
                // 색상 설정 (이 단계에서는 알 수 없음)
                tracked_cone.color = "unknown";
                
                // Sequential tracking ID starting from 1 (sorted order)
                // Note: This is different from UKF track_id which are persistent across frames
                if (i > static_cast<size_t>(std::numeric_limits<int32_t>::max() - 1)) {
                    RCLCPP_WARN(this->get_logger(), "Track ID overflow: index %zu exceeds int32_t max", i);
                    tracked_cone.track_id = std::numeric_limits<int32_t>::max();
                } else {
                    tracked_cone.track_id = static_cast<int32_t>(i + 1);  // Start from 1
                }
                
                msg.cones.push_back(tracked_cone);
            }
            
            publisher->publish(msg);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishTrackedConeArray: %s", e.what());
    }
}

// 정렬된 콘 데이터를 TrackedConeArray로 퍼블리싱
void OutlierFilter::publishTrackedConeArray(
    const rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr &publisher,
    const std::vector<kalman_filters::tracking::TrackedObject> &tracked_objects,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    
    if (!publisher) {
        return;
    }
    
    try {
        if (publisher->get_subscription_count() > 0) {
            custom_interface::msg::TrackedConeArray msg;
            msg.header.stamp = timestamp;
            msg.header.frame_id = frame_id;
            
            // Convert tracked objects to TrackedCone messages
            for (const auto& obj : tracked_objects) {
                custom_interface::msg::TrackedCone cone;
                // Use persistent UKF track_id (different from sequential IDs in other overload)
                cone.track_id = obj.track_id;
                
                // Add NaN/Inf validation for tracked objects
                if (std::isnan(obj.x) || std::isinf(obj.x)) {
                    cone.position.x = 0.0;
                } else {
                    cone.position.x = obj.x;
                }
                
                if (std::isnan(obj.y) || std::isinf(obj.y)) {
                    cone.position.y = 0.0;
                } else {
                    cone.position.y = obj.y;
                }
                
                if (std::isnan(obj.z) || std::isinf(obj.z)) {
                    cone.position.z = 0.0;
                } else {
                    cone.position.z = obj.z;
                }
                
                cone.color = "Unknown"; // LiDAR doesn't have color info
                
                msg.cones.push_back(cone);
            }
            
            publisher->publish(msg);
            RCLCPP_DEBUG(this->get_logger(), "Published %zu tracked cones", msg.cones.size());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishTrackedConeArray: %s", e.what());
    }
}


// ROI 영역의 각도를 계산
float OutlierFilter::ROI_theta(float x, float y) {
    // NaN 체크
    if (std::isnan(x) || std::isnan(y)) {
        return 0.0f;
    }
    
    // 0으로 나누기 방지
    if (std::abs(x) < 1e-6 && std::abs(y) < 1e-6) {
        return 0.0f;
    }
    
    return std::atan2(y, x) * 180.0f / M_PI;
}

// 콘 클러스터 최종 검증 함수 (기존 validateCones에서 이름 변경 및 로직 일부 이전)
void OutlierFilter::validateConesFinalChecks(
    const std::vector<ConeDescriptor>& initial_cones,
    std::vector<ConeDescriptor>& validated_cones,
    const pcl::ModelCoefficients::ConstPtr& plane_coefs) // plane_coefs는 이제 사용 안 함, 향후 지면기반 높이 검증 등에 사용 가능
{
    validated_cones.clear();
    // if (!plane_coefs || plane_coefs->values.size() < 4) { // 지면 계수는 이 함수에서 필수는 아님
    //     RCLCPP_WARN(this->get_logger(), "Invalid plane coefficients for final validation. Skipping some checks or using all cones.");
    //     // validated_cones = initial_cones; // 필요 시 모든 콘을 통과시킬 수 있음
    //     // return;
    // }
    
    if (initial_cones.empty()) {
        return;
    }

    try {
        // Eigen::Vector3f ground_normal(plane_coefs->values[0], plane_coefs->values[1], plane_coefs->values[2]);
        // const float MIN_NORMAL_LENGTH = 1e-6f;
        // if (ground_normal.norm() < MIN_NORMAL_LENGTH) {
        //     RCLCPP_WARN(this->get_logger(), "Ground normal vector too small for final checks, using all cones.");
        //     validated_cones = initial_cones;
        //     return;
        // }

        validated_cones.reserve(initial_cones.size());
        
        for (const auto& cone : initial_cones) {
            try {
                // Basic cloud validity check
                // 또는 ConeDescriptor의 valid 플래그를 여기서 한번 더 확인할 수 있음
                if(!cone.valid) { // ConeDescriptor.calculate() 에서 설정된 valid 플래그 확인
                    RCLCPP_DEBUG(this->get_logger(), "Cone not valid based on initial descriptor calculation.");
                    continue;
                }

                // 높이 검증 - 포인트 사이의 Z 차이 계산
                // 이 높이 계산은 ConeDescriptor::calculate()에서 이미 수행되었을 수 있음.
                // 여기서는 최종적인 min/max_cone_height 파라미터와 비교.
                float min_z = std::numeric_limits<float>::max();
                float max_z = std::numeric_limits<float>::lowest();
                bool has_valid_points = false;

                if (!cone.cloud || cone.cloud->empty()) {
                    RCLCPP_WARN(this->get_logger(), "Cone cloud is empty in final checks.");
                    continue;
                }
                
                for(const auto& pt : cone.cloud->points) {
                    if (!std::isnan(pt.z)) {
                        min_z = std::min(min_z, pt.z);
                        max_z = std::max(max_z, pt.z);
                        has_valid_points = true;
                    }
                }
                
                if (!has_valid_points) {
                    RCLCPP_WARN(this->get_logger(), "No valid Z points in cone for height check.");
                    continue;
                }
                
                float height = max_z - min_z;

                if (height < params_.min_cone_height || height > params_.max_cone_height) {
                    RCLCPP_DEBUG(this->get_logger(), "Cone height %.2f not in range [%.2f, %.2f]", 
                                height, params_.min_cone_height, params_.max_cone_height);
                    continue;
                }

                // 모든 검증 통과
                validated_cones.push_back(cone);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Exception in individual cone final validation: %s", e.what());
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Final Validation: %zu / %zu cones passed.", validated_cones.size(), initial_cones.size());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in validateConesFinalChecks: %s", e.what());
        validated_cones = initial_cones;  // 에러 발생 시 중간 콘 그대로 반환 (선택적)
    }
}

// 바운딩 박스 퍼블리싱 함수
void OutlierFilter::publishBoundingBoxes(
    const std::vector<ConeDescriptor> &cones,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    
    if (!bbox_publisher_ || bbox_publisher_->get_subscription_count() == 0) {
        return;
    }
    
    try {
        auto bbox_array_msg = std::make_unique<vision_msgs::msg::BoundingBox3DArray>();
        bbox_array_msg->header.stamp = timestamp;
        bbox_array_msg->header.frame_id = frame_id;

        // 각 콘에 대해 바운딩 박스 생성
        for (const auto& cone : cones) {
            if (!cone.valid || !cone.cloud || cone.cloud->empty()) {
                continue;
            }

            // BoundingBox3D 메시지 생성
            vision_msgs::msg::BoundingBox3D bbox_msg;

            // 중심점을 콘의 중심으로 설정
            bbox_msg.center.position.x = cone.mean.x;
            bbox_msg.center.position.y = cone.mean.y;
            bbox_msg.center.position.z = cone.mean.z;

            // 방향 (회전 없음)
            bbox_msg.center.orientation.x = 0.0;
            bbox_msg.center.orientation.y = 0.0;
            bbox_msg.center.orientation.z = 0.0;
            bbox_msg.center.orientation.w = 1.0;

            // 고정 크기 설정 (0.3m x 0.3m x 0.7m)
            bbox_msg.size.x = 0.3;
            bbox_msg.size.y = 0.3;
            bbox_msg.size.z = 0.7;

            bbox_array_msg->boxes.push_back(bbox_msg);
        }

        // 바운딩 박스가 하나라도 있을 때만 퍼블리시
        if (!bbox_array_msg->boxes.empty()) {
            bbox_publisher_->publish(std::move(bbox_array_msg));
            RCLCPP_DEBUG(this->get_logger(), "Published %zu bounding boxes", bbox_array_msg->boxes.size());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishBoundingBoxes: %s", e.what());
    }
}


}  // namespace LIDAR

// 프로그램 진입점 (main 함수)
int main(int argc, char **argv) {
    try {
        rclcpp::init(argc, argv);
        auto node = std::make_shared<LIDAR::OutlierFilter>();
        rclcpp::spin(node);
        rclcpp::shutdown();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Exception in main: %s", e.what());
        return 1;
    } catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Unknown exception in main");
        return 1;
    }
    return 0;
}
