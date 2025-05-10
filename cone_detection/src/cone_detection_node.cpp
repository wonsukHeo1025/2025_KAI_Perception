#include "../include/cone_detection/cone_detection_node.h"
#include <memory>
#include <limits>

namespace LIDAR {

// OutlierFilter 클래스 생성자: ROS2 노드 초기화 및 설정
OutlierFilter::OutlierFilter()
    : Node("outlier_filter"), last_plane_coefs_(new pcl::ModelCoefficients) {
    
    std::vector<std::pair<std::string, std::string*>> str_params = {
        {"input_topic_name", &params_.input_topic_name}
    };
    std::vector<std::pair<std::string, bool*>> bool_params = {
        {"x_threshold_enable", &params_.x_threshold_enable},
        {"y_threshold_enable", &params_.y_threshold_enable},
        {"z_threshold_enable", &params_.z_threshold_enable}
    };
    std::vector<std::pair<std::string, int*>> int_params = {
        {"ec_min_cluster_size", &params_.ec_min_cluster_size},
        {"ec_max_cluster_size", &params_.ec_max_cluster_size}
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
        {"pca_orientation_threshold", &params_.pca_orientation_threshold},
        {"min_cone_height", &params_.min_cone_height},
        {"max_cone_height", &params_.max_cone_height}
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

    // 퍼블리셔 초기화
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker", 10);
    cones_time_pub = this->create_publisher<custom_interface::msg::ModifiedFloat32MultiArray>("/sorted_cones_time", 10);
    pub_cones_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cones", 10);
    pub_points_fixed_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ouster/points_fixed", 10);

    // 서브스크라이버 초기화 (포인트 클라우드 데이터 수신)
    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params_.input_topic_name, rclcpp::SensorDataQoS(),
        std::bind(&OutlierFilter::callback, this, std::placeholders::_1));

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
            } else if (pub_points_fixed_ && pub_points_fixed_->get_subscription_count() > 0) {
                sensor_msgs::msg::PointCloud2 points_fixed_msg;
                pcl::toROSMsg(*cloud_positive_x, points_fixed_msg);
                points_fixed_msg.header.stamp = msg->header.stamp;
                points_fixed_msg.header.frame_id = "os_sensor";
                pub_points_fixed_->publish(points_fixed_msg);
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
        std::vector<ConeDescriptor> initial_cones;
        try {
            if (cloud_filtered->size() < params_.ec_min_cluster_size) {
                RCLCPP_WARN(this->get_logger(), "Too few points for clustering: %zu", cloud_filtered->size());
                return;
            }
            clusterCones(cloud_filtered, initial_cones);
            
            if (initial_cones.empty()) {
                RCLCPP_INFO(this->get_logger(), "No cones detected");
                return;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in clustering: %s", e.what());
            return;
        }

        // 검증 단계 수행
        std::vector<ConeDescriptor> validated_cones;
        try {
            // 지면 계수가 유효할 때만 검증 수행
            if (last_plane_coefs_ && !last_plane_coefs_->values.empty()) {
                validateCones(initial_cones, validated_cones, 
                            pcl::ModelCoefficients::ConstPtr(last_plane_coefs_));
            } else {
                RCLCPP_WARN_ONCE(this->get_logger(), "Ground plane coefficients not valid, skipping validation");
                validated_cones = initial_cones;
            }
            
            if (validated_cones.empty()) {
                RCLCPP_INFO(this->get_logger(), "No validated cones");
                return;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in cone validation: %s", e.what());
            return;
        }

        // 검증된 콘 정렬 및 결과 퍼블리싱
        try {
            std::vector<std::vector<double>> sorted_cones = sortCones(validated_cones);
            publishArrayWithTimestamp(cones_time_pub, sorted_cones, msg->header.stamp, "os_sensor");

            // 콘 데이터를 기반으로 MarkerArray 발행
            visualizeCones(validated_cones, "os_sensor");
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
        // 최소 포인트 수 검사
        if (cloud_in->points.size() < static_cast<size_t>(params_.ec_min_cluster_size)) {
            RCLCPP_WARN(this->get_logger(), "Too few points for clustering: %zu", cloud_in->points.size());
            return;
        }
        
        // KD-Tree 생성
        pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
        tree->setInputCloud(cloud_in);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<Point> ec;
        ec.setClusterTolerance(params_.ec_cluster_tolerance);
        ec.setMinClusterSize(params_.ec_min_cluster_size);
        ec.setMaxClusterSize(params_.ec_max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_in);
        ec.extract(cluster_indices);

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
            pcl::toROSMsg(*cloud, cloud_msg);
            cloud_msg.header.frame_id = frame_id;
            cloud_msg.header.stamp = timestamp;
            publisher->publish(cloud_msg);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishCloud: %s", e.what());
    }
}

// 정렬된 콘 데이터를 타임스탬프와 함께 퍼블리싱
void OutlierFilter::publishArrayWithTimestamp(
    const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
    const std::vector<std::vector<double>> &array,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    
    if (!publisher) {
        return;
    }
    
    try {
        if (publisher->get_subscription_count() > 0) {
            custom_interface::msg::ModifiedFloat32MultiArray msg;

            msg.header.stamp = timestamp;
            msg.header.frame_id = frame_id;
            
            // 메시지 레이아웃 설정
            msg.layout.dim.resize(2);
            if (!array.empty()) {
                msg.layout.dim[0].size = array.size();
                msg.layout.dim[1].size = array[0].size();
                msg.layout.dim[0].stride = array.size() * array[0].size();
                msg.layout.dim[1].stride = array[0].size();
                
                // 기본값으로 "Unknown" 설정
                msg.class_names.resize(array.size(), "Unknown");
                
                // 데이터 추가
                msg.data.reserve(array.size() * array[0].size());  // 메모리 미리 할당
                for (const auto &row : array) {
                    for (const auto &val : row) {
                        // NaN 체크
                        if (std::isnan(val)) {
                            msg.data.push_back(0.0);  // NaN 대신 0으로 대체
                        } else {
                            msg.data.push_back(val);
                        }
                    }
                }
                
                publisher->publish(msg);
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishArrayWithTimestamp: %s", e.what());
    }
}

// cones 클러스터를 시각화
void OutlierFilter::visualizeCones(const std::vector<ConeDescriptor> &cones, const std::string& frame_id) {
    if (!marker_pub_ || marker_pub_->get_subscription_count() == 0 || cones.empty()) {
        return;
    }
    
    try {
        visualization_msgs::msg::MarkerArray markers;
        const auto current_time = this->now();
        markers.markers.reserve(previous_marker_count_ + cones.size());  // 메모리 미리 할당
        
        // 이전 마커 삭제
        for (int i = 0; i < previous_marker_count_; ++i) {
            visualization_msgs::msg::Marker delete_marker;
            delete_marker.header.frame_id = frame_id;
            delete_marker.header.stamp = current_time;
            delete_marker.ns = "cones";
            delete_marker.id = i;
            delete_marker.action = visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(delete_marker);
        }

        // 새 마커 추가
        int id = 0;
        for (const auto &cone : cones) {
            if (cone.valid) {
                // NaN 체크
                if (std::isnan(cone.mean.x) || std::isnan(cone.mean.y) || std::isnan(cone.mean.z)) {
                    continue;
                }
                
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = current_time;
                marker.ns = "cones";
                marker.id = id++;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = cone.mean.x;
                marker.pose.position.y = cone.mean.y;
                marker.pose.position.z = cone.mean.z;
                marker.scale.x = marker.scale.y = marker.scale.z = 0.3;
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 1.0;
                marker.lifetime = rclcpp::Duration::from_seconds(0.5);
                markers.markers.push_back(marker);
            }
        }
        
        previous_marker_count_ = id;
        marker_pub_->publish(markers);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in visualizeCones: %s", e.what());
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

// 콘 클러스터 검증 함수
void OutlierFilter::validateCones(
    const std::vector<ConeDescriptor>& initial_cones,
    std::vector<ConeDescriptor>& validated_cones,
    const pcl::ModelCoefficients::ConstPtr& plane_coefs)
{
    validated_cones.clear();
    if (!plane_coefs || plane_coefs->values.size() < 4) {
        RCLCPP_ERROR(this->get_logger(), "Invalid plane coefficients for validation.");
        return;
    }
    
    if (initial_cones.empty()) {
        return;
    }

    try {
        // 지면 법선 벡터 추출
        Eigen::Vector3f ground_normal(plane_coefs->values[0], plane_coefs->values[1], plane_coefs->values[2]);
        
        // 법선 벡터가 유효한지 확인 (매우 작은 값이 아닌지)
        const float MIN_NORMAL_LENGTH = 1e-6f;
        if (ground_normal.norm() < MIN_NORMAL_LENGTH) {
            RCLCPP_WARN(this->get_logger(), "Ground normal vector too small, skipping validation");
            validated_cones = initial_cones;
            return;
        }

        validated_cones.reserve(initial_cones.size());
        
        for (const auto& cone : initial_cones) {
            try {
                if (!cone.cloud || cone.cloud->empty() || cone.cloud->size() < 3) {
                    continue;  // PCA는 최소 3점 필요
                }

                // 1. PCA 방향성 검증
                pcl::PCA<Point> pca;
                pca.setInputCloud(cone.cloud);
                Eigen::Vector3f cluster_axis = pca.getEigenVectors().col(2);  // 가장 분산이 작은 축
                
                // NaN 체크
                if (std::isnan(cluster_axis(0)) || std::isnan(cluster_axis(1)) || std::isnan(cluster_axis(2))) {
                    continue;
                }
                
                double dot_product_abs = std::abs(cluster_axis.dot(ground_normal));

                if (dot_product_abs < params_.pca_orientation_threshold) {
                    continue;  // 방향성 임계값 미달
                }

                // 2. 높이 검증 - 포인트 사이의 Z 차이 계산
                float min_z = std::numeric_limits<float>::max();
                float max_z = std::numeric_limits<float>::lowest();
                
                bool has_valid_points = false;
                for(const auto& pt : cone.cloud->points) {
                    if (!std::isnan(pt.z)) {
                        min_z = std::min(min_z, pt.z);
                        max_z = std::max(max_z, pt.z);
                        has_valid_points = true;
                    }
                }
                
                if (!has_valid_points) {
                    continue;  // 유효 z값 없음
                }
                
                float height = max_z - min_z;

                if (height < params_.min_cone_height || height > params_.max_cone_height) {
                    continue;  // 높이 범위 벗어남
                }

                // 모든 검증 통과
                validated_cones.push_back(cone);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Exception in cone validation: %s", e.what());
                // 개별 콘 검증에 실패해도 계속 진행
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Validation finished: %zu / %zu cones passed.", validated_cones.size(), initial_cones.size());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in validateCones: %s", e.what());
        validated_cones = initial_cones;  // 에러 발생 시 초기 콘 그대로 반환
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
