#include "../include/cone_detection/cone_detection_node.h"
#include <memory>
#include <limits>

namespace LIDAR {

// OutlierFilter 클래스 생성자: ROS2 노드 초기화 및 설정
OutlierFilter::OutlierFilter()
    : Node("outlier_filter"), last_plane_coefs_(new pcl::ModelCoefficients) {
    
    // ROS2 파라미터 선언
    this->declare_parameter("topic_name", params_.topic_name);
    this->declare_parameter("frame_id_", params_.frame_id_);
    this->declare_parameter("x_threshold_enable", params_.x_threshold_enable);
    this->declare_parameter("y_threshold_enable", params_.y_threshold_enable);
    this->declare_parameter("z_threshold_enable", params_.z_threshold_enable);
    this->declare_parameter("x_threshold_min", params_.x_threshold_min);
    this->declare_parameter("x_threshold_max", params_.x_threshold_max);
    this->declare_parameter("y_threshold_min", params_.y_threshold_min);
    this->declare_parameter("y_threshold_max", params_.y_threshold_max);
    this->declare_parameter("z_threshold_min", params_.z_threshold_min);
    this->declare_parameter("z_threshold_max", params_.z_threshold_max);
    this->declare_parameter("min_distance", params_.min_distance);
    this->declare_parameter("max_distance", params_.max_distance);
    this->declare_parameter("intensity_threshold", params_.intensity_threshold);
    this->declare_parameter("plane_distance_threshold", params_.plane_distance_threshold);
    this->declare_parameter("roi_angle_min", params_.roi_angle_min);
    this->declare_parameter("roi_angle_max", params_.roi_angle_max);
    this->declare_parameter("voxel_leaf_size", params_.voxel_leaf_size);
    this->declare_parameter("ec_cluster_tolerance", params_.ec_cluster_tolerance);
    this->declare_parameter("ec_min_cluster_size", params_.ec_min_cluster_size);
    this->declare_parameter("ec_max_cluster_size", params_.ec_max_cluster_size);
    this->declare_parameter("pca_orientation_threshold", params_.pca_orientation_threshold);
    this->declare_parameter("min_cone_height", params_.min_cone_height);
    this->declare_parameter("max_cone_height", params_.max_cone_height);

    // Load parameters from Config file
    this->get_parameter("topic_name", params_.topic_name);
    this->get_parameter("frame_id_", params_.frame_id_);
    this->get_parameter("x_threshold_enable", params_.x_threshold_enable);
    this->get_parameter("y_threshold_enable", params_.y_threshold_enable);
    this->get_parameter("z_threshold_enable", params_.z_threshold_enable);
    this->get_parameter("x_threshold_min", params_.x_threshold_min);
    this->get_parameter("x_threshold_max", params_.x_threshold_max);
    this->get_parameter("y_threshold_min", params_.y_threshold_min);
    this->get_parameter("y_threshold_max", params_.y_threshold_max);
    this->get_parameter("z_threshold_min", params_.z_threshold_min);
    this->get_parameter("z_threshold_max", params_.z_threshold_max);
    this->get_parameter("min_distance", params_.min_distance);
    this->get_parameter("max_distance", params_.max_distance);
    this->get_parameter("intensity_threshold", params_.intensity_threshold);
    this->get_parameter("plane_distance_threshold", params_.plane_distance_threshold);
    this->get_parameter("roi_angle_min", params_.roi_angle_min);
    this->get_parameter("roi_angle_max", params_.roi_angle_max);
    this->get_parameter("voxel_leaf_size", params_.voxel_leaf_size);
    this->get_parameter("ec_cluster_tolerance", params_.ec_cluster_tolerance);
    this->get_parameter("ec_min_cluster_size", params_.ec_min_cluster_size);
    this->get_parameter("ec_max_cluster_size", params_.ec_max_cluster_size);
    this->get_parameter("pca_orientation_threshold", params_.pca_orientation_threshold);
    this->get_parameter("min_cone_height", params_.min_cone_height);
    this->get_parameter("max_cone_height", params_.max_cone_height);


    // Log loaded parameters for verification
    RCLCPP_INFO(this->get_logger(), "Loaded Parameters:");
    RCLCPP_INFO(this->get_logger(), "  topic_name: %s", params_.topic_name.c_str());
    RCLCPP_INFO(this->get_logger(), "  frame_id: %s", params_.frame_id_.c_str());
    RCLCPP_INFO(this->get_logger(), "  x_threshold_enable: %s", params_.x_threshold_enable ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  x_threshold_min: %.2f", params_.x_threshold_min);
    RCLCPP_INFO(this->get_logger(), "  x_threshold_max: %.2f", params_.x_threshold_max);
    RCLCPP_INFO(this->get_logger(), "  y_threshold_enable: %s", params_.y_threshold_enable ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  y_threshold_min: %.2f", params_.y_threshold_min);
    RCLCPP_INFO(this->get_logger(), "  y_threshold_max: %.2f", params_.y_threshold_max);
    RCLCPP_INFO(this->get_logger(), "  z_threshold_enable: %s", params_.z_threshold_enable ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  z_threshold_min: %.2f", params_.z_threshold_min);
    RCLCPP_INFO(this->get_logger(), "  z_threshold_max: %.2f", params_.z_threshold_max);
    RCLCPP_INFO(this->get_logger(), "  min_distance: %.2f", params_.min_distance);
    RCLCPP_INFO(this->get_logger(), "  max_distance: %.2f", params_.max_distance);
    RCLCPP_INFO(this->get_logger(), "  intensity_threshold: %.2f", params_.intensity_threshold);
    RCLCPP_INFO(this->get_logger(), "  plane_distance_threshold: %.2f", params_.plane_distance_threshold);
    RCLCPP_INFO(this->get_logger(), "  roi_angle_min: %.2f", params_.roi_angle_min);
    RCLCPP_INFO(this->get_logger(), "  roi_angle_max: %.2f", params_.roi_angle_max);
    RCLCPP_INFO(this->get_logger(), "  voxel_leaf_size: %.2f", params_.voxel_leaf_size);
    RCLCPP_INFO(this->get_logger(), "  ec_cluster_tolerance: %.2f", params_.ec_cluster_tolerance);
    RCLCPP_INFO(this->get_logger(), "  ec_min_cluster_size: %d", params_.ec_min_cluster_size);
    RCLCPP_INFO(this->get_logger(), "  ec_max_cluster_size: %d", params_.ec_max_cluster_size);
    RCLCPP_INFO(this->get_logger(), "  pca_orientation_threshold: %.2f", params_.pca_orientation_threshold);
    RCLCPP_INFO(this->get_logger(), "  min_cone_height: %.2f", params_.min_cone_height);
    RCLCPP_INFO(this->get_logger(), "  max_cone_height: %.2f", params_.max_cone_height);


    // 퍼블리셔 초기화 (마커, 정렬된 콘, 처리된 포인트 클라우드)
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker", 10);
    cones_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/sorted_cones", 10);
    cones_time_pub = this->create_publisher<custom_interface::msg::ModifiedFloat32MultiArray>("/sorted_cones_time", 10);
    pub_cones_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cones", 10);

    // 서브스크라이버 초기화 (포인트 클라우드 데이터 수신)
    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params_.topic_name, rclcpp::SensorDataQoS(), // <-- QoS '10' -> 'rclcpp::SensorDataQoS()'로 바꿈.
        std::bind(&OutlierFilter::callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Cone_detection_node has been started!!!!!!!!!!!!!!!!!!!");  // 노드 시작 로그 출력
}

// 콜백 함수: 수신된 포인트 클라우드 데이터를 처리
void OutlierFilter::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    Cloud::Ptr cloud_in(new Cloud), cloud_filtered(new Cloud);

    // ROS 메시지를 PCL 포인트 클라우드로 변환
    pcl::fromROSMsg(*msg, *cloud_in);

    // 여기에서 LiDAR 좌표계를 센서 좌표계로 변환
    lidarToSensorTransform(cloud_in);

    // 이상점 제거 및 필터링 수행
    filterPointCloud(cloud_in, cloud_filtered);

    // 필터링된 포인트 클라우드를 퍼블리싱
    publishCloud(pub_cones_cloud_, cloud_filtered, msg->header.stamp, "os_sensor");

    // 초기 클러스터링 수행
    std::vector<ConeDescriptor> initial_cones;
    clusterCones(cloud_filtered, initial_cones);

    // 검증 단계 수행
    std::vector<ConeDescriptor> validated_cones;
    // 지면 계수가 유효할 때만 검증 수행
    if (last_plane_coefs_ && !last_plane_coefs_->values.empty()) {
        validateCones(initial_cones, validated_cones, 
                     pcl::ModelCoefficients::ConstPtr(last_plane_coefs_));
    } else {
        RCLCPP_WARN_ONCE(this->get_logger(), "Ground plane coefficients not valid, skipping validation");
        validated_cones = initial_cones;
    }

    // 검증된 콘 정렬 및 결과 퍼블리싱
    std::vector<std::vector<double>> sorted_cones = sortCones(validated_cones);
    publishArray(cones_pub_, sorted_cones);
    publishArrayWithTimestamp(cones_time_pub, sorted_cones, msg->header.stamp, "os_sensor");

    // 콘 데이터를 기반으로 MarkerArray 발행
    // publishSortedConesMarkers(sorted_cones, params_.frame_id_);
    visualizeCones(validated_cones, "os_sensor");
}


void OutlierFilter::voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size) {
    pcl::VoxelGrid<Point> voxel_filter;
    voxel_filter.setInputCloud(cloud_in);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*cloud_out);
}


// 포인트 클라우드 필터링 함수
void OutlierFilter::filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out) {
    Cloud::Ptr downsampled_cloud(new Cloud);
    Cloud::Ptr roi_filtered_cloud(new Cloud);

    // Voxelization (downsampling)
    voxelizeCloud(cloud_in, downsampled_cloud, params_.voxel_leaf_size);

    // 1. ROI 각도 및 거리 필터링을 위한 PointIndices 생성
    pcl::PointIndices::Ptr roi_indices(new pcl::PointIndices);
    for (size_t i = 0; i < downsampled_cloud->points.size(); ++i) {
        const auto& point = downsampled_cloud->points[i];
        float angle = ROI_theta(point.y, point.x);
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        
        if ((params_.roi_angle_min <= angle && angle <= params_.roi_angle_max) &&
            (params_.min_distance <= distance && distance <= params_.max_distance) &&
            (params_.intensity_threshold <= point.intensity)) {
            roi_indices->indices.push_back(i);
        }
    }

    // ROI 필터링된 포인트 추출
    pcl::ExtractIndices<Point> extract_roi;
    extract_roi.setInputCloud(downsampled_cloud);
    extract_roi.setIndices(roi_indices);
    extract_roi.setNegative(false);
    extract_roi.filter(*roi_filtered_cloud);

    // 2. X, Y, Z 축 필터링 (PassThrough 필터 체인 사용)
    Cloud::Ptr filtered_cloud(new Cloud);
    pcl::PassThrough<Point> pass;
    
    if (params_.x_threshold_enable) {
        pass.setInputCloud(roi_filtered_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(params_.x_threshold_min, params_.x_threshold_max);
        pass.filter(*filtered_cloud);
        roi_filtered_cloud = filtered_cloud;
        filtered_cloud.reset(new Cloud);
    }
    
    if (params_.y_threshold_enable) {
        pass.setInputCloud(roi_filtered_cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(params_.y_threshold_min, params_.y_threshold_max);
        pass.filter(*filtered_cloud);
        roi_filtered_cloud = filtered_cloud;
        filtered_cloud.reset(new Cloud);
    }
    
    if (params_.z_threshold_enable) {
        pass.setInputCloud(roi_filtered_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(params_.z_threshold_min, params_.z_threshold_max);
        pass.filter(*filtered_cloud);
        roi_filtered_cloud = filtered_cloud;
    } else {
        filtered_cloud = roi_filtered_cloud;
    }

    // 필터링된 포인트 클라우드를 cloud_out에 복사
    *cloud_out = *filtered_cloud;

    // 3. 평면 제거를 위한 RANSAC 세그먼테이션
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
}

// LiDAR 좌표계를 센서 좌표계로 변환 (os_lidar to os_sensor)
void OutlierFilter::lidarToSensorTransform(Cloud::Ptr &cloud) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // 변환 행렬 설정:
    // [-1  0  0  0    ]  // X_sensor = -X_lidar
    // [ 0 -1  0  0    ]  // Y_sensor = -Y_lidar
    // [ 0  0  1  38.195]  // Z_sensor = Z_lidar + 38.195 mm (단위: mm -> m로 변환)
    // [ 0  0  0  1    ]
    
    // 회전 부분 설정 (X와 Y 축 반전)
    transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    
    // 이동 부분 설정 (Z축 오프셋)
    transform.translation() << 0.0f, 0.0f, 0.038195f; // mm를 m 단위로 변환 (38.195mm -> 0.038195m)

    // 포인트 클라우드에 변환 적용
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

// 클러스터링 수행 (콘 클러스터 식별)
void OutlierFilter::clusterCones(Cloud::Ptr &cloud_in, std::vector<ConeDescriptor> &cones) {
    if (cloud_in->empty()) {
        return; // 입력 클라우드가 비었으면 종료
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

    cones.reserve(cluster_indices.size());
    pcl::ExtractIndices<Point> extract;
    extract.setInputCloud(cloud_in);

    for (const auto &indices : cluster_indices) {
        ConeDescriptor cone;
        pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
        extract.setIndices(indices_ptr);
        extract.filter(*cone.cloud); // 클러스터 포인트 클라우드 추출
        if (!cone.cloud->empty()) { // 클러스터가 비어있지 않은 경우에만 처리
             cone.calculate(); // 기본 통계 계산 (mean 등)
             cones.push_back(cone);
        }
    }
}

// 클러스터된 콘을 정렬
std::vector<std::vector<double>> OutlierFilter::sortCones(const std::vector<ConeDescriptor> &cones) {
    std::vector<std::vector<double>> sorted_cones;
    // 각 클러스터의 무게중심 좌표(X, Y, Z)를 벡터에 추가
    for (const auto &cone : cones) {
        // Include cone.mean.z here
        sorted_cones.push_back({cone.mean.x, cone.mean.y, cone.mean.z}); // sorted_cones는 이제 각 클러스터의 중심 좌표(X, Y, Z)를 담음
    }

    // x축을 기준으로 정렬 (This sorting logic remains the same)
    std::sort(sorted_cones.begin(), sorted_cones.end(),
              [](const std::vector<double> &a, const std::vector<double> &b) {
                  return a[0] < b[0]; // Sort based on the first element (X)
              });

    return sorted_cones;
}

// 포인트 클라우드 퍼블리싱
void OutlierFilter::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
    Cloud::Ptr &cloud,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = timestamp;
    publisher->publish(cloud_msg);
}

// 정렬된 콘 데이터를 퍼블리싱
void OutlierFilter::publishArray(
    const rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr &publisher,
    const std::vector<std::vector<double>> &array) {
    std_msgs::msg::Float64MultiArray msg;

    // 메시지 레이아웃 설정
    msg.layout.dim.resize(2); // 2차원 배열 형태 
    if (!array.empty()) {
        msg.layout.dim[0].size = array.size(); // 행 개수(클러스터 개수)
        msg.layout.dim[1].size = array[0].size(); // 열 개수(각 클러스터의 x, y 좌표)
        msg.layout.dim[0].stride = array.size() * array[0].size(); // 전체 데이터 크기
        msg.layout.dim[1].stride = array[0].size(); // 각 클러스터 데이터 크기
    }
    // 데이터를 메시지의 배열에 추가
    for (const auto &row : array) {
        for (const auto &val : row) {
            msg.data.push_back(val); // x,y 좌표 순차적으로 추가
        }
    }

    publisher->publish(msg);
}

// 정렬된 콘 데이터를 타임스탬프와 함께 퍼블리싱
void OutlierFilter::publishArrayWithTimestamp(
    const rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr &publisher,
    const std::vector<std::vector<double>> &array,
    const rclcpp::Time &timestamp,
    const std::string& frame_id) {
    custom_interface::msg::ModifiedFloat32MultiArray msg;

    msg.header.stamp = timestamp;
    msg.header.frame_id = frame_id;
    
    // 메시지 레이아웃 설정
    msg.layout.dim.resize(2); // 2차원 배열 형태 
    if (!array.empty()) {
        msg.layout.dim[0].size = array.size(); // 행 개수(클러스터 개수)
        msg.layout.dim[1].size = array[0].size(); // 열 개수(각 클러스터의 x, y 좌표)
        msg.layout.dim[0].stride = array.size() * array[0].size(); // 전체 데이터 크기
        msg.layout.dim[1].stride = array[0].size(); // 각 클러스터 데이터 크기
        
        // Initialize class_names with "Unknown" for each cone
        msg.class_names.resize(array.size());
        std::fill(msg.class_names.begin(), msg.class_names.end(), "Unknown");
    }
    
    // 데이터를 메시지의 배열에 추가
    for (const auto &row : array) {
        for (const auto &val : row) {
            msg.data.push_back(val); // x,y 좌표 순차적으로 추가
        }
    }

    publisher->publish(msg);
}

// cones 클러스터를 시각화
void OutlierFilter::visualizeCones(const std::vector<ConeDescriptor> &cones, const std::string& frame_id) {
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;
    
    // Clear previous markers
    for (int i = 0; i < previous_marker_count_; ++i) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id;
        delete_marker.header.stamp = this->now();
        delete_marker.ns = "cones"; // Use the same namespace
        delete_marker.id = i;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        markers.markers.push_back(delete_marker);
    }

    // Add new markers
    for (const auto &cone : cones) {
        if (cone.valid) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = this->now();
            marker.ns = "cones";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = cone.mean.x;
            marker.pose.position.y = cone.mean.y;
            marker.pose.position.z = cone.mean.z;
            marker.scale.x = 0.3;
            marker.scale.y = 0.3;
            marker.scale.z = 0.3;
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
}

// sorted_cones 클러스터를 시각화
void OutlierFilter::publishSortedConesMarkers(const std::vector<std::vector<double>> &sorted_cones, const std::string& frame_id) {
    visualization_msgs::msg::MarkerArray markers;
    
    // 1. 기존 마커를 삭제
    for (int id = 0; id < previous_marker_count_; ++id) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id;
        delete_marker.header.stamp = this->now();
        delete_marker.ns = "sorted_cones";
        delete_marker.id = id;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        markers.markers.push_back(delete_marker);
    }

    // 2. 새로운 마커를 추가
    int id = 0;
    for (const auto &cone : sorted_cones) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = "sorted_cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Assign x, y from sorted_cones and set a fixed z value
        marker.pose.position.x = cone[0];
        marker.pose.position.y = cone[1];
        marker.pose.position.z = cone[2];
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;

        // Color settings (red as an example)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        markers.markers.push_back(marker);
    }

    // 3. 마커 갯수 갱신
    previous_marker_count_ = id;

    // 4. 마커 퍼블리싱
    marker_pub_->publish(markers);
}


// ROI 영역의 각도를 계산
float OutlierFilter::ROI_theta(float x, float y) {
    return std::atan2(y, x) * 180 / M_PI;
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

    // 지면 법선 벡터 추출 (정규화 필요 없음, PCL 계수는 보통 정규화되어 있음)
    Eigen::Vector3f ground_normal(plane_coefs->values[0], plane_coefs->values[1], plane_coefs->values[2]);
    // ground_normal.normalize(); // 필요시 정규화

    for (const auto& cone : initial_cones) {
        if (cone.cloud->size() < 3) { // PCA는 최소 3점 필요
             continue;
        }

        // --- 1. PCA 방향성 검증 ---
        pcl::PCA<Point> pca;
        pca.setInputCloud(cone.cloud);

        // 주축 계산 (가장 분산이 *작은* 축 = 수직 방향 축)
        Eigen::Vector3f cluster_axis = pca.getEigenVectors().col(2);

        // ground_normal과 cluster_axis가 평행한지 확인 (dot product 절대값)
        double dot_product_abs = std::abs(cluster_axis.dot(ground_normal));

        if (dot_product_abs < params_.pca_orientation_threshold) {
            // RCLCPP_DEBUG(this->get_logger(), "Cluster failed PCA check: dot=%.2f", dot_product_abs);
            continue; // 방향성 임계값 미달 시, 이 클러스터는 탈락
        }

        // --- 2. 높이 검증 ---
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        for(const auto& pt : cone.cloud->points) {
            if (pt.z < min_z) min_z = pt.z;
            if (pt.z > max_z) max_z = pt.z;
        }
        float height = max_z - min_z;

        if (height < params_.min_cone_height || height > params_.max_cone_height) {
             // RCLCPP_DEBUG(this->get_logger(), "Cluster failed height check: height=%.2f", height);
             continue; // 높이 범위 벗어나면 탈락
        }

        // --- 모든 검증 통과 ---
        validated_cones.push_back(cone);
        // RCLCPP_DEBUG(this->get_logger(), "Cluster PASSED validation. Dot=%.2f, Height=%.2f", dot_product_abs, height);
    }
    RCLCPP_INFO(this->get_logger(), "Validation finished: %zu / %zu cones passed.", validated_cones.size(), initial_cones.size());
}

}  // namespace LIDAR

// 프로그램 진입점 (main 함수)
int main(int argc, char **argv) {
    // ROS2 노드 초기화
    rclcpp::init(argc, argv);

    // OutlierFilter 노드 생성 및 실행
    auto node = std::make_shared<LIDAR::OutlierFilter>();
    rclcpp::spin(node);

    // ROS2 노드 종료
    rclcpp::shutdown();
    return 0;
}
