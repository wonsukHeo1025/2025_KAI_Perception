// cone_detection_node.cpp
// ROS2 기반 콘 디텍션 노드 구현
// 라이다 포인트 클라우드 데이터를 받아 필터링, 클러스터링, 콘 후보 검출, 정렬 및 시각화를 수행
#include "/home/wonsuk1025/kai_ws/src/cone_detection/include/cone_detection/cone_detection_node.h"
#include <memory>

namespace LIDAR {

// OutlierFilter 클래스 생성자: ROS2 노드 초기화 및 설정
OutlierFilter::OutlierFilter()
    : Node("outlier_filter") {
    
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
    Cloud::Ptr cloud_in(new Cloud), cloud_out(new Cloud);

    // ROS 메시지를 PCL 포인트 클라우드로 변환
    pcl::fromROSMsg(*msg, *cloud_in);

    // 이상점 제거 및 필터링 수행
    filterPointCloud(cloud_in, cloud_out);

    // 필터링된 포인트 클라우드를 퍼블리싱
    publishCloud(pub_cones_cloud_, cloud_out, msg->header.stamp);

    // 클러스터링 및 결과 퍼블리싱
    std::vector<ConeDescriptor> cones;
    clusterCones(cloud_out, cones);

    // 콘 정렬 및 결과 퍼블리싱
    std::vector<std::vector<double>> sorted_cones = sortCones(cones);
    publishArray(cones_pub_, sorted_cones);
    publishArrayWithTimestamp(cones_time_pub, sorted_cones, msg->header.stamp);

    // 콘 데이터를 기반으로 MarkerArray 발행
    publishSortedConesMarkers(sorted_cones); // 추가된 부분

    // 콘 시각화
    visualizeCones(cones);
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

    // Voxelization (downsampling)
    voxelizeCloud(cloud_in, downsampled_cloud, params_.voxel_leaf_size);

    std::vector<Point> filtered_points;

    for (const auto &point : downsampled_cloud->points) {
        float angle = ROI_theta(point.y, point.x);
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

        // Config 파일에서 로드된 파라미터로 필터링
        if ((params_.roi_angle_min <= angle && angle <= params_.roi_angle_max) && // ROI 각도 범위
            (!params_.x_threshold_enable || (params_.x_threshold_min <= point.x && point.x <= params_.x_threshold_max)) && // X축 범위
            (!params_.y_threshold_enable || (params_.y_threshold_min <= point.y && point.y <= params_.y_threshold_max)) && // Y축 범위
            (!params_.z_threshold_enable || (params_.z_threshold_min <= point.z && point.z <= params_.z_threshold_max)) && // Z축 범위
            (params_.min_distance <= distance && distance <= params_.max_distance) && // 거리 범위
            (params_.intensity_threshold <= point.intensity)) { // 강도 범위

            filtered_points.push_back(point);
        }

    }

    // 필터링된 포인트 클라우드를 cloud_out에 복사
    cloud_out->points.clear();
    cloud_out->points.insert(cloud_out->points.end(), filtered_points.begin(), filtered_points.end());
    cloud_out->width = filtered_points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = downsampled_cloud->is_dense;

    // 평면 제거를 위한 RANSAC 세그먼테이션
    pcl::ModelCoefficients::Ptr plane_coefs(new pcl::ModelCoefficients); // 평면의 방정식 계수
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); // RANSAC -> 평면 포인트 명단
    pcl::SACSegmentation<Point> seg; // RANSAC 세그먼테이션 객체
    seg.setOptimizeCoefficients(true); // 계수 최적화
    seg.setModelType(pcl::SACMODEL_PLANE); // 평면 모델 타입
    seg.setMethodType(pcl::SAC_RANSAC); // RANSAC 방법 타입
    // 거리 허용 오차 설정
    seg.setDistanceThreshold(params_.plane_distance_threshold); // 특정 점이 평면에 포함되려면, 해당 점과 평면 사이의 거리가 이것 이하여야 함

    seg.setInputCloud(cloud_out); // 입력 포인트 클라우드
    seg.segment(*inliers, *plane_coefs); // 평면 세그멘테이션 수행

    // 평면 포인트 제거
    if (!inliers->indices.empty()) { // 평면 포인트 명단에 뭔가 있으면
        pcl::ExtractIndices<Point> extract;
        extract.setInputCloud(cloud_out); // 입력 포인트 클라우드
        extract.setIndices(inliers); // 평면 포인트 명단
        extract.setNegative(true); // 평면 포인트 제거
        extract.filter(*cloud_out); // 평면 포인트 제거
    }

    // 여기에서 180도 회전 보정
    // correctYawRotation(cloud_out);
}

// 필터링 후 180도 회전 보정
void OutlierFilter::correctYawRotation(Cloud::Ptr &cloud) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // Z축 기준으로 180도 회전
    transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));

    // 포인트 클라우드에 변환 적용
    pcl::transformPointCloud(*cloud, *cloud, transform);
}


// 클러스터링 수행 (콘 클러스터 식별)
void OutlierFilter::clusterCones(Cloud::Ptr &cloud_out, std::vector<ConeDescriptor> &cones) {
    // 클러스터링 객체 생성
    pcl::EuclideanClusterExtraction<Point> ec; // PCL 라이브러리에서 제공하는 유클리드 거리 기반 클러스터링 클래스
    std::vector<pcl::PointIndices> cluster_indices; // 클러스터링 결과인 각 클러스터 점 인덱스 목록을 담을 벡터

    // 클러스터링 파라미터 설정
    ec.setClusterTolerance(params_.ec_cluster_tolerance); // 한 클러스터 내 포인트 간 최대 거리 임계치
    ec.setMinClusterSize(params_.ec_min_cluster_size);    // 클러스터로 인정하기 위한 최소 포인트 수
    ec.setMaxClusterSize(params_.ec_max_cluster_size);    // 클러스터로 인정하기 위한 최대 포인트 수
    ec.setInputCloud(cloud_out); // 입력 포인트 클라우드
    
    // 클러스터링 수행
    ec.extract(cluster_indices); // 결과로 cluster_indices에 각 클러스터별로 포인트 목록이 저장

    // 클러스터 포인트 추출 준비
    pcl::ExtractIndices<Point> extract;
    extract.setInputCloud(cloud_out);

    cones.reserve(cluster_indices.size()); // 클러스터 정보를 담을 벡터(파라미터) 초기화

    // 각 클러스터의 인덱스를 이용해 ConeDescriptor 생성
    for (const auto &indices : cluster_indices) { // cluster_indices는 pcl::PointIndices의 모음, 각 포인트 목록에 대해 반복
        ConeDescriptor cone;
        pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices)); // 한 클러스터에 대한 포인터 인덱스를 PCL타입으로 변환
        extract.setIndices(indices_ptr); // extract 객체에 추출할 클러스터의 인덱스들만 추출하여 cone.cloud에 저장
        extract.filter(*cone.cloud); // indices_ptr에 해당하는 포인트만 cone.cloud에 추출 -> 현재 클러스터를 하나의 점 구름(cone.cloud)으로 분리
        cone.calculate(); // ConeDescriptor 내부 계산(무게중심, 평균 등), common_defs.h에서 정의된 함수
        cones.push_back(cone); // 완성된 ConeDescriptor를 벡터에 추가
    }
}

// 클러스터된 콘을 정렬
std::vector<std::vector<double>> OutlierFilter::sortCones(const std::vector<ConeDescriptor> &cones) {
    std::vector<std::vector<double>> sorted_cones;
    // 각 클러스터의 무게중심 좌표(X, Y)를 벡터에 추가
    for (const auto &cone : cones) {
        sorted_cones.push_back({cone.mean.x, cone.mean.y}); // sorted_cones는 벡터로 구성된 리스트, 각 클러스터의 중심 좌표(X, Y)를 담음
    }

    // x축을 기준으로 정렬
    std::sort(sorted_cones.begin(), sorted_cones.end(),
              [](const std::vector<double> &a, const std::vector<double> &b) {
                  return a[0] < b[0];
              });

    return sorted_cones;
}

// 포인트 클라우드 퍼블리싱
void OutlierFilter::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
    Cloud::Ptr &cloud,
    const rclcpp::Time &timestamp) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = params_.frame_id_;
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
    const rclcpp::Time &timestamp) {
    custom_interface::msg::ModifiedFloat32MultiArray msg;

    msg.header.stamp = timestamp;
    msg.header.frame_id = params_.frame_id_;
    
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
void OutlierFilter::visualizeCones(const std::vector<ConeDescriptor> &cones) {
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;

    for (const auto &cone : cones) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = params_.frame_id_;
        marker.header.stamp = this->now();
        marker.ns = "cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = cone.mean.x;
        marker.pose.position.y = cone.mean.y;
        marker.pose.position.z = 0.3;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
}

// sorted_cones 클러스터를 시각화
void OutlierFilter::publishSortedConesMarkers(const std::vector<std::vector<double>> &sorted_cones) {
    visualization_msgs::msg::MarkerArray markers;
    
    // 1. 기존 마커를 삭제
    for (int id = 0; id < previous_marker_count_; ++id) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = params_.frame_id_;
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
        marker.header.frame_id = params_.frame_id_;
        marker.header.stamp = this->now();
        marker.ns = "sorted_cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Assign x, y from sorted_cones and set a fixed z value
        marker.pose.position.x = cone[0];
        marker.pose.position.y = cone[1];
        marker.pose.position.z = 0.3;  // Fixed height for visualization
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