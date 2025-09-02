#include "filc/projection_utils.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <thread>
#include <future>
#include <execution>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace filc {

ProjectionUtils::ProjectionUtils() {
#ifdef _OPENMP
    // 사용 가능한 코어 수에 맞춰 스레드 설정
    int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 8);
    omp_set_num_threads(num_threads);
#endif
}

ProjectionUtils::~ProjectionUtils() {
}

bool ProjectionUtils::convertPointCloud2ToPCL(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud) {
    
    try {
        pcl::fromROSMsg(*cloud_msg, *pcl_cloud);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error converting PointCloud2 to PCL: " << e.what() << std::endl;
        return false;
    }
}

std::vector<cv::Point3f> ProjectionUtils::transformPoints(
    const std::vector<cv::Point3f>& points,
    const Eigen::Matrix4d& transform_matrix) {
    
    std::vector<cv::Point3f> transformed_points;
    transformed_points.resize(points.size());
    
    // OpenMP가 활성화되어 있고 포인트가 충분히 많을 때만 병렬 처리
#ifdef _OPENMP
    if (points.size() > 1000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const auto& point = points[i];
            Eigen::Vector4d homogeneous_point(point.x, point.y, point.z, 1.0);
            Eigen::Vector4d transformed = transform_matrix * homogeneous_point;
            
            if (std::abs(transformed(3)) > 1e-8) {
                transformed_points[i] = cv::Point3f(
                    static_cast<float>(transformed(0) / transformed(3)),
                    static_cast<float>(transformed(1) / transformed(3)),
                    static_cast<float>(transformed(2) / transformed(3))
                );
            } else {
                transformed_points[i] = cv::Point3f(0, 0, 0); // Invalid point
            }
        }
    } else {
#endif
        // 직렬 처리 (포인트가 적거나 OpenMP가 없을 때)
        for (size_t i = 0; i < points.size(); ++i) {
            const auto& point = points[i];
            Eigen::Vector4d homogeneous_point(point.x, point.y, point.z, 1.0);
            Eigen::Vector4d transformed = transform_matrix * homogeneous_point;
            
            if (std::abs(transformed(3)) > 1e-8) {
                transformed_points[i] = cv::Point3f(
                    static_cast<float>(transformed(0) / transformed(3)),
                    static_cast<float>(transformed(1) / transformed(3)),
                    static_cast<float>(transformed(2) / transformed(3))
                );
            } else {
                transformed_points[i] = cv::Point3f(0, 0, 0); // Invalid point
            }
        }
#ifdef _OPENMP
    }
#endif
    
    return transformed_points;
}

std::vector<ProjectedPoint> ProjectionUtils::projectPointsToImage(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& lidar_to_camera_transform,
    const cv::Size& image_size) {
    
    std::vector<ProjectedPoint> projected_points;
    
    if (!point_cloud || point_cloud->empty()) {
        return projected_points;
    }
    
    // Convert PCL points to OpenCV format for transformation
    std::vector<cv::Point3f> lidar_points;
    std::vector<float> intensities;
    
    for (const auto& point : point_cloud->points) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            lidar_points.emplace_back(point.x, point.y, point.z);
            intensities.push_back(point.intensity);
        }
    }
    
    if (lidar_points.empty()) {
        return projected_points;
    }
    
    // Transform points to camera coordinate system
    std::vector<cv::Point3f> camera_points = transformPoints(lidar_points, lidar_to_camera_transform);
    
    // Filter points that are behind the camera
    std::vector<cv::Point3f> valid_camera_points;
    std::vector<float> valid_intensities;
    std::vector<size_t> valid_indices;
    
    for (size_t i = 0; i < camera_points.size(); ++i) {
        if (camera_points[i].z > 0.1) { // Points must be at least 0.1m in front of camera
            valid_camera_points.push_back(camera_points[i]);
            valid_intensities.push_back(intensities[i]);
            valid_indices.push_back(i);
        }
    }
    
    if (valid_camera_points.empty()) {
        return projected_points;
    }
    
    // Project 3D points to 2D image plane
    std::vector<cv::Point2f> image_points;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::projectPoints(valid_camera_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    
    // Create ProjectedPoint objects
    projected_points.reserve(image_points.size());
    
    for (size_t i = 0; i < image_points.size(); ++i) {
        ProjectedPoint proj_point;
        proj_point.image_point = image_points[i];
        proj_point.world_point = valid_camera_points[i];
        proj_point.intensity = valid_intensities[i];
        proj_point.range = std::sqrt(
            valid_camera_points[i].x * valid_camera_points[i].x +
            valid_camera_points[i].y * valid_camera_points[i].y +
            valid_camera_points[i].z * valid_camera_points[i].z
        );
        
        // Check if point is within image bounds
        proj_point.valid = (image_points[i].x >= 0 && image_points[i].x < image_size.width &&
                           image_points[i].y >= 0 && image_points[i].y < image_size.height);
        
        projected_points.push_back(proj_point);
    }
    
    return projected_points;
}

cv::Vec3b ProjectionUtils::intensityToColor(float intensity, float min_intensity, float max_intensity) {
    return getRainbowColor(intensity, min_intensity, max_intensity);
}

cv::Vec3b ProjectionUtils::rangeToColor(float range, float min_range, float max_range) {
    return getRainbowColor(range, min_range, max_range);
}

cv::Mat ProjectionUtils::drawProjectedPoints(
    const cv::Mat& image,
    const std::vector<ProjectedPoint>& projected_points,
    const std::string& color_mode,
    float min_value,
    float max_value) {
    
    cv::Mat result_image = image.clone();
    
    for (const auto& proj_point : projected_points) {
        if (!proj_point.valid) continue;
        
        cv::Point2i pixel_point(
            static_cast<int>(std::round(proj_point.image_point.x)),
            static_cast<int>(std::round(proj_point.image_point.y))
        );
        
        cv::Vec3b color;
        if (color_mode == "intensity") {
            color = intensityToColor(proj_point.intensity, min_value, max_value);
        } else if (color_mode == "range") {
            color = rangeToColor(proj_point.range, min_value, max_value);
        } else {
            color = cv::Vec3b(0, 255, 0); // Default green
        }
        
        // Draw point with a small circle
        cv::circle(result_image, pixel_point, 2, cv::Scalar(color[0], color[1], color[2]), -1);
    }
    
    return result_image;
}

std::vector<ProjectedPoint> ProjectionUtils::filterPointsInFOV(
    const std::vector<ProjectedPoint>& points,
    const cv::Size& image_size,
    float margin) {
    
    std::vector<ProjectedPoint> filtered_points;
    
    for (const auto& point : points) {
        if (point.image_point.x >= margin &&
            point.image_point.x < image_size.width - margin &&
            point.image_point.y >= margin &&
            point.image_point.y < image_size.height - margin) {
            filtered_points.push_back(point);
        }
    }
    
    return filtered_points;
}

std::vector<ProjectedPoint> ProjectionUtils::filterPointsByDepth(
    const std::vector<ProjectedPoint>& points,
    float min_depth,
    float max_depth) {
    
    std::vector<ProjectedPoint> filtered_points;
    
    for (const auto& point : points) {
        if (point.world_point.z >= min_depth && point.world_point.z <= max_depth) {
            filtered_points.push_back(point);
        }
    }
    
    return filtered_points;
}

cv::Vec3b ProjectionUtils::getRainbowColor(float value, float min_val, float max_val) {
    if (max_val <= min_val) {
        return cv::Vec3b(128, 128, 128); // Gray for invalid range
    }
    
    // Normalize value to [0, 1]
    float normalized = std::clamp((value - min_val) / (max_val - min_val), 0.0f, 1.0f);
    
    // Convert to HSV color space for rainbow effect
    float hue = (1.0f - normalized) * 240.0f; // Blue to Red (240° to 0°)
    
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2, 255, 255)); // OpenCV uses H in [0,179]
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    
    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return color;
}

std::vector<ProjectedPoint> ProjectionUtils::projectPointsToImageOptimized(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& lidar_to_camera_transform,
    const cv::Size& image_size,
    int num_threads) {
    
    std::vector<ProjectedPoint> projected_points;
    
    if (!point_cloud || point_cloud->empty()) {
        return projected_points;
    }
    
    const size_t num_points = point_cloud->size();
    
    // 포인트가 너무 적으면 단일 스레드로 처리
    if (num_points < 5000 || num_threads <= 1) {
        return projectPointsToImage(point_cloud, camera_matrix, dist_coeffs, 
                                   lidar_to_camera_transform, image_size);
    }
    
    // 스레드 수를 안전하게 제한
    num_threads = std::min(num_threads, static_cast<int>(std::thread::hardware_concurrency()));
    num_threads = std::max(1, std::min(num_threads, 8)); // 최대 8개 스레드로 제한
    
    const size_t chunk_size = num_points / num_threads;
    if (chunk_size < 1000) {
        // 청크가 너무 작으면 단일 스레드로 처리
        return projectPointsToImage(point_cloud, camera_matrix, dist_coeffs, 
                                   lidar_to_camera_transform, image_size);
    }
    
    // 스레드 결과를 저장할 벡터
    std::vector<std::vector<ProjectedPoint>> thread_results(num_threads);
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions(num_threads, nullptr);
    
    // 스레드들 생성 및 실행
    for (int i = 0; i < num_threads; ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (i == num_threads - 1) ? num_points : (i + 1) * chunk_size;
        
        threads.emplace_back([i, start_idx, end_idx, &thread_results, &exceptions,
                             &point_cloud, &camera_matrix, &dist_coeffs, 
                             &lidar_to_camera_transform, &image_size]() {
            try {
                thread_results[i] = processPointChunk(
                    point_cloud, start_idx, end_idx, 
                    camera_matrix, dist_coeffs, 
                    lidar_to_camera_transform, image_size
                );
            } catch (...) {
                exceptions[i] = std::current_exception();
            }
        });
    }
    
    // 모든 스레드 완료 대기 (타임아웃 추가)
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // 예외 처리
    for (size_t i = 0; i < exceptions.size(); ++i) {
        if (exceptions[i]) {
            std::rethrow_exception(exceptions[i]);
        }
    }
    
    // 결과 합치기
    for (const auto& result : thread_results) {
        projected_points.insert(projected_points.end(), 
                              result.begin(), result.end());
    }
    
    return projected_points;
}

std::vector<ProjectedPoint> ProjectionUtils::processPointChunk(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
    size_t start_idx,
    size_t end_idx,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& lidar_to_camera_transform,
    const cv::Size& image_size) {
    
    std::vector<ProjectedPoint> chunk_results;
    std::vector<cv::Point3f> lidar_points;
    std::vector<float> intensities;
    
    // 청크 데이터 추출
    for (size_t i = start_idx; i < end_idx; ++i) {
        const auto& point = point_cloud->points[i];
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            lidar_points.emplace_back(point.x, point.y, point.z);
            intensities.push_back(point.intensity);
        }
    }
    
    if (lidar_points.empty()) {
        return chunk_results;
    }
    
    // 좌표 변환
    std::vector<cv::Point3f> camera_points = transformPoints(lidar_points, lidar_to_camera_transform);
    
    // 카메라 뒤쪽 포인트 필터링
    std::vector<cv::Point3f> valid_camera_points;
    std::vector<float> valid_intensities;
    
    valid_camera_points.reserve(camera_points.size());
    valid_intensities.reserve(intensities.size());
    
    for (size_t i = 0; i < camera_points.size(); ++i) {
        if (camera_points[i].z > 0.1f) {
            valid_camera_points.push_back(camera_points[i]);
            valid_intensities.push_back(intensities[i]);
        }
    }
    
    if (valid_camera_points.empty()) {
        return chunk_results;
    }
    
    // 이미지 평면 투영
    std::vector<cv::Point2f> image_points;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::projectPoints(valid_camera_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    
    // 결과 생성
    chunk_results.reserve(image_points.size());
    
    for (size_t i = 0; i < image_points.size(); ++i) {
        ProjectedPoint proj_point;
        proj_point.image_point = image_points[i];
        proj_point.world_point = valid_camera_points[i];
        proj_point.intensity = valid_intensities[i];
        
        const auto& wp = valid_camera_points[i];
        proj_point.range = std::sqrt(wp.x * wp.x + wp.y * wp.y + wp.z * wp.z);
        
        // 이미지 경계 체크
        proj_point.valid = (image_points[i].x >= 0 && image_points[i].x < image_size.width &&
                           image_points[i].y >= 0 && image_points[i].y < image_size.height);
        
        chunk_results.push_back(proj_point);
    }
    
    return chunk_results;
}

std::vector<ProjectedPoint> ProjectionUtils::projectPointsToImageSpatialOpt(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& lidar_to_camera_transform,
    const cv::Size& image_size,
    float frustum_near,
    float frustum_far) {
    
    std::vector<ProjectedPoint> projected_points;
    
    if (!point_cloud || point_cloud->empty()) {
        return projected_points;
    }
    
    // Frustum culling을 위한 카메라 파라미터 추출
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);
    
    // FOV 계산 (approximate)
    double half_fov_x = std::atan(image_size.width * 0.5 / fx);
    double half_fov_y = std::atan(image_size.height * 0.5 / fy);
    
    std::vector<cv::Point3f> lidar_points;
    std::vector<float> intensities;
    
    // Frustum culling을 적용한 포인트 필터링
    for (const auto& point : point_cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        
        // 카메라 좌표계로 변환
        Eigen::Vector4d lidar_point(point.x, point.y, point.z, 1.0);
        Eigen::Vector4d camera_point = lidar_to_camera_transform * lidar_point;
        
        // 카메라 앞쪽에 있는지 확인
        if (camera_point.z() < frustum_near || camera_point.z() > frustum_far) {
            continue;
        }
        
        // FOV 체크 (approximate)
        double angle_x = std::atan2(std::abs(camera_point.x()), camera_point.z());
        double angle_y = std::atan2(std::abs(camera_point.y()), camera_point.z());
        
        if (angle_x > half_fov_x || angle_y > half_fov_y) {
            continue;
        }
        
        lidar_points.emplace_back(point.x, point.y, point.z);
        intensities.push_back(point.intensity);
    }
    
    if (lidar_points.empty()) {
        return projected_points;
    }
    
    // 나머지는 일반 투영과 동일
    std::vector<cv::Point3f> camera_points = transformPoints(lidar_points, lidar_to_camera_transform);
    
    // 이미지 평면 투영
    std::vector<cv::Point2f> image_points;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::projectPoints(camera_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    
    // 결과 생성
    projected_points.reserve(image_points.size());
    
    for (size_t i = 0; i < image_points.size(); ++i) {
        ProjectedPoint proj_point;
        proj_point.image_point = image_points[i];
        proj_point.world_point = camera_points[i];
        proj_point.intensity = intensities[i];
        proj_point.range = std::sqrt(
            camera_points[i].x * camera_points[i].x +
            camera_points[i].y * camera_points[i].y +
            camera_points[i].z * camera_points[i].z
        );
        
        proj_point.valid = (image_points[i].x >= 0 && image_points[i].x < image_size.width &&
                           image_points[i].y >= 0 && image_points[i].y < image_size.height);
        
        projected_points.push_back(proj_point);
    }
    
    return projected_points;
}

std::vector<ProjectedPoint> ProjectionUtils::projectPointsToImageMemoryOpt(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& lidar_to_camera_transform,
    const cv::Size& image_size,
    bool use_memory_pool) {
    
    // 현재는 기본 구현과 동일하게 하고, 나중에 메모리 풀 최적화 추가 가능
    return projectPointsToImage(point_cloud, camera_matrix, dist_coeffs, 
                               lidar_to_camera_transform, image_size);
}

cv::Mat ProjectionUtils::drawProjectedPointsOptimized(
    const cv::Mat& image,
    const std::vector<ProjectedPoint>& projected_points,
    const std::string& color_mode,
    float min_value,
    float max_value,
    bool use_gpu) {
    
    // GPU 가속이 비활성화되어 있으면 일반 버전 사용
    if (!use_gpu) {
        return drawProjectedPoints(image, projected_points, color_mode, min_value, max_value);
    }
    
    // GPU 가속 버전 (현재는 일반 버전과 동일, 나중에 CUDA 구현 추가 가능)
    return drawProjectedPoints(image, projected_points, color_mode, min_value, max_value);
}

bool ProjectionUtils::isPointInFrustum(
    const cv::Point3f& point,
    const cv::Mat& camera_matrix,
    const cv::Size& image_size,
    float near_plane,
    float far_plane) {
    
    // 깊이 체크
    if (point.z < near_plane || point.z > far_plane) {
        return false;
    }
    
    // FOV 체크를 위한 카메라 파라미터
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    
    // FOV 계산
    double half_fov_x = std::atan(image_size.width * 0.5 / fx);
    double half_fov_y = std::atan(image_size.height * 0.5 / fy);
    
    // 각도 계산
    double angle_x = std::atan2(std::abs(point.x), point.z);
    double angle_y = std::atan2(std::abs(point.y), point.z);
    
    return (angle_x <= half_fov_x && angle_y <= half_fov_y);
}

// Thread-local storage 초기화
thread_local std::vector<cv::Point3f> ProjectionUtils::point_buffer_;
thread_local std::vector<float> ProjectionUtils::intensity_buffer_;
thread_local std::vector<cv::Point2f> ProjectionUtils::projection_buffer_;
thread_local ProjectionUtils::PerformanceMetrics ProjectionUtils::perf_metrics_;

ProjectionUtils::PerformanceMetrics ProjectionUtils::getLastPerformanceMetrics() {
    return perf_metrics_;
}

void ProjectionUtils::resetPerformanceMetrics() {
    perf_metrics_ = PerformanceMetrics{};
}

} // namespace filc 