#ifndef filc_PROJECTION_UTILS_HPP
#define filc_PROJECTION_UTILS_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

namespace filc {

struct ProjectedPoint {
    cv::Point2f image_point;
    cv::Point3f world_point;
    float intensity;
    float range;
    bool valid;
};

class ProjectionUtils {
public:
    ProjectionUtils();
    ~ProjectionUtils();
    
    // Point cloud conversion
    static bool convertPointCloud2ToPCL(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud
    );
    
    // Coordinate transformation
    static std::vector<cv::Point3f> transformPoints(
        const std::vector<cv::Point3f>& points,
        const Eigen::Matrix4d& transform_matrix
    );
    
    // Camera projection - Original version
    static std::vector<ProjectedPoint> projectPointsToImage(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& lidar_to_camera_transform,
        const cv::Size& image_size
    );
    
    // Camera projection - Optimized multithreaded version
    static std::vector<ProjectedPoint> projectPointsToImageOptimized(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& lidar_to_camera_transform,
        const cv::Size& image_size,
        int num_threads = std::thread::hardware_concurrency()
    );
    
    // Spatial optimization: 공간 분할 기반 최적화
    static std::vector<ProjectedPoint> projectPointsToImageSpatialOpt(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& lidar_to_camera_transform,
        const cv::Size& image_size,
        float frustum_near = 0.1f,
        float frustum_far = 50.0f
    );
    
    // Memory optimization: 메모리 풀 기반 처리
    static std::vector<ProjectedPoint> projectPointsToImageMemoryOpt(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& lidar_to_camera_transform,
        const cv::Size& image_size,
        bool use_memory_pool = true
    );
    
    // Color mapping for intensity/range visualization
    static cv::Vec3b intensityToColor(float intensity, float min_intensity, float max_intensity);
    static cv::Vec3b rangeToColor(float range, float min_range, float max_range);
    
    // Image processing utilities
    static cv::Mat drawProjectedPoints(
        const cv::Mat& image,
        const std::vector<ProjectedPoint>& projected_points,
        const std::string& color_mode = "intensity",
        float min_value = 0.0f,
        float max_value = 100.0f
    );
    
    // Optimized drawing with GPU acceleration when available
    static cv::Mat drawProjectedPointsOptimized(
        const cv::Mat& image,
        const std::vector<ProjectedPoint>& projected_points,
        const std::string& color_mode = "intensity",
        float min_value = 0.0f,
        float max_value = 100.0f,
        bool use_gpu = false
    );
    
    // Filter points utilities
    static std::vector<ProjectedPoint> filterPointsInFOV(
        const std::vector<ProjectedPoint>& points,
        const cv::Size& image_size,
        float margin = 0.0f
    );
    
    static std::vector<ProjectedPoint> filterPointsByDepth(
        const std::vector<ProjectedPoint>& points,
        float min_depth = 0.1f,
        float max_depth = 100.0f
    );
    
    // Performance monitoring utilities
    struct PerformanceMetrics {
        double transformation_time_ms;
        double projection_time_ms;
        double filtering_time_ms;
        double total_time_ms;
        size_t processed_points;
        size_t valid_points;
    };
    
    static PerformanceMetrics getLastPerformanceMetrics();
    static void resetPerformanceMetrics();

private:
    // Rainbow colormap for visualization
    static cv::Vec3b getRainbowColor(float value, float min_val, float max_val);
    
    // Helper function for multithreaded processing
    static std::vector<ProjectedPoint> processPointChunk(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud,
        size_t start_idx,
        size_t end_idx,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& lidar_to_camera_transform,
        const cv::Size& image_size
    );
    
    // Frustum culling optimization
    static bool isPointInFrustum(
        const cv::Point3f& point,
        const cv::Mat& camera_matrix,
        const cv::Size& image_size,
        float near_plane,
        float far_plane
    );
    
    // Memory pool for optimization
    static thread_local std::vector<cv::Point3f> point_buffer_;
    static thread_local std::vector<float> intensity_buffer_;
    static thread_local std::vector<cv::Point2f> projection_buffer_;
    
    // Performance tracking
    static thread_local PerformanceMetrics perf_metrics_;
};

} // namespace filc

#endif // filc_PROJECTION_UTILS_HPP 