#include "calico/utils/projection_utils.hpp"
#include <opencv2/calib3d.hpp>
#include <rclcpp/rclcpp.hpp>

namespace calico {
namespace utils {

std::pair<std::vector<cv::Point2f>, std::vector<int>> ProjectionUtils::projectLidarToCamera(
    const std::vector<Point3D>& lidar_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const Eigen::Matrix4d& extrinsic_matrix) {
    
    std::vector<cv::Point2f> projected_points;
    std::vector<int> original_indices;
    
    if (lidar_points.empty()) {
        return std::make_pair(projected_points, original_indices);
    }
    
    // Transform LiDAR points to camera coordinate system
    std::vector<Point3D> camera_points = transformPoints(lidar_points, extrinsic_matrix);
    
    // Convert to OpenCV format and track original indices
    std::vector<cv::Point3f> cv_points;
    cv_points.reserve(camera_points.size());
    
    for (size_t i = 0; i < camera_points.size(); ++i) {
        // Only project points in front of the camera
        if (camera_points[i].z > 1e-3) {  // 1mm minimum distance (same as Python)
            cv_points.emplace_back(camera_points[i].x, camera_points[i].y, camera_points[i].z);
            original_indices.push_back(i);  // Store the original index
        }
    }
    
    if (cv_points.empty()) {
        return std::make_pair(projected_points, original_indices);
    }
    
    // Project points to image plane
    cv::projectPoints(cv_points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                     camera_matrix, dist_coeffs, projected_points);
    
    return std::make_pair(projected_points, original_indices);
}

std::vector<Point3D> ProjectionUtils::transformPoints(
    const std::vector<Point3D>& points,
    const Eigen::Matrix4d& transform_matrix) {
    
    std::vector<Point3D> transformed_points;
    transformed_points.reserve(points.size());
    
    for (const auto& pt : points) {
        // Convert to homogeneous coordinates
        Eigen::Vector4d pt_h(pt.x, pt.y, pt.z, 1.0);
        
        // Apply transformation
        Eigen::Vector4d pt_transformed = transform_matrix * pt_h;
        
        // Convert back to 3D
        transformed_points.emplace_back(
            pt_transformed(0),
            pt_transformed(1),
            pt_transformed(2)
        );
    }
    
    return transformed_points;
}

bool ProjectionUtils::isPointInImage(const cv::Point2f& point, 
                                   int image_width, 
                                   int image_height) {
    return point.x >= 0 && point.x < image_width &&
           point.y >= 0 && point.y < image_height;
}

std::pair<std::vector<cv::Point2f>, std::vector<int>> 
ProjectionUtils::filterProjectedPoints(const std::vector<cv::Point2f>& projected_points,
                                     const std::vector<int>& indices,
                                     int image_width,
                                     int image_height) {
    std::vector<cv::Point2f> filtered_points;
    std::vector<int> filtered_indices;
    
    for (size_t i = 0; i < projected_points.size(); ++i) {
        if (isPointInImage(projected_points[i], image_width, image_height)) {
            filtered_points.push_back(projected_points[i]);
            if (i < indices.size()) {
                filtered_indices.push_back(indices[i]);
            } else {
                filtered_indices.push_back(i);
            }
        }
    }
    
    return std::make_pair(filtered_points, filtered_indices);
}

cv::Mat ProjectionUtils::eigenToCV(const Eigen::Matrix4d& eigen_mat) {
    cv::Mat cv_mat(4, 4, CV_64F);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            cv_mat.at<double>(i, j) = eigen_mat(i, j);
        }
    }
    
    return cv_mat;
}

} // namespace utils
} // namespace calico