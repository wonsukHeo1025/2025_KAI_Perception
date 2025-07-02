#ifndef CALICO_UTILS_PROJECTION_UTILS_HPP
#define CALICO_UTILS_PROJECTION_UTILS_HPP

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace calico {
namespace utils {

/**
 * @brief 3D point structure
 */
struct Point3D {
    double x, y, z;
    Point3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

/**
 * @brief 2D point structure
 */
struct Point2D {
    double x, y;
    Point2D(double x = 0, double y = 0) : x(x), y(y) {}
};

/**
 * @brief Utility functions for 3D to 2D projection
 */
class ProjectionUtils {
public:
    /**
     * @brief Project 3D LiDAR points to 2D camera image plane
     * @param lidar_points 3D points in LiDAR coordinate system
     * @param camera_matrix Camera intrinsic matrix (3x3)
     * @param dist_coeffs Camera distortion coefficients
     * @param extrinsic_matrix LiDAR to camera transformation matrix (4x4)
     * @return Pair of: 1) Vector of 2D projected points, 2) Vector of original indices
     */
    static std::pair<std::vector<cv::Point2f>, std::vector<int>> projectLidarToCamera(
        const std::vector<Point3D>& lidar_points,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const Eigen::Matrix4d& extrinsic_matrix);
    
    /**
     * @brief Transform 3D points using transformation matrix
     * @param points Input 3D points
     * @param transform_matrix 4x4 transformation matrix
     * @return Transformed 3D points
     */
    static std::vector<Point3D> transformPoints(
        const std::vector<Point3D>& points,
        const Eigen::Matrix4d& transform_matrix);
    
    /**
     * @brief Check if a 2D point is within image bounds
     * @param point 2D point to check
     * @param image_width Image width in pixels
     * @param image_height Image height in pixels
     * @return True if point is within bounds, false otherwise
     */
    static bool isPointInImage(const cv::Point2f& point, 
                               int image_width, 
                               int image_height);
    
    /**
     * @brief Filter projected points to keep only those within image bounds
     * @param projected_points Vector of projected 2D points
     * @param indices Original indices of the points
     * @param image_width Image width in pixels
     * @param image_height Image height in pixels
     * @return Pair of filtered points and their corresponding indices
     */
    static std::pair<std::vector<cv::Point2f>, std::vector<int>> 
    filterProjectedPoints(const std::vector<cv::Point2f>& projected_points,
                         const std::vector<int>& indices,
                         int image_width,
                         int image_height);
    
private:
    /**
     * @brief Convert Eigen matrix to OpenCV format
     * @param eigen_mat Eigen matrix
     * @return OpenCV Mat
     */
    static cv::Mat eigenToCV(const Eigen::Matrix4d& eigen_mat);
};

} // namespace utils
} // namespace calico

#endif // CALICO_UTILS_PROJECTION_UTILS_HPP