#ifndef CALICO_UTILS_MESSAGE_CONVERTER_HPP
#define CALICO_UTILS_MESSAGE_CONVERTER_HPP

#include <vector>
#include <string>
#include <memory>
#include <custom_interface/msg/modified_float32_multi_array.hpp>
#include <custom_interface/msg/tracked_cone.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <yolo_msgs/msg/detection.hpp>
#include <yolo_msgs/msg/detection_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <Eigen/Core>
#include <kalman_filters/tracking/tracking_types.hpp>

namespace calico {
namespace utils {

/**
 * @brief Internal cone representation
 */
struct Cone {
    double x, y, z;
    std::string color;
    int id = -1;
    double confidence = 1.0;
    
    // Velocity information (optional)
    bool has_velocity = false;
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

/**
 * @brief YOLO detection representation
 */
struct Detection {
    int x, y, width, height;  // Bounding box
    std::string class_name;
    double confidence;
};

/**
 * @brief IMU data representation
 */
struct IMUData {
    double linear_accel_x, linear_accel_y, linear_accel_z;
    double angular_vel_x, angular_vel_y, angular_vel_z;
    geometry_msgs::msg::Quaternion orientation;
    double timestamp;
};

/**
 * @brief Message converter utility class
 * 
 * Handles conversion between ROS2 messages and internal representations
 */
class MessageConverter {
public:
    /**
     * @brief Convert ModifiedFloat32MultiArray to internal cone representation
     * @param msg Input ROS message
     * @return Vector of cones
     */
    static std::vector<Cone> fromModifiedFloat32MultiArray(
        const custom_interface::msg::ModifiedFloat32MultiArray& msg);
    
    /**
     * @brief Convert internal cones to ModifiedFloat32MultiArray
     * @param cones Vector of cones
     * @return ROS message
     */
    static custom_interface::msg::ModifiedFloat32MultiArray 
    toModifiedFloat32MultiArray(const std::vector<Cone>& cones);
    
    /**
     * @brief Convert DetectionArray to internal detection representation
     * @param msg Input ROS message
     * @return Vector of detections
     */
    static std::vector<Detection> fromDetectionArray(
        const yolo_msgs::msg::DetectionArray& msg);
    
    /**
     * @brief Convert TrackedConeArray to internal cone representation
     * @param msg Input ROS message
     * @return Vector of cones with tracking IDs
     */
    static std::vector<Cone> fromTrackedConeArray(
        const custom_interface::msg::TrackedConeArray& msg);
    
    /**
     * @brief Convert internal cones to TrackedConeArray
     * @param cones Vector of tracked cones
     * @return ROS message
     */
    static custom_interface::msg::TrackedConeArray 
    toTrackedConeArray(const std::vector<Cone>& cones);
    
    /**
     * @brief Extract IMU data from sensor_msgs::msg::Imu
     * @param msg Input IMU message
     * @return Internal IMU data representation
     */
    static IMUData fromImuMsg(const sensor_msgs::msg::Imu& msg);
    
    /**
     * @brief Get center point of YOLO bounding box
     * @param detection Detection with bounding box
     * @return Center point (x, y)
     */
    static std::pair<double, double> getDetectionCenter(const Detection& detection);
    
    /**
     * @brief Map YOLO class name to cone color
     * @param class_name YOLO class name
     * @return Cone color string
     */
    static std::string mapClassToColor(const std::string& class_name);
    
    /**
     * @brief Convert between Cone and kalman_filters Detection types
     */
    static kalman_filters::tracking::Detection coneToDetection(const Cone& cone);
    static Cone detectionToCone(const kalman_filters::tracking::Detection& detection);
    static std::vector<kalman_filters::tracking::Detection> conesToDetections(const std::vector<Cone>& cones);
    static std::vector<Cone> detectionsToCones(const std::vector<kalman_filters::tracking::Detection>& detections);
    
    /**
     * @brief Convert TrackedObject to Cone with velocity
     */
    static Cone trackedObjectToCone(const kalman_filters::tracking::TrackedObject& obj);
    static std::vector<Cone> trackedObjectsToCones(const std::vector<kalman_filters::tracking::TrackedObject>& objects);
    
    /**
     * @brief Convert IMUData between calico and kalman_filters formats
     */
    static kalman_filters::tracking::IMUData toKalmanIMUData(const IMUData& imu_data);
    static IMUData fromKalmanIMUData(const kalman_filters::tracking::IMUData& kf_imu_data);
};

} // namespace utils
} // namespace calico

#endif // CALICO_UTILS_MESSAGE_CONVERTER_HPP