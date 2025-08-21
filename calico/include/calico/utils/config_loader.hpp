#ifndef CALICO_UTILS_CONFIG_LOADER_HPP
#define CALICO_UTILS_CONFIG_LOADER_HPP

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace calico {
namespace utils {

/**
 * @brief Configuration structure for multi-camera setup
 */
struct CameraConfig {
    std::string id;
    std::string detections_topic;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    Eigen::Matrix4d extrinsic_matrix;
};

/**
 * @brief Main configuration structure for CALICO
 */
struct CalicoConfig {
    // Topic names
    std::string cones_topic;
    std::string output_topic;
    
    // Matching parameters
    double max_matching_distance;
    
    // Camera configurations
    std::vector<CameraConfig> cameras;
    
    // QoS settings
    int history_depth;
    double sync_slop;
    int sync_queue_size;
    
    // File paths
    std::string config_folder;
    std::string camera_extrinsic_calibration_file;
    std::string camera_intrinsic_calibration_file;
};

/**
 * @brief Configuration loader for CALICO package
 * 
 * Loads YAML configuration files for the CALICO sensor fusion package
 */
class ConfigLoader {
public:
    ConfigLoader() = default;
    ~ConfigLoader() = default;
    
    /**
     * @brief Load main configuration from YAML file
     * @param config_file_path Path to the multi_hungarian_config.yaml file
     * @return Parsed configuration structure
     */
    CalicoConfig loadConfig(const std::string& config_file_path);
    
    /**
     * @brief Load camera intrinsic parameters from YAML file
     * @param intrinsic_file_path Path to the intrinsic calibration file
     * @return Map of camera ID to intrinsic parameters
     */
    std::unordered_map<std::string, std::pair<cv::Mat, cv::Mat>> 
    loadIntrinsics(const std::string& intrinsic_file_path);
    
    /**
     * @brief Load camera extrinsic parameters from YAML file
     * @param extrinsic_file_path Path to the extrinsic calibration file
     * @return Map of camera ID to extrinsic matrix
     */
    std::unordered_map<std::string, Eigen::Matrix4d> 
    loadExtrinsics(const std::string& extrinsic_file_path);
    
private:
    /**
     * @brief Parse camera matrix from YAML node
     * @param node YAML node containing camera matrix data
     * @return OpenCV camera matrix
     */
    cv::Mat parseCameraMatrix(const YAML::Node& node);
    
    /**
     * @brief Parse distortion coefficients from YAML node
     * @param node YAML node containing distortion data
     * @return OpenCV distortion coefficients
     */
    cv::Mat parseDistortionCoeffs(const YAML::Node& node);
    
    /**
     * @brief Parse 4x4 transformation matrix from YAML node
     * @param node YAML node containing matrix data
     * @return Eigen 4x4 transformation matrix
     */
    Eigen::Matrix4d parseExtrinsicMatrix(const YAML::Node& node);
};

} // namespace utils
} // namespace calico

#endif // CALICO_UTILS_CONFIG_LOADER_HPP