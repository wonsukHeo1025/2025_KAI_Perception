#ifndef filc_CONFIG_LOADER_HPP
#define filc_CONFIG_LOADER_HPP

#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace filc {

struct CameraConfig {
    std::string image_topic;
    std::string projected_topic;
    std::string frame_id;
    cv::Size image_size;
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    Eigen::Matrix4d extrinsic_matrix;
};

struct LidarConfig {
    std::string lidar_topic;
    std::string frame_id;
};

struct GeneralConfig {
    std::string config_folder;
    std::string data_folder;
    std::string camera_intrinsic_calibration;
    std::string camera_extrinsic_calibration;
    double slop;
};

class ConfigLoader {
public:
    ConfigLoader();
    ~ConfigLoader();
    
    bool loadConfiguration(const std::string& config_file_path);
    
    // Getters
    const std::map<std::string, CameraConfig>& getCameraConfigs() const { return camera_configs_; }
    const LidarConfig& getLidarConfig() const { return lidar_config_; }
    const GeneralConfig& getGeneralConfig() const { return general_config_; }
    
    // Helper functions
    static cv::Mat yamlToMat(const YAML::Node& node);
    static Eigen::Matrix4d yamlToEigen4d(const YAML::Node& node);

private:
    std::map<std::string, CameraConfig> camera_configs_;
    LidarConfig lidar_config_;
    GeneralConfig general_config_;
    
    bool loadCameraIntrinsics(const std::string& intrinsic_file_path);
    bool loadCameraExtrinsics(const std::string& extrinsic_file_path);
};

} // namespace filc

#endif // filc_CONFIG_LOADER_HPP 