#include "calico/utils/config_loader.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>

namespace calico {
namespace utils {

CalicoConfig ConfigLoader::loadConfig(const std::string& config_file_path) {
    CalicoConfig config;
    
    try {
        YAML::Node yaml = YAML::LoadFile(config_file_path);
        YAML::Node hungarian_node = yaml["hungarian_association"];
        
        // Load topic names
        config.cones_topic = hungarian_node["cones_topic"].as<std::string>();
        config.output_topic = hungarian_node["output_topic"].as<std::string>();
        
        // Load matching parameters with validation
        config.max_matching_distance = hungarian_node["max_matching_distance"].as<double>();
        if (config.max_matching_distance <= 0.0 || config.max_matching_distance > 100.0) {
            throw std::runtime_error("Invalid max_matching_distance: must be between 0 and 100");
        }
        
        // Load camera configurations
        YAML::Node cameras_node = hungarian_node["cameras"];
        for (const auto& cam_node : cameras_node) {
            CameraConfig cam_config;
            cam_config.id = cam_node["id"].as<std::string>();
            cam_config.detections_topic = cam_node["detections_topic"].as<std::string>();
            config.cameras.push_back(cam_config);
        }
        
        // Load calibration file paths
        YAML::Node calib_node = hungarian_node["calibration"];
        config.config_folder = calib_node["config_folder"].as<std::string>();
        config.camera_extrinsic_calibration_file = calib_node["camera_extrinsic_calibration"].as<std::string>();
        config.camera_intrinsic_calibration_file = calib_node["camera_intrinsic_calibration"].as<std::string>();
        
        // Load QoS settings with validation
        YAML::Node qos_node = hungarian_node["qos"];
        config.history_depth = qos_node["history_depth"].as<int>();
        if (config.history_depth <= 0 || config.history_depth > 1000) {
            throw std::runtime_error("Invalid history_depth: must be between 1 and 1000");
        }
        
        config.sync_slop = qos_node["sync_slop"].as<double>();
        if (config.sync_slop < 0.0 || config.sync_slop > 10.0) {
            throw std::runtime_error("Invalid sync_slop: must be between 0 and 10 seconds");
        }
        
        config.sync_queue_size = qos_node["sync_queue_size"].as<int>();
        if (config.sync_queue_size <= 0 || config.sync_queue_size > 1000) {
            throw std::runtime_error("Invalid sync_queue_size: must be between 1 and 1000");
        }
        
        // Load intrinsic and extrinsic parameters for each camera
        // Use filesystem::path for safe path concatenation to prevent path traversal attacks
        std::filesystem::path base_path(config.config_folder);
        std::filesystem::path intrinsic_path = base_path / config.camera_intrinsic_calibration_file;
        std::filesystem::path extrinsic_path = base_path / config.camera_extrinsic_calibration_file;
        
        auto intrinsics = loadIntrinsics(intrinsic_path.string());
        auto extrinsics = loadExtrinsics(extrinsic_path.string());
        
        // Assign calibration data to cameras
        for (auto& cam : config.cameras) {
            if (intrinsics.find(cam.id) != intrinsics.end()) {
                cam.camera_matrix = intrinsics[cam.id].first;
                cam.dist_coeffs = intrinsics[cam.id].second;
            } else {
                RCLCPP_WARN(rclcpp::get_logger("config_loader"), 
                           "No intrinsic calibration found for camera: %s", cam.id.c_str());
            }
            
            if (extrinsics.find(cam.id) != extrinsics.end()) {
                cam.extrinsic_matrix = extrinsics[cam.id];
            } else {
                RCLCPP_WARN(rclcpp::get_logger("config_loader"), 
                           "No extrinsic calibration found for camera: %s", cam.id.c_str());
            }
        }
        
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config file: " + std::string(e.what()));
    }
    
    return config;
}

std::unordered_map<std::string, std::pair<cv::Mat, cv::Mat>> 
ConfigLoader::loadIntrinsics(const std::string& intrinsic_file_path) {
    std::unordered_map<std::string, std::pair<cv::Mat, cv::Mat>> intrinsics;
    
    try {
        YAML::Node yaml = YAML::LoadFile(intrinsic_file_path);
        
        for (YAML::const_iterator it = yaml.begin(); it != yaml.end(); ++it) {
            std::string camera_id = it->first.as<std::string>();
            YAML::Node cam_node = it->second;
            
            cv::Mat camera_matrix = parseCameraMatrix(cam_node["camera_matrix"]);
            cv::Mat dist_coeffs = parseDistortionCoeffs(cam_node["distortion_coefficients"]);
            
            intrinsics[camera_id] = std::make_pair(camera_matrix, dist_coeffs);
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load intrinsic calibration: " + std::string(e.what()));
    }
    
    return intrinsics;
}

std::unordered_map<std::string, Eigen::Matrix4d> 
ConfigLoader::loadExtrinsics(const std::string& extrinsic_file_path) {
    std::unordered_map<std::string, Eigen::Matrix4d> extrinsics;
    
    try {
        YAML::Node yaml = YAML::LoadFile(extrinsic_file_path);
        
        for (YAML::const_iterator it = yaml.begin(); it != yaml.end(); ++it) {
            std::string camera_id = it->first.as<std::string>();
            YAML::Node cam_node = it->second;
            
            Eigen::Matrix4d extrinsic = parseExtrinsicMatrix(cam_node["extrinsic_matrix"]);
            extrinsics[camera_id] = extrinsic;
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load extrinsic calibration: " + std::string(e.what()));
    }
    
    return extrinsics;
}

cv::Mat ConfigLoader::parseCameraMatrix(const YAML::Node& node) {
    cv::Mat camera_matrix(3, 3, CV_64F);
    
    auto data = node["data"];
    // Handle nested array format [[a,b,c], [d,e,f], [g,h,i]]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix.at<double>(i, j) = data[i][j].as<double>();
        }
    }
    
    return camera_matrix;
}

cv::Mat ConfigLoader::parseDistortionCoeffs(const YAML::Node& node) {
    auto data = node["data"];
    int cols = node["columns"].as<int>();
    
    cv::Mat dist_coeffs(1, cols, CV_64F);
    for (int i = 0; i < cols; ++i) {
        dist_coeffs.at<double>(0, i) = data[i].as<double>();
    }
    
    return dist_coeffs;
}

Eigen::Matrix4d ConfigLoader::parseExtrinsicMatrix(const YAML::Node& node) {
    Eigen::Matrix4d matrix;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix(i, j) = node[i][j].as<double>();
        }
    }
    
    return matrix;
}

} // namespace utils
} // namespace calico