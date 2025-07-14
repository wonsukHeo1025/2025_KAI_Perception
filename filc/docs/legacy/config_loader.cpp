#include "filc/config_loader.hpp"
#include <filesystem>
#include <iostream>

namespace filc {

ConfigLoader::ConfigLoader() {
}

ConfigLoader::~ConfigLoader() {
}

bool ConfigLoader::loadConfiguration(const std::string& config_file_path) {
    try {
        // Load main configuration file
        YAML::Node config = YAML::LoadFile(config_file_path);
        
        // Load LiDAR configuration
        if (config["lidar"]) {
            lidar_config_.lidar_topic = config["lidar"]["lidar_topic"].as<std::string>();
            lidar_config_.frame_id = config["lidar"]["frame_id"].as<std::string>();
        } else {
            std::cerr << "Error: LiDAR configuration not found in " << config_file_path << std::endl;
            return false;
        }
        
        // Load general configuration
        if (config["general"]) {
            general_config_.config_folder = config["general"]["config_folder"].as<std::string>();
            general_config_.data_folder = config["general"]["data_folder"].as<std::string>();
            general_config_.camera_intrinsic_calibration = config["general"]["camera_intrinsic_calibration"].as<std::string>();
            general_config_.camera_extrinsic_calibration = config["general"]["camera_extrinsic_calibration"].as<std::string>();
            general_config_.slop = config["general"]["slop"].as<double>();
        } else {
            std::cerr << "Error: General configuration not found in " << config_file_path << std::endl;
            return false;
        }
        
        // Load camera configurations
        if (config["cameras"]) {
            for (const auto& camera_node : config["cameras"]) {
                std::string camera_id = camera_node.first.as<std::string>();
                CameraConfig camera_config;
                
                camera_config.image_topic = camera_node.second["image_topic"].as<std::string>();
                camera_config.projected_topic = camera_node.second["projected_topic"].as<std::string>();
                
                if (camera_node.second["frame_id"]) {
                    camera_config.frame_id = camera_node.second["frame_id"].as<std::string>();
                }
                
                if (camera_node.second["image_size"]) {
                    camera_config.image_size.width = camera_node.second["image_size"]["width"].as<int>();
                    camera_config.image_size.height = camera_node.second["image_size"]["height"].as<int>();
                }
                
                camera_configs_[camera_id] = camera_config;
            }
        } else {
            std::cerr << "Error: Camera configurations not found in " << config_file_path << std::endl;
            return false;
        }
        
        // Load camera intrinsic and extrinsic parameters
        std::string intrinsic_path = general_config_.config_folder + "/" + general_config_.camera_intrinsic_calibration;
        std::string extrinsic_path = general_config_.config_folder + "/" + general_config_.camera_extrinsic_calibration;
        
        if (!loadCameraIntrinsics(intrinsic_path)) {
            std::cerr << "Failed to load camera intrinsics from " << intrinsic_path << std::endl;
            return false;
        }
        
        if (!loadCameraExtrinsics(extrinsic_path)) {
            std::cerr << "Failed to load camera extrinsics from " << extrinsic_path << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigLoader::loadCameraIntrinsics(const std::string& intrinsic_file_path) {
    try {
        YAML::Node intrinsic_config = YAML::LoadFile(intrinsic_file_path);
        
        for (auto& [camera_id, camera_config] : camera_configs_) {
            if (intrinsic_config[camera_id]) {
                const auto& cam_node = intrinsic_config[camera_id];
                
                // Load camera matrix
                if (cam_node["camera_matrix"]) {
                    camera_config.camera_matrix = yamlToMat(cam_node["camera_matrix"]);
                }
                
                // Load distortion coefficients
                if (cam_node["distortion_coefficients"]) {
                    camera_config.distortion_coefficients = yamlToMat(cam_node["distortion_coefficients"]);
                }
                
                // Update image size if available
                if (cam_node["image_size"]) {
                    camera_config.image_size.width = cam_node["image_size"]["width"].as<int>();
                    camera_config.image_size.height = cam_node["image_size"]["height"].as<int>();
                }
            } else {
                std::cerr << "Warning: Intrinsic parameters for camera " << camera_id << " not found" << std::endl;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading camera intrinsics: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigLoader::loadCameraExtrinsics(const std::string& extrinsic_file_path) {
    try {
        YAML::Node extrinsic_config = YAML::LoadFile(extrinsic_file_path);
        
        for (auto& [camera_id, camera_config] : camera_configs_) {
            if (extrinsic_config[camera_id] && extrinsic_config[camera_id]["extrinsic_matrix"]) {
                camera_config.extrinsic_matrix = yamlToEigen4d(extrinsic_config[camera_id]["extrinsic_matrix"]);
            } else {
                std::cerr << "Warning: Extrinsic parameters for camera " << camera_id << " not found" << std::endl;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading camera extrinsics: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat ConfigLoader::yamlToMat(const YAML::Node& node) {
    int rows = node["rows"].as<int>();
    int cols = node["columns"].as<int>();
    
    cv::Mat matrix(rows, cols, CV_64F);
    
    const auto& data = node["data"];
    if (data.IsSequence() && data.size() == static_cast<size_t>(rows)) {
        // If data is structured as rows
        for (int i = 0; i < rows; ++i) {
            const auto& row = data[i];
            if (row.IsSequence() && row.size() == static_cast<size_t>(cols)) {
                for (int j = 0; j < cols; ++j) {
                    matrix.at<double>(i, j) = row[j].as<double>();
                }
            }
        }
    } else if (data.IsSequence() && data.size() == static_cast<size_t>(rows * cols)) {
        // If data is flattened
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix.at<double>(i, j) = data[i * cols + j].as<double>();
            }
        }
    } else {
        std::cerr << "Error: Invalid data structure in YAML matrix" << std::endl;
    }
    
    return matrix;
}

Eigen::Matrix4d ConfigLoader::yamlToEigen4d(const YAML::Node& node) {
    Eigen::Matrix4d matrix;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix(i, j) = node[i][j].as<double>();
        }
    }
    
    return matrix;
}

} // namespace filc 