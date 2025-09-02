#ifndef CALICO_FUSION_MULTI_CAMERA_FUSION_HPP
#define CALICO_FUSION_MULTI_CAMERA_FUSION_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "calico/utils/config_loader.hpp"
#include "calico/utils/message_converter.hpp"
#include "calico/utils/memory_pool.hpp"
#include "calico/fusion/hungarian_matcher.hpp"

namespace calico {
namespace fusion {

/**
 * @brief Fusion result for a single camera
 */
struct CameraFusionResult {
    std::string camera_id;
    std::vector<int> matched_cone_indices;
    std::vector<std::string> matched_colors;
    std::vector<int> unmatched_cone_indices;
};

/**
 * @brief Multi-camera sensor fusion class
 * 
 * Fuses LiDAR cone detections with YOLO detections from multiple cameras
 */
class MultiCameraFusion {
public:
    MultiCameraFusion();
    ~MultiCameraFusion() = default;
    
    /**
     * @brief Initialize fusion with camera configurations
     * @param camera_configs Vector of camera configurations
     */
    void initialize(const std::vector<utils::CameraConfig>& camera_configs);
    
    /**
     * @brief Perform fusion between LiDAR cones and YOLO detections
     * @param lidar_cones 3D cone positions from LiDAR
     * @param camera_detections Map of camera_id to YOLO detections
     * @return Fused cones with assigned colors/classes
     */
    std::vector<utils::Cone> fuse(
        const std::vector<utils::Cone>& lidar_cones,
        const std::unordered_map<std::string, std::vector<utils::Detection>>& camera_detections);
    
    /**
     * @brief Set maximum matching distance threshold
     * @param distance Maximum pixel distance for valid matches
     */
    void setMaxMatchingDistance(double distance) { max_matching_distance_ = distance; }
    
    /**
     * @brief Get fusion results for individual cameras
     * @return Map of camera_id to fusion results
     */
    const std::unordered_map<std::string, CameraFusionResult>& 
    getCameraResults() const { return camera_results_; }
    
private:
    /**
     * @brief Perform fusion for a single camera
     * @param camera_config Camera configuration
     * @param lidar_cones LiDAR cone positions
     * @param yolo_detections YOLO detections for this camera
     * @return Fusion result for this camera
     */
    CameraFusionResult fuseForCamera(
        const utils::CameraConfig& camera_config,
        const std::vector<utils::Cone>& lidar_cones,
        const std::vector<utils::Detection>& yolo_detections);
    
    /**
     * @brief Merge fusion results from multiple cameras
     * @param lidar_cones Original LiDAR cones
     * @param camera_results Individual camera fusion results
     * @return Final fused cones
     */
    std::vector<utils::Cone> mergeCameraResults(
        const std::vector<utils::Cone>& lidar_cones,
        const std::unordered_map<std::string, CameraFusionResult>& camera_results);
    
    /**
     * @brief Resolve conflicts when multiple cameras match the same cone
     * @param cone_matches Map of cone index to camera matches
     * @return Resolved color assignment for each cone
     */
    std::unordered_map<int, std::string> resolveConflicts(
        const std::unordered_map<int, std::vector<std::pair<std::string, std::string>>>& cone_matches);
    
private:
    std::vector<utils::CameraConfig> camera_configs_;
    std::unique_ptr<HungarianMatcher> matcher_;
    double max_matching_distance_;
    std::unordered_map<std::string, CameraFusionResult> camera_results_;
    Eigen::Matrix4d T_sensor_to_lidar_;  // Transform from os_sensor to os_lidar frame
    
    // Memory pools for reducing allocations
    utils::VectorPool<int> int_pool_;
    utils::VectorPool<std::string> string_pool_;
    utils::VectorPool<std::pair<double, double>> pair_pool_;
};

} // namespace fusion
} // namespace calico

#endif // CALICO_FUSION_MULTI_CAMERA_FUSION_HPP