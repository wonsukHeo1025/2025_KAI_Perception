#include "calico/fusion/multi_camera_fusion.hpp"
#include "calico/utils/projection_utils.hpp"
#include <algorithm>
#include <rclcpp/rclcpp.hpp>

namespace calico {
namespace fusion {

MultiCameraFusion::MultiCameraFusion() 
    : matcher_(std::make_unique<HungarianMatcher>()),
      max_matching_distance_(50.0),
      int_pool_(20, 100),  // Pre-allocate 20 vectors with capacity 100
      string_pool_(10, 100),  // Pre-allocate 10 vectors with capacity 100
      pair_pool_(10, 500) {  // Pre-allocate 10 vectors with capacity 500 for point pairs
    
    // Initialize T_sensor_to_lidar from Ouster OS1 documentation
    // This transforms from os_sensor frame to os_lidar frame
    T_sensor_to_lidar_ << -1,  0,  0,  0,
                           0, -1,  0,  0,
                           0,  0,  1, -0.038195,
                           0,  0,  0,  1;
}

void MultiCameraFusion::initialize(const std::vector<utils::CameraConfig>& camera_configs) {
    camera_configs_ = camera_configs;
    camera_results_.clear();
}

std::vector<utils::Cone> MultiCameraFusion::fuse(
    const std::vector<utils::Cone>& lidar_cones,
    const std::unordered_map<std::string, std::vector<utils::Detection>>& camera_detections) {
    
    // Clear previous results
    camera_results_.clear();
    
    // Process each camera independently
    for (const auto& cam_config : camera_configs_) {
        auto det_it = camera_detections.find(cam_config.id);
        if (det_it != camera_detections.end()) {
            CameraFusionResult result = fuseForCamera(cam_config, lidar_cones, det_it->second);
            camera_results_[cam_config.id] = result;
        }
    }
    
    // Merge results from all cameras
    return mergeCameraResults(lidar_cones, camera_results_);
}

CameraFusionResult MultiCameraFusion::fuseForCamera(
    const utils::CameraConfig& camera_config,
    const std::vector<utils::Cone>& lidar_cones,
    const std::vector<utils::Detection>& yolo_detections) {
    
    CameraFusionResult result;
    result.camera_id = camera_config.id;
    
    if (lidar_cones.empty() || yolo_detections.empty()) {
        // No matching possible
        result.unmatched_cone_indices.reserve(lidar_cones.size());
        for (size_t i = 0; i < lidar_cones.size(); ++i) {
            result.unmatched_cone_indices.push_back(i);
        }
        return result;
    }
    
    // Convert LiDAR cones to 3D points
    std::vector<utils::Point3D> lidar_points;
    lidar_points.reserve(lidar_cones.size());
    for (const auto& cone : lidar_cones) {
        lidar_points.emplace_back(cone.x, cone.y, cone.z);
    }
    
    // Compute T_sensor_to_cam = T_lidar_to_cam * T_sensor_to_lidar
    // Since cones are in os_sensor frame, we need the full transformation
    Eigen::Matrix4d T_sensor_to_cam = camera_config.extrinsic_matrix * T_sensor_to_lidar_;
    
    // Project LiDAR points to camera image plane
    // This now returns both projected points and their original indices
    auto [projected_points, original_indices] = utils::ProjectionUtils::projectLidarToCamera(
        lidar_points,
        camera_config.camera_matrix,
        camera_config.dist_coeffs,
        T_sensor_to_cam
    );
    
    // Note: Python implementation doesn't filter by image bounds, only by Z > 0
    // The ProjectionUtils::projectLidarToCamera already filters Z > 0 and returns matching indices
    
    if (projected_points.empty()) {
        // No points project into image
        RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                    "Camera %s: No LiDAR points project into image (out of %zu points)",
                    camera_config.id.c_str(), lidar_points.size());
        for (size_t i = 0; i < lidar_cones.size(); ++i) {
            result.unmatched_cone_indices.push_back(i);
        }
        return result;
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                "Camera %s: %zu/%zu LiDAR points project into image, %zu YOLO detections",
                camera_config.id.c_str(), projected_points.size(), lidar_points.size(), yolo_detections.size());
    
    // Convert projected points and YOLO detections to pairs for matching
    // Use memory pool to reduce allocations
    auto proj_pairs_ptr = pair_pool_.acquire();
    proj_pairs_ptr->reserve(projected_points.size());
    for (const auto& pt : projected_points) {
        proj_pairs_ptr->emplace_back(pt.x, pt.y);
    }
    
    auto yolo_pairs_ptr = pair_pool_.acquire();
    yolo_pairs_ptr->reserve(yolo_detections.size());
    for (const auto& det : yolo_detections) {
        yolo_pairs_ptr->push_back(utils::MessageConverter::getDetectionCenter(det));
    }
    
    // Compute cost matrix (YOLO detections as rows, projected points as columns)
    auto cost_matrix = HungarianMatcher::computeCostMatrix(*yolo_pairs_ptr, *proj_pairs_ptr);
    
    // Perform Hungarian matching
    auto match_result = matcher_->match(cost_matrix, max_matching_distance_);
    
    // Return pools
    pair_pool_.release(proj_pairs_ptr);
    pair_pool_.release(yolo_pairs_ptr);
    
    // Process matches
    result.matched_cone_indices.resize(lidar_cones.size(), -1);
    result.matched_colors.resize(lidar_cones.size(), "Unknown");
    
    RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                "Camera %s: Hungarian matching found %zu matches",
                camera_config.id.c_str(), match_result.matches.size());
    
    for (const auto& match : match_result.matches) {
        int yolo_idx = match.first;
        int proj_idx = match.second;
        
        if (proj_idx < original_indices.size()) {
            int orig_cone_idx = original_indices[proj_idx];
            result.matched_cone_indices[orig_cone_idx] = yolo_idx;
            
            std::string yolo_class = yolo_detections[yolo_idx].class_name;
            std::string mapped_color = utils::MessageConverter::mapClassToColor(yolo_class);
            result.matched_colors[orig_cone_idx] = mapped_color;
            
            RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                        "Camera %s: Matched cone %d with YOLO %d (class='%s' -> color='%s')",
                        camera_config.id.c_str(), orig_cone_idx, yolo_idx, 
                        yolo_class.c_str(), mapped_color.c_str());
        }
    }
    
    // Mark unmatched cones
    result.unmatched_cone_indices.reserve(lidar_cones.size());  // Pre-allocate worst case
    for (size_t i = 0; i < lidar_cones.size(); ++i) {
        if (result.matched_cone_indices[i] == -1) {
            result.unmatched_cone_indices.push_back(i);
        }
    }
    
    return result;
}

std::vector<utils::Cone> MultiCameraFusion::mergeCameraResults(
    const std::vector<utils::Cone>& lidar_cones,
    const std::unordered_map<std::string, CameraFusionResult>& camera_results) {
    
    // Create output cones as copy of input
    std::vector<utils::Cone> fused_cones = lidar_cones;
    
    // Initialize all colors to "Unknown" (matching Python behavior)
    for (auto& cone : fused_cones) {
        cone.color = "Unknown";
    }
    
    // Collect all matches for each cone
    std::unordered_map<int, std::vector<std::pair<std::string, std::string>>> cone_matches;
    
    for (const auto& [cam_id, result] : camera_results) {
        for (size_t i = 0; i < result.matched_cone_indices.size(); ++i) {
            if (result.matched_cone_indices[i] >= 0 && i < result.matched_colors.size()) {
                cone_matches[i].emplace_back(cam_id, result.matched_colors[i]);
                RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                            "Camera %s matched cone %zu with color %s",
                            cam_id.c_str(), i, result.matched_colors[i].c_str());
            }
        }
    }
    
    // Resolve conflicts and assign colors
    auto resolved_colors = resolveConflicts(cone_matches);
    
    RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                "Resolved %zu cone colors from %zu camera results",
                resolved_colors.size(), camera_results.size());
    
    for (const auto& [cone_idx, color] : resolved_colors) {
        if (cone_idx < fused_cones.size()) {
            fused_cones[cone_idx].color = color;
            RCLCPP_DEBUG(rclcpp::get_logger("multi_camera_fusion"),
                        "Assigned color '%s' to cone %d",
                        color.c_str(), cone_idx);
        }
    }
    
    return fused_cones;
}

std::unordered_map<int, std::string> MultiCameraFusion::resolveConflicts(
    const std::unordered_map<int, std::vector<std::pair<std::string, std::string>>>& cone_matches) {
    
    std::unordered_map<int, std::string> resolved;
    
    for (const auto& [cone_idx, matches] : cone_matches) {
        if (matches.empty()) {
            continue;
        }
        
        if (matches.size() == 1) {
            // No conflict
            resolved[cone_idx] = matches[0].second;
        } else {
            // Multiple cameras detected this cone - use voting
            std::unordered_map<std::string, int> color_votes;
            for (const auto& [cam_id, color] : matches) {
                if (color != "Unknown") {
                    color_votes[color]++;
                }
            }
            
            // Find color with most votes
            std::string best_color = "unknown";  // Start with lowercase
            int max_votes = 0;
            for (const auto& [color, votes] : color_votes) {
                if (votes > max_votes) {
                    max_votes = votes;
                    best_color = color;
                }
            }
            
            // If we didn't find any color, use "Unknown" (capitalized) to match Python
            if (max_votes == 0) {
                best_color = "Unknown";
            }
            
            resolved[cone_idx] = best_color;
        }
    }
    
    return resolved;
}

} // namespace fusion
} // namespace calico