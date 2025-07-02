#include "calico/tracking/ukf_tracker.hpp"
#include <algorithm>
#include <rclcpp/rclcpp.hpp>

namespace calico {
namespace tracking {

UKFTracker::UKFTracker(const UKFConfig& config)
    : config_(config),
      matcher_(std::make_unique<fusion::HungarianMatcher>()),
      last_timestamp_(0.0),
      next_track_id_(0) {
}

void UKFTracker::update(const std::vector<utils::Cone>& detections,
                       double timestamp,
                       const utils::IMUData* imu_data) {
    // Compute time delta
    double dt = 0.0;
    if (last_timestamp_ > 0) {
        dt = timestamp - last_timestamp_;
    }
    last_timestamp_ = timestamp;
    
    if (dt > 0 && dt < 1.0) {  // Sanity check on dt
        // Predict step
        predict(dt, imu_data);
    }
    
    // Associate detections with tracks
    fusion::MatchResult matches = associate(detections);
    
    // Update matched tracks
    updateTracks(matches.matches, detections);
    
    // Create new tracks for unmatched detections
    createNewTracks(matches.unmatched_detections, detections);
    
    // Delete lost tracks
    deleteLostTracks();
}

std::vector<utils::Cone> UKFTracker::getTrackedCones() const {
    std::vector<utils::Cone> tracked_cones;
    
    for (const auto& [id, track] : tracks_) {
        if (track->isConfirmed(config_.min_hits_before_confirmation)) {
            utils::Cone cone;
            auto pos = track->getPosition();
            cone.x = pos(0);
            cone.y = pos(1);
            cone.z = pos(2);
            cone.color = track->getColor();
            cone.id = track->getId();
            tracked_cones.push_back(cone);
        }
    }
    
    return tracked_cones;
}

void UKFTracker::reset() {
    tracks_.clear();
    next_track_id_ = 0;
    last_timestamp_ = 0.0;
}

void UKFTracker::predict(double dt, const utils::IMUData* imu_data) {
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
    
    if (imu_data) {
        // Use IMU data if available
        angular_velocity << imu_data->angular_vel_x,
                           imu_data->angular_vel_y,
                           imu_data->angular_vel_z;
        
        linear_acceleration << imu_data->linear_accel_x,
                              imu_data->linear_accel_y,
                              imu_data->linear_accel_z;
    }
    
    // Predict all tracks with IMU compensation
    for (auto& [id, track] : tracks_) {
        track->predict(dt, angular_velocity, linear_acceleration);
    }
}

fusion::MatchResult UKFTracker::associate(const std::vector<utils::Cone>& detections) {
    // Extract positions for cost matrix computation
    std::vector<std::pair<double, double>> detection_positions;
    for (const auto& det : detections) {
        detection_positions.emplace_back(det.x, det.y);
    }
    
    std::vector<std::pair<double, double>> track_positions;
    std::vector<int> track_ids;
    for (const auto& [id, track] : tracks_) {
        auto pos = track->getPosition();
        track_positions.emplace_back(pos(0), pos(1));
        track_ids.push_back(id);
    }
    
    if (detection_positions.empty() || track_positions.empty()) {
        // No association possible
        fusion::MatchResult result;
        for (size_t i = 0; i < detections.size(); ++i) {
            result.unmatched_detections.push_back(i);
        }
        return result;
    }
    
    // Compute cost matrix with gating (matching Python logic)
    auto cost_matrix = fusion::HungarianMatcher::computeCostMatrix(
        detection_positions, track_positions);
    
    // Apply threshold gating - set costs above threshold to threshold + 1
    // This matches Python's approach for better track stability
    for (int i = 0; i < cost_matrix.rows(); ++i) {
        for (int j = 0; j < cost_matrix.cols(); ++j) {
            if (cost_matrix(i, j) > config_.max_association_distance) {
                cost_matrix(i, j) = config_.max_association_distance + 1.0;
            }
        }
    }
    
    // Perform matching
    auto match_result = matcher_->match(cost_matrix, config_.max_association_distance);
    
    // Convert track indices to track IDs
    fusion::MatchResult final_result;
    for (const auto& [det_idx, track_idx] : match_result.matches) {
        if (track_idx < track_ids.size()) {
            final_result.matches.emplace_back(det_idx, track_ids[track_idx]);
        }
    }
    final_result.unmatched_detections = match_result.unmatched_detections;
    
    return final_result;
}

void UKFTracker::updateTracks(const std::vector<std::pair<int, int>>& matches,
                             const std::vector<utils::Cone>& detections) {
    for (const auto& [det_idx, track_id] : matches) {
        if (det_idx < detections.size() && tracks_.find(track_id) != tracks_.end()) {
            const auto& det = detections[det_idx];
            tracks_[track_id]->update(det.x, det.y, det.z, det.color);
        }
    }
    
    // Mark unmatched tracks
    std::set<int> matched_track_ids;
    for (const auto& [det_idx, track_id] : matches) {
        matched_track_ids.insert(track_id);
    }
    
    for (auto& [id, track] : tracks_) {
        if (matched_track_ids.find(id) == matched_track_ids.end()) {
            track->incrementTimeSinceUpdate();
        }
    }
}

void UKFTracker::createNewTracks(const std::vector<int>& unmatched_detections,
                                const std::vector<utils::Cone>& detections) {
    for (int det_idx : unmatched_detections) {
        if (det_idx < detections.size()) {
            const auto& det = detections[det_idx];
            int new_id = generateTrackId();
            // Pass IMU transform and UKF parameters to new track
            tracks_[new_id] = std::make_shared<Track>(
                new_id, det.x, det.y, det.z,
                config_.imu_to_sensor_transform,
                config_.p_initial_pos, config_.p_initial_vel,
                config_.r_pos, config_.q_pos, config_.q_vel);
            tracks_[new_id]->update(det.x, det.y, det.z, det.color);
        }
    }
}

void UKFTracker::deleteLostTracks() {
    std::vector<int> tracks_to_delete;
    
    for (const auto& [id, track] : tracks_) {
        if (track->getTimeSinceUpdate() > config_.max_age_before_deletion) {
            tracks_to_delete.push_back(id);
        }
    }
    
    for (int id : tracks_to_delete) {
        tracks_.erase(id);
        RCLCPP_DEBUG(rclcpp::get_logger("ukf_tracker"), 
                    "Deleted track %d (age: %d)", id, config_.max_age_before_deletion);
    }
}

int UKFTracker::generateTrackId() {
    return next_track_id_++;
}

} // namespace tracking
} // namespace calico