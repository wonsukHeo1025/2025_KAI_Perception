#include "kalman_filters/tracking/multi_tracker.hpp"
#include "kalman_filters/tracking/hungarian_matcher.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace kalman_filters {
namespace tracking {

MultiTracker::MultiTracker(const TrackingConfig& config)
    : config_(config), next_track_id_(0), last_timestamp_(0.0) {
    // Set default association function
    association_func_ = greedyAssociate;
}

void MultiTracker::update(const std::vector<Detection>& detections,
                         double timestamp,
                         const IMUData* imu_data) {
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
    AssociationResult result = associate(detections);
    
    // Update matched tracks
    updateTracks(result.matches, detections);
    
    // Create new tracks for unmatched detections
    createNewTracks(result.unmatched_detections, detections);
    
    // Delete lost tracks
    deleteLostTracks();
}

std::vector<TrackedObject> MultiTracker::getTrackedObjects() const {
    std::vector<TrackedObject> tracked_objects;
    
    for (const auto& [id, track] : tracks_) {
        if (track->isConfirmed()) {
            tracked_objects.push_back(track->getTrackedObject());
        }
    }
    
    return tracked_objects;
}

void MultiTracker::setConfig(const TrackingConfig& config) {
    config_ = config;
    
    // Update config for all existing tracks
    for (auto& [id, track] : tracks_) {
        track->setConfig(config);
    }
}

void MultiTracker::reset() {
    tracks_.clear();
    next_track_id_ = 0;
    last_timestamp_ = 0.0;
}

void MultiTracker::predict(double dt, const IMUData* imu_data) {
    for (auto& [id, track] : tracks_) {
        track->predict(dt, imu_data);
    }
}

AssociationResult MultiTracker::associate(const std::vector<Detection>& detections) {
    if (association_func_) {
        return association_func_(detections, tracks_, config_.max_association_dist);
    }
    
    // Fallback to greedy association
    return greedyAssociate(detections, tracks_, config_.max_association_dist);
}

void MultiTracker::updateTracks(const std::vector<std::pair<int, int>>& matches,
                               const std::vector<Detection>& detections) {
    for (const auto& [det_idx, track_id] : matches) {
        if (det_idx < detections.size() && tracks_.find(track_id) != tracks_.end()) {
            tracks_[track_id]->update(detections[det_idx]);
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

void MultiTracker::createNewTracks(const std::vector<int>& unmatched_detections,
                                  const std::vector<Detection>& detections) {
    for (int det_idx : unmatched_detections) {
        if (det_idx < detections.size()) {
            const auto& det = detections[det_idx];
            int new_id = generateTrackId();
            
            tracks_[new_id] = std::make_shared<UKFTrack>(
                new_id, det.x, det.y, det.z, config_);
            
            // Update immediately with the detection
            tracks_[new_id]->update(det);
        }
    }
}

void MultiTracker::deleteLostTracks() {
    std::vector<int> tracks_to_delete;
    
    for (const auto& [id, track] : tracks_) {
        if (track->getTimeSinceUpdate() > config_.max_age) {
            tracks_to_delete.push_back(id);
        }
    }
    
    for (int id : tracks_to_delete) {
        tracks_.erase(id);
    }
}

int MultiTracker::generateTrackId() {
    return next_track_id_++;
}

// Static greedy association method
AssociationResult MultiTracker::greedyAssociate(
    const std::vector<Detection>& detections,
    const std::unordered_map<int, std::shared_ptr<UKFTrack>>& tracks,
    double max_distance) {
    
    AssociationResult result;
    
    if (detections.empty() || tracks.empty()) {
        // All detections are unmatched
        for (size_t i = 0; i < detections.size(); ++i) {
            result.unmatched_detections.push_back(i);
        }
        // All tracks are unmatched
        for (const auto& [id, track] : tracks) {
            result.unmatched_tracks.push_back(id);
        }
        return result;
    }
    
    // Greedy nearest neighbor matching
    std::set<int> used_detections;
    std::set<int> used_tracks;
    
    // For each track, find the nearest detection
    for (const auto& [track_id, track] : tracks) {
        double min_dist = std::numeric_limits<double>::max();
        int best_det_idx = -1;
        
        Eigen::Vector3d track_pos = track->getPosition();
        
        for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
            if (used_detections.find(det_idx) != used_detections.end()) {
                continue;
            }
            
            const auto& det = detections[det_idx];
            double dx = det.x - track_pos(0);
            double dy = det.y - track_pos(1);
            double dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist < min_dist && dist < max_distance) {
                min_dist = dist;
                best_det_idx = det_idx;
            }
        }
        
        if (best_det_idx >= 0) {
            result.matches.emplace_back(best_det_idx, track_id);
            used_detections.insert(best_det_idx);
            used_tracks.insert(track_id);
        }
    }
    
    // Find unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (used_detections.find(i) == used_detections.end()) {
            result.unmatched_detections.push_back(i);
        }
    }
    
    // Find unmatched tracks
    for (const auto& [id, track] : tracks) {
        if (used_tracks.find(id) == used_tracks.end()) {
            result.unmatched_tracks.push_back(id);
        }
    }
    
    return result;
}

void MultiTracker::enableHungarianMatching() {
    // Set association function to use Hungarian algorithm
    association_func_ = HungarianMatcher::createAssociationFunction();
}

} // namespace tracking
} // namespace kalman_filters