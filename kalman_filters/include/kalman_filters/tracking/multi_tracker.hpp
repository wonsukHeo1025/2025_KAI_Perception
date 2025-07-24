#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <Eigen/Core>
#include "ukf_track.hpp"
#include "tracking_types.hpp"

namespace kalman_filters {
namespace tracking {

/**
 * @brief Multi-object tracker using Unscented Kalman Filter
 * 
 * Tracks multiple objects over time using UKF for state estimation.
 * Supports custom association methods for flexibility.
 */
class MultiTracker {
public:
    /**
     * @brief Association function type
     * Takes detections and current tracks, returns association result
     */
    using AssociationFunc = std::function<AssociationResult(
        const std::vector<Detection>&,
        const std::unordered_map<int, std::shared_ptr<UKFTrack>>&,
        double max_distance)>;
    
    /**
     * @brief Constructor
     * @param config Tracking configuration
     */
    explicit MultiTracker(const TrackingConfig& config = TrackingConfig());
    ~MultiTracker() = default;
    
    /**
     * @brief Update tracker with new detections
     * @param detections New object detections
     * @param timestamp Current timestamp
     * @param imu_data Optional IMU data for motion compensation
     */
    void update(const std::vector<Detection>& detections,
                double timestamp,
                const IMUData* imu_data = nullptr);
    
    /**
     * @brief Get current tracked objects
     * @return Vector of tracked objects (only confirmed tracks)
     */
    std::vector<TrackedObject> getTrackedObjects() const;
    
    /**
     * @brief Get all active tracks
     * @return Map of track ID to track object
     */
    const std::unordered_map<int, std::shared_ptr<UKFTrack>>& getTracks() const { 
        return tracks_; 
    }
    
    /**
     * @brief Set custom association function
     * @param func Association function
     */
    void setAssociationFunction(AssociationFunc func) {
        association_func_ = func;
    }
    
    /**
     * @brief Set tracking configuration
     * @param config New configuration
     */
    void setConfig(const TrackingConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const TrackingConfig& getConfig() const { return config_; }
    
    /**
     * @brief Reset tracker, removing all tracks
     */
    void reset();
    
    /**
     * @brief Get number of active tracks
     * @return Number of tracks
     */
    size_t getNumTracks() const { return tracks_.size(); }
    
    /**
     * @brief Default greedy nearest neighbor association
     * @param detections New detections
     * @param tracks Current tracks
     * @param max_distance Maximum association distance
     * @return Association result
     */
    static AssociationResult greedyAssociate(
        const std::vector<Detection>& detections,
        const std::unordered_map<int, std::shared_ptr<UKFTrack>>& tracks,
        double max_distance);
    
    /**
     * @brief Enable Hungarian algorithm for data association
     * Requires dlib library. Falls back to greedy if not available.
     */
    void enableHungarianMatching();
    
private:
    /**
     * @brief Predict step for all tracks
     * @param dt Time since last update
     * @param imu_data Optional IMU data
     */
    void predict(double dt, const IMUData* imu_data);
    
    /**
     * @brief Associate detections with existing tracks
     * @param detections New detections
     * @return Association result
     */
    AssociationResult associate(const std::vector<Detection>& detections);
    
    /**
     * @brief Update matched tracks with measurements
     * @param matches Matched pairs from association
     * @param detections Detection measurements
     */
    void updateTracks(const std::vector<std::pair<int, int>>& matches,
                     const std::vector<Detection>& detections);
    
    /**
     * @brief Create new tracks for unmatched detections
     * @param unmatched_detections Indices of unmatched detections
     * @param detections All detections
     */
    void createNewTracks(const std::vector<int>& unmatched_detections,
                        const std::vector<Detection>& detections);
    
    /**
     * @brief Delete old tracks that haven't been updated
     */
    void deleteLostTracks();
    
    /**
     * @brief Generate unique track ID
     * @return New track ID
     */
    int generateTrackId();
    
private:
    TrackingConfig config_;
    std::unordered_map<int, std::shared_ptr<UKFTrack>> tracks_;
    AssociationFunc association_func_;
    double last_timestamp_;
    int next_track_id_;
};

} // namespace tracking
} // namespace kalman_filters