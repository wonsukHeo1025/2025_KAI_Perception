#pragma once

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "tracking_types.hpp"

namespace kalman_filters {
namespace tracking {

/**
 * @brief Single object track using Unscented Kalman Filter
 * 
 * Implements a 2D constant velocity model with UKF state estimation.
 * Compatible with calico's Track implementation for easy migration.
 */
class UKFTrack {
public:
    /**
     * @brief Constructor
     * @param track_id Unique track identifier
     * @param initial_x Initial X position
     * @param initial_y Initial Y position
     * @param initial_z Initial Z position (not filtered)
     * @param config Tracking configuration
     */
    UKFTrack(int track_id, double initial_x, double initial_y, double initial_z,
             const TrackingConfig& config = TrackingConfig());
    ~UKFTrack() = default;
    
    /**
     * @brief Predict next state using motion model
     * @param dt Time delta since last update
     * @param imu_data Optional IMU data for motion compensation
     */
    void predict(double dt, const IMUData* imu_data = nullptr);
    
    /**
     * @brief Update state with new measurement
     * @param detection New detection measurement
     */
    void update(const Detection& detection);
    
    /**
     * @brief Update state with position measurement
     * @param x Measured X position
     * @param y Measured Y position
     * @param z Measured Z position (passed through)
     * @param label Detected label/class
     */
    void update(double x, double y, double z, const std::string& label = "");
    
    /**
     * @brief Get current tracked object state
     * @return Tracked object with position, velocity, and metadata
     */
    TrackedObject getTrackedObject() const;
    
    /**
     * @brief Get current position estimate
     * @return 3D position (x, y, z)
     */
    Eigen::Vector3d getPosition() const;
    
    /**
     * @brief Get current velocity estimate
     * @return 2D velocity (vx, vy)
     */
    Eigen::Vector2d getVelocity() const;
    
    /**
     * @brief Get current state vector
     * @return 4D state [x, y, vx, vy]
     */
    Eigen::Vector4d getState() const { return state_; }
    
    /**
     * @brief Get current covariance matrix
     * @return 4x4 covariance matrix
     */
    Eigen::Matrix4d getCovariance() const { return covariance_; }
    
    /**
     * @brief Get most likely label based on history
     * @return Label string with highest confidence
     */
    std::string getLabel() const;
    
    // Track metadata getters
    int getId() const { return track_id_; }
    int getAge() const { return age_; }
    int getHitCount() const { return hit_count_; }
    int getTimeSinceUpdate() const { return time_since_update_; }
    bool isConfirmed() const { return hit_count_ >= config_.min_hits; }
    
    /**
     * @brief Increment time since update (called when no match)
     */
    void incrementTimeSinceUpdate() { time_since_update_++; }
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void setConfig(const TrackingConfig& config);
    
private:
    /**
     * @brief Initialize UKF parameters
     */
    void initializeUKF();
    
    /**
     * @brief Generate sigma points for UKF
     * @return Matrix of sigma points (4x9)
     */
    Eigen::MatrixXd generateSigmaPoints() const;
    
    /**
     * @brief Predict sigma points through motion model
     * @param sigma_points Input sigma points
     * @param dt Time delta
     * @param imu_data Optional IMU data
     * @return Predicted sigma points
     */
    Eigen::MatrixXd predictSigmaPoints(const Eigen::MatrixXd& sigma_points,
                                      double dt, const IMUData* imu_data) const;
    
    /**
     * @brief Compute weights for sigma points
     */
    void computeWeights();
    
    /**
     * @brief Update label history and vote
     * @param label New label observation
     */
    void updateLabelHistory(const std::string& label);
    
private:
    // Track identification
    int track_id_;
    int age_;
    int hit_count_;
    int time_since_update_;
    
    // Configuration
    TrackingConfig config_;
    
    // UKF state: [x, y, vx, vy]
    Eigen::Vector4d state_;
    Eigen::Matrix4d covariance_;
    
    // UKF parameters
    static constexpr int STATE_DIM = 4;
    static constexpr int MEAS_DIM = 2;
    double lambda_;
    Eigen::VectorXd weights_mean_;
    Eigen::VectorXd weights_cov_;
    
    // Process and measurement noise
    Eigen::Matrix4d Q_;  // Process noise covariance
    Eigen::Matrix2d R_;  // Measurement noise covariance
    
    // Z coordinate (not filtered)
    double z_;
    
    // IMU to sensor transform (extracted from config)
    Eigen::Matrix3d R_imu_to_sensor_;
    Eigen::Vector3d t_imu_to_sensor_;
    
    // Label history for voting
    std::deque<std::string> label_history_;
    std::unordered_map<std::string, int> label_counts_;
    std::string definite_label_;
    static constexpr size_t MAX_LABEL_HISTORY = 20;
    static constexpr int LABEL_CONFIDENCE_THRESHOLD = 3;
};

} // namespace tracking
} // namespace kalman_filters