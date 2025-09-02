#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>

namespace kalman_filters {
namespace tracking {

/**
 * @brief Detected object structure
 * Generic structure for any trackable object
 */
struct Detection {
    double x = 0.0;           // X position
    double y = 0.0;           // Y position  
    double z = 0.0;           // Z position
    std::string label = "";   // Object class/color/type
    int id = -1;              // Detection ID (optional)
    double confidence = 1.0;  // Detection confidence (optional)
    
    Detection() = default;
    Detection(double x_, double y_, double z_, const std::string& label_ = "")
        : x(x_), y(y_), z(z_), label(label_) {}
};

/**
 * @brief Tracked object structure
 * Includes state estimation and track metadata
 */
struct TrackedObject : public Detection {
    int track_id = -1;        // Unique track ID
    double vx = 0.0;          // X velocity
    double vy = 0.0;          // Y velocity
    int age = 0;              // Track age (frames)
    int hits = 0;             // Number of updates
    
    TrackedObject() = default;
    TrackedObject(const Detection& det) : Detection(det) {}
};

/**
 * @brief IMU data structure for motion compensation
 */
struct IMUData {
    double timestamp = 0.0;
    // Angular velocity (rad/s)
    double angular_vel_x = 0.0;
    double angular_vel_y = 0.0;
    double angular_vel_z = 0.0;
    // Linear acceleration (m/s²)
    double linear_accel_x = 0.0;
    double linear_accel_y = 0.0;
    double linear_accel_z = 0.0;
    
    Eigen::Vector3d getAngularVelocity() const {
        return Eigen::Vector3d(angular_vel_x, angular_vel_y, angular_vel_z);
    }
    
    Eigen::Vector3d getLinearAcceleration() const {
        return Eigen::Vector3d(linear_accel_x, linear_accel_y, linear_accel_z);
    }
};

/**
 * @brief Association result structure
 */
struct AssociationResult {
    std::vector<std::pair<int, int>> matches;  // (detection_idx, track_id)
    std::vector<int> unmatched_detections;     // Indices of unmatched detections
    std::vector<int> unmatched_tracks;         // IDs of unmatched tracks
};

/**
 * @brief Tracking configuration
 */
struct TrackingConfig {
    // UKF parameters
    double alpha = 0.1;       // Spread of sigma points
    double beta = 2.0;        // Prior knowledge of distribution  
    double kappa = -1.0;      // Secondary scaling parameter
    
    // Process noise
    double q_pos = 0.1;       // Position process noise
    double q_vel = 0.1;       // Velocity process noise
    
    // Measurement noise
    double r_pos = 0.1;       // Position measurement noise
    
    // Initial covariance
    double p_initial_pos = 0.001;   // Initial position uncertainty
    double p_initial_vel = 100.0;   // Initial velocity uncertainty
    
    // Track management
    int max_age = 4;                    // Max frames without update
    int min_hits = 3;                   // Min hits for confirmation
    double max_association_dist = 0.7;  // Max distance for matching
    
    // IMU compensation
    bool enable_imu_compensation = false;
    Eigen::Matrix4d imu_to_sensor_transform = Eigen::Matrix4d::Identity();
};

} // namespace tracking
} // namespace kalman_filters