#ifndef CALICO_TRACKING_TRACK_HPP
#define CALICO_TRACKING_TRACK_HPP

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace calico {
namespace tracking {

/**
 * @brief Single object track using Unscented Kalman Filter
 * 
 * Represents a tracked cone with state estimation and color history
 */
class Track {
public:
    /**
     * @brief Constructor
     * @param track_id Unique track identifier
     * @param initial_x Initial X position
     * @param initial_y Initial Y position
     * @param initial_z Initial Z position (not filtered)
     * @param imu_to_sensor_transform 4x4 transform from IMU to sensor frame
     * @param p_initial_pos Initial position covariance
     * @param p_initial_vel Initial velocity covariance
     * @param r_measurement Measurement noise
     * @param q_pos Position process noise
     * @param q_vel Velocity process noise
     */
    Track(int track_id, double initial_x, double initial_y, double initial_z,
          const Eigen::Matrix4d& imu_to_sensor_transform = Eigen::Matrix4d::Identity(),
          double p_initial_pos = 0.001, double p_initial_vel = 100.0,
          double r_measurement = 0.1, double q_pos = 0.1, double q_vel = 0.1);
    ~Track() = default;
    
    /**
     * @brief Predict next state using motion model with IMU compensation
     * @param dt Time delta since last update
     * @param angular_velocity 3D angular velocity from IMU (rad/s)
     * @param linear_acceleration 3D linear acceleration from IMU (m/s²)
     */
    void predict(double dt, const Eigen::Vector3d& angular_velocity = Eigen::Vector3d::Zero(),
                const Eigen::Vector3d& linear_acceleration = Eigen::Vector3d::Zero());
    
    /**
     * @brief Update state with new measurement
     * @param x Measured X position
     * @param y Measured Y position
     * @param z Measured Z position (passed through)
     * @param color Detected color/class
     */
    void update(double x, double y, double z, const std::string& color);
    
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
     * @brief Get most likely color based on history
     * @return Color string with highest confidence
     */
    std::string getColor() const;
    
    /**
     * @brief Get track ID
     * @return Unique track identifier
     */
    int getId() const { return track_id_; }
    
    /**
     * @brief Get track age (frames since creation)
     * @return Track age
     */
    int getAge() const { return age_; }
    
    /**
     * @brief Get number of successful updates
     * @return Hit count
     */
    int getHitCount() const { return hit_count_; }
    
    /**
     * @brief Get frames since last update
     * @return Time since last hit
     */
    int getTimeSinceUpdate() const { return time_since_update_; }
    
    /**
     * @brief Check if track is confirmed (enough hits)
     * @param min_hits Minimum hits required
     * @return True if confirmed
     */
    bool isConfirmed(int min_hits = 3) const { return hit_count_ >= min_hits; }
    
    /**
     * @brief Increment time since update (called when no match)
     */
    void incrementTimeSinceUpdate() { time_since_update_++; }
    
    /**
     * @brief Set measurement noise (R matrix)
     * @param r_measurement New measurement noise value
     */
    void setMeasurementNoise(double r_measurement);
    
    /**
     * @brief Set process noise (Q matrix)
     * @param q_pos Position process noise
     * @param q_vel Velocity process noise
     */
    void setProcessNoise(double q_pos, double q_vel);
    
private:
    /**
     * @brief Initialize UKF parameters
     * @param x Initial X position
     * @param y Initial Y position
     * @param p_initial_pos Initial position covariance
     * @param p_initial_vel Initial velocity covariance
     * @param r_measurement Measurement noise
     * @param q_pos Position process noise
     * @param q_vel Velocity process noise
     */
    void initializeUKF(double x, double y, double p_initial_pos, double p_initial_vel,
                       double r_measurement, double q_pos, double q_vel);
    
    /**
     * @brief Generate sigma points for UKF
     * @return Matrix of sigma points
     */
    Eigen::MatrixXd generateSigmaPoints();
    
    /**
     * @brief Predict sigma points through motion model with IMU compensation
     * @param sigma_points Input sigma points
     * @param dt Time delta
     * @param omega_imu Angular velocity in IMU frame
     * @param accel_imu Linear acceleration in IMU frame
     * @return Predicted sigma points
     */
    Eigen::MatrixXd predictSigmaPoints(const Eigen::MatrixXd& sigma_points,
                                      double dt, const Eigen::Vector3d& omega_imu,
                                      const Eigen::Vector3d& accel_imu);
    
    /**
     * @brief Compute weights for sigma points
     */
    void computeWeights();
    
    /**
     * @brief Update color history and vote
     * @param color New color observation
     */
    void updateColorHistory(const std::string& color);
    
private:
    // Track identification
    int track_id_;
    int age_;
    int hit_count_;
    int time_since_update_;
    
    // UKF state: [x, y, vx, vy]
    Eigen::Vector4d state_;
    Eigen::Matrix4d covariance_;
    
    // UKF parameters
    static constexpr int STATE_DIM = 4;
    static constexpr int MEAS_DIM = 2;
    double alpha_, beta_, kappa_, lambda_;
    Eigen::VectorXd weights_mean_;
    Eigen::VectorXd weights_cov_;
    
    // Process and measurement noise
    Eigen::Matrix4d Q_;  // Process noise covariance
    Eigen::Matrix2d R_;  // Measurement noise covariance
    
    // Z coordinate (not filtered)
    double z_;
    
    // IMU to sensor transform
    Eigen::Matrix4d T_imu_to_sensor_;
    Eigen::Matrix3d R_imu_to_sensor_;
    Eigen::Vector3d t_imu_to_sensor_;
    
    // Color history for voting
    std::deque<std::string> color_history_;
    std::unordered_map<std::string, int> color_counts_;
    std::string definite_color_;
    static constexpr size_t MAX_COLOR_HISTORY = 20;
    static constexpr int COLOR_CONFIDENCE_THRESHOLD = 3;
};

} // namespace tracking
} // namespace calico

#endif // CALICO_TRACKING_TRACK_HPP