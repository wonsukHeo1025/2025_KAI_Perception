#include "calico/tracking/track.hpp"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Cholesky>
#include <unordered_map>

namespace calico {
namespace tracking {

Track::Track(int track_id, double initial_x, double initial_y, double initial_z,
             const Eigen::Matrix4d& imu_to_sensor_transform,
             double p_initial_pos, double p_initial_vel,
             double r_measurement, double q_pos, double q_vel)
    : track_id_(track_id), age_(0), hit_count_(0), time_since_update_(0), z_(initial_z),
      T_imu_to_sensor_(imu_to_sensor_transform), definite_color_("") {
    
    // Extract rotation and translation from transform
    R_imu_to_sensor_ = T_imu_to_sensor_.block<3, 3>(0, 0);
    t_imu_to_sensor_ = T_imu_to_sensor_.block<3, 1>(0, 3);
    
    // Initialize UKF parameters
    initializeUKF(initial_x, initial_y, p_initial_pos, p_initial_vel,
                  r_measurement, q_pos, q_vel);
    
    // Initialize color history
    color_history_.clear();
    color_counts_.clear();
    color_counts_["unknown"] = 0;
    color_counts_["blue cone"] = 0;
    color_counts_["red cone"] = 0;
    color_counts_["yellow cone"] = 0;
}

void Track::initializeUKF(double x, double y, double p_initial_pos, double p_initial_vel,
                         double r_measurement, double q_pos, double q_vel) {
    // State: [x, y, vx, vy]
    state_ = Eigen::Vector4d(x, y, 0.0, 0.0);
    
    // Initial covariance
    covariance_ = Eigen::Matrix4d::Identity();
    covariance_.block<2, 2>(0, 0) *= p_initial_pos;
    covariance_.block<2, 2>(2, 2) *= p_initial_vel;
    
    // UKF parameters (matching Python defaults)
    alpha_ = 0.1;  // Changed from 0.001 to match Python
    beta_ = 2.0;
    kappa_ = 3.0 - STATE_DIM;  // Matching Python: (3.0 - dim_x)
    lambda_ = alpha_ * alpha_ * (STATE_DIM + kappa_) - STATE_DIM;
    
    // Compute weights
    computeWeights();
    
    // Process noise
    Q_ = Eigen::Matrix4d::Identity();
    Q_.block<2, 2>(0, 0) *= q_pos;  // Position process noise
    Q_.block<2, 2>(2, 2) *= q_vel;  // Velocity process noise
    
    // Measurement noise
    R_ = Eigen::Matrix2d::Identity() * r_measurement;
}

void Track::computeWeights() {
    double weight_0 = lambda_ / (STATE_DIM + lambda_);
    weights_mean_ = Eigen::VectorXd(2 * STATE_DIM + 1);
    weights_cov_ = Eigen::VectorXd(2 * STATE_DIM + 1);
    
    weights_mean_(0) = weight_0;
    weights_cov_(0) = weight_0 + (1 - alpha_ * alpha_ + beta_);
    
    for (int i = 1; i < 2 * STATE_DIM + 1; ++i) {
        weights_mean_(i) = 0.5 / (STATE_DIM + lambda_);
        weights_cov_(i) = weights_mean_(i);
    }
}

void Track::predict(double dt, const Eigen::Vector3d& angular_velocity,
                   const Eigen::Vector3d& linear_acceleration) {
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints();
    
    // Predict sigma points with IMU compensation
    Eigen::MatrixXd predicted_sigma = predictSigmaPoints(sigma_points, dt, 
                                                       angular_velocity, linear_acceleration);
    
    // Compute predicted mean
    state_.setZero();
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        state_ += weights_mean_(i) * predicted_sigma.col(i);
    }
    
    // Compute predicted covariance
    covariance_.setZero();
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        Eigen::Vector4d diff = predicted_sigma.col(i) - state_;
        covariance_ += weights_cov_(i) * (diff * diff.transpose());
    }
    covariance_ += Q_;
    
    // Increment time since update
    time_since_update_++;
    age_++;
}

Eigen::MatrixXd Track::generateSigmaPoints() {
    Eigen::MatrixXd sigma_points(STATE_DIM, 2 * STATE_DIM + 1);
    
    // Compute square root of covariance
    Eigen::Matrix4d sqrt_cov = ((STATE_DIM + lambda_) * covariance_).llt().matrixL();
    
    // First sigma point is the mean
    sigma_points.col(0) = state_;
    
    // Generate other sigma points
    for (int i = 0; i < STATE_DIM; ++i) {
        sigma_points.col(i + 1) = state_ + sqrt_cov.col(i);
        sigma_points.col(i + 1 + STATE_DIM) = state_ - sqrt_cov.col(i);
    }
    
    return sigma_points;
}

Eigen::MatrixXd Track::predictSigmaPoints(const Eigen::MatrixXd& sigma_points,
                                         double dt, const Eigen::Vector3d& omega_imu,
                                         const Eigen::Vector3d& accel_imu) {
    Eigen::MatrixXd predicted(STATE_DIM, 2 * STATE_DIM + 1);
    
    // Transform IMU measurements to sensor frame (XY plane only)
    Eigen::Vector3d omega_sensor = R_imu_to_sensor_ * omega_imu;
    Eigen::Vector3d accel_sensor = R_imu_to_sensor_ * accel_imu;
    
    // Use Z-axis rotation only
    double omega_z = omega_sensor.z();
    Eigen::Vector2d accel_xy(accel_sensor.x(), accel_sensor.y());
    
    // Compute rotation during dt (2D)
    double theta = omega_z * dt;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    Eigen::Matrix2d R_delta;
    R_delta << cos_theta, -sin_theta,
               sin_theta, cos_theta;
    
    // Inverse rotation (k+1 frame to k frame)
    Eigen::Matrix2d R_compensation = R_delta.transpose();
    
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        Eigen::Vector2d current_pos(sigma_points(0, i), sigma_points(1, i));
        Eigen::Vector2d current_vel(sigma_points(2, i), sigma_points(3, i));
        
        // Predict sensor velocity at k+1
        Eigen::Vector2d predicted_vel_sensor = current_vel + accel_xy * dt;
        
        // Predict cone position at k+1
        // When sensor moves, cone appears to move in opposite direction
        Eigen::Vector2d delta_pos_sensor = current_vel * dt + 0.5 * accel_xy * dt * dt;
        Eigen::Vector2d pos_before_rotation = current_pos - delta_pos_sensor;
        
        // Apply rotation compensation
        Eigen::Vector2d predicted_pos_cone = R_compensation * pos_before_rotation;
        predicted_vel_sensor = R_compensation * predicted_vel_sensor;
        
        // Store predicted state
        predicted(0, i) = predicted_pos_cone.x();
        predicted(1, i) = predicted_pos_cone.y();
        predicted(2, i) = predicted_vel_sensor.x();
        predicted(3, i) = predicted_vel_sensor.y();
    }
    
    return predicted;
}

void Track::update(double x, double y, double z, const std::string& color) {
    // Update z directly (no filtering)
    z_ = z;
    
    // Measurement
    Eigen::Vector2d measurement(x, y);
    
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints();
    
    // Transform sigma points to measurement space
    Eigen::MatrixXd sigma_z(MEAS_DIM, 2 * STATE_DIM + 1);
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        sigma_z(0, i) = sigma_points(0, i);  // x
        sigma_z(1, i) = sigma_points(1, i);  // y
    }
    
    // Compute predicted measurement mean
    Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        z_pred += weights_mean_(i) * sigma_z.col(i);
    }
    
    // Compute innovation covariance
    Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        Eigen::Vector2d diff = sigma_z.col(i) - z_pred;
        S += weights_cov_(i) * (diff * diff.transpose());
    }
    S += R_;
    
    // Compute cross covariance
    Eigen::Matrix<double, 4, 2> Pxz = Eigen::Matrix<double, 4, 2>::Zero();
    for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
        Eigen::Vector4d x_diff = sigma_points.col(i) - state_;
        Eigen::Vector2d z_diff = sigma_z.col(i) - z_pred;
        Pxz += weights_cov_(i) * (x_diff * z_diff.transpose());
    }
    
    // Kalman gain (using LU decomposition to avoid inverse)
    Eigen::Matrix<double, 4, 2> K = Pxz * S.lu().solve(Eigen::Matrix2d::Identity());
    
    // Update state and covariance
    Eigen::Vector2d innovation = measurement - z_pred;
    state_ += K * innovation;
    covariance_ -= K * S * K.transpose();
    
    // Update tracking info
    time_since_update_ = 0;
    hit_count_++;
    
    // Update color history
    updateColorHistory(color);
}

void Track::updateColorHistory(const std::string& color) {
    // Convert to lowercase for consistency
    std::string lowercase_color = color;
    std::transform(lowercase_color.begin(), lowercase_color.end(), 
                   lowercase_color.begin(), ::tolower);
    
    // Add to history and update counts
    color_history_.push_back(lowercase_color);
    if (color_history_.size() > MAX_COLOR_HISTORY) {
        std::string old_color = color_history_.front();
        color_history_.pop_front();
        if (color_counts_.find(old_color) != color_counts_.end()) {
            color_counts_[old_color] = std::max(0, color_counts_[old_color] - 1);
        }
    }
    
    color_counts_[lowercase_color]++;
    
    // Check if we should set definite color (matching Python logic)
    if (definite_color_.empty()) {
        // Check blue cone, red cone, yellow cone counts
        for (const auto& cone_color : {"blue cone", "red cone", "yellow cone"}) {
            if (color_counts_[cone_color] >= COLOR_CONFIDENCE_THRESHOLD) {
                definite_color_ = cone_color;
                break;
            }
        }
    }
}

Eigen::Vector3d Track::getPosition() const {
    return Eigen::Vector3d(state_(0), state_(1), z_);
}

Eigen::Vector2d Track::getVelocity() const {
    return Eigen::Vector2d(state_(2), state_(3));
}

std::string Track::getColor() const {
    // If definite color is set, return it (matching Python logic)
    if (!definite_color_.empty()) {
        // Capitalize first letter of each word to match Python output
        std::string result = definite_color_;
        result[0] = std::toupper(result[0]);
        size_t pos = result.find(' ');
        if (pos != std::string::npos && pos + 1 < result.length()) {
            result[pos + 1] = std::toupper(result[pos + 1]);
        }
        return result;
    }
    
    // Otherwise find best color from counts
    std::string best_color = "unknown";
    int max_count = 0;
    
    for (const auto& [color, count] : color_counts_) {
        if (color != "unknown" && count > max_count) {
            max_count = count;
            best_color = color;
        }
    }
    
    // Capitalize to match Python output format
    if (best_color != "unknown") {
        best_color[0] = std::toupper(best_color[0]);
        size_t pos = best_color.find(' ');
        if (pos != std::string::npos && pos + 1 < best_color.length()) {
            best_color[pos + 1] = std::toupper(best_color[pos + 1]);
        }
        return best_color;
    }
    
    return "Unknown";  // Capitalized to match Python
}

void Track::setMeasurementNoise(double r_measurement) {
    R_ = Eigen::Matrix2d::Identity() * r_measurement;
}

void Track::setProcessNoise(double q_pos, double q_vel) {
    Q_ = Eigen::Matrix4d::Identity();
    Q_.block<2, 2>(0, 0) *= q_pos;  // Position process noise
    Q_.block<2, 2>(2, 2) *= q_vel;  // Velocity process noise
}

} // namespace tracking
} // namespace calico