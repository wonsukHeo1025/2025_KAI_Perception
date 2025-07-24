#include "kalman_filters/tracking/ukf_track.hpp"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace kalman_filters {
namespace tracking {

UKFTrack::UKFTrack(int track_id, double initial_x, double initial_y, double initial_z,
                   const TrackingConfig& config)
    : track_id_(track_id), age_(0), hit_count_(0), time_since_update_(0), 
      z_(initial_z), config_(config), definite_label_("") {
    
    // Extract rotation and translation from transform
    R_imu_to_sensor_ = config_.imu_to_sensor_transform.block<3, 3>(0, 0);
    t_imu_to_sensor_ = config_.imu_to_sensor_transform.block<3, 1>(0, 3);
    
    // Initialize state
    state_ = Eigen::Vector4d(initial_x, initial_y, 0.0, 0.0);
    
    // Initialize UKF
    initializeUKF();
    
    // Initialize label history
    label_counts_.clear();
}

void UKFTrack::initializeUKF() {
    // Initial covariance
    covariance_ = Eigen::Matrix4d::Identity();
    covariance_.block<2, 2>(0, 0) *= config_.p_initial_pos;
    covariance_.block<2, 2>(2, 2) *= config_.p_initial_vel;
    
    // UKF parameters (matching calico/filterpy)
    double kappa = config_.kappa;
    if (kappa < 0) {
        kappa = 3.0 - STATE_DIM;  // Default: 3 - n
    }
    lambda_ = config_.alpha * config_.alpha * (STATE_DIM + kappa) - STATE_DIM;
    
    // Compute weights
    computeWeights();
    
    // Process noise
    Q_ = Eigen::Matrix4d::Identity();
    Q_.block<2, 2>(0, 0) *= config_.q_pos;  // Position process noise
    Q_.block<2, 2>(2, 2) *= config_.q_vel;  // Velocity process noise
    
    // Measurement noise
    R_ = Eigen::Matrix2d::Identity() * config_.r_pos;
}

void UKFTrack::computeWeights() {
    double weight_0 = lambda_ / (STATE_DIM + lambda_);
    weights_mean_ = Eigen::VectorXd(2 * STATE_DIM + 1);
    weights_cov_ = Eigen::VectorXd(2 * STATE_DIM + 1);
    
    weights_mean_(0) = weight_0;
    weights_cov_(0) = weight_0 + (1 - config_.alpha * config_.alpha + config_.beta);
    
    for (int i = 1; i < 2 * STATE_DIM + 1; ++i) {
        weights_mean_(i) = 0.5 / (STATE_DIM + lambda_);
        weights_cov_(i) = weights_mean_(i);
    }
}

void UKFTrack::predict(double dt, const IMUData* imu_data) {
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints();
    
    // Predict sigma points
    Eigen::MatrixXd predicted_sigma = predictSigmaPoints(sigma_points, dt, imu_data);
    
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

Eigen::MatrixXd UKFTrack::generateSigmaPoints() const {
    Eigen::MatrixXd sigma_points(STATE_DIM, 2 * STATE_DIM + 1);
    
    // Compute square root of scaled covariance
    Eigen::Matrix4d scaled_cov = (STATE_DIM + lambda_) * covariance_;
    Eigen::Matrix4d sqrt_cov;
    
    // Try Cholesky decomposition first
    Eigen::LLT<Eigen::Matrix4d> chol(scaled_cov);
    if (chol.info() == Eigen::Success) {
        sqrt_cov = chol.matrixL();
    } else {
        // Fallback to SVD if Cholesky fails
        // Add small regularization
        Eigen::Matrix4d cov_reg = scaled_cov + 1e-6 * Eigen::Matrix4d::Identity();
        
        // Try Cholesky again with regularization
        Eigen::LLT<Eigen::Matrix4d> chol_reg(cov_reg);
        if (chol_reg.info() == Eigen::Success) {
            sqrt_cov = chol_reg.matrixL();
        } else {
            // Use SVD as last resort
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(cov_reg, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector4d s = svd.singularValues();
            
            // Ensure all singular values are positive
            for (int i = 0; i < s.size(); ++i) {
                if (s(i) < 1e-10) s(i) = 1e-10;
            }
            
            // Reconstruct square root matrix
            sqrt_cov = svd.matrixU() * s.cwiseSqrt().asDiagonal();
        }
    }
    
    // First sigma point is the mean
    sigma_points.col(0) = state_;
    
    // Generate other sigma points
    for (int i = 0; i < STATE_DIM; ++i) {
        sigma_points.col(i + 1) = state_ + sqrt_cov.col(i);
        sigma_points.col(i + 1 + STATE_DIM) = state_ - sqrt_cov.col(i);
    }
    
    return sigma_points;
}

Eigen::MatrixXd UKFTrack::predictSigmaPoints(const Eigen::MatrixXd& sigma_points,
                                            double dt, const IMUData* imu_data) const {
    Eigen::MatrixXd predicted(STATE_DIM, 2 * STATE_DIM + 1);
    
    if (config_.enable_imu_compensation && imu_data) {
        // With IMU compensation
        Eigen::Vector3d omega_imu = imu_data->getAngularVelocity();
        Eigen::Vector3d accel_imu = imu_data->getLinearAcceleration();
        
        // Transform IMU measurements to sensor frame
        Eigen::Vector3d omega_sensor = R_imu_to_sensor_ * omega_imu;
        Eigen::Vector3d accel_sensor = R_imu_to_sensor_ * accel_imu;
        
        // Use Z-axis rotation only for 2D tracking
        double omega_z = omega_sensor.z();
        Eigen::Vector2d accel_xy(accel_sensor.x(), accel_sensor.y());
        
        // Compute rotation during dt
        double theta = omega_z * dt;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        Eigen::Matrix2d R_delta;
        R_delta << cos_theta, -sin_theta,
                   sin_theta, cos_theta;
        
        // Inverse rotation for compensation
        Eigen::Matrix2d R_compensation = R_delta.transpose();
        
        for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
            Eigen::Vector2d current_pos(sigma_points(0, i), sigma_points(1, i));
            Eigen::Vector2d current_vel(sigma_points(2, i), sigma_points(3, i));
            
            // Predict velocity
            Eigen::Vector2d predicted_vel = current_vel + accel_xy * dt;
            
            // Predict position with compensation
            Eigen::Vector2d delta_pos = current_vel * dt + 0.5 * accel_xy * dt * dt;
            Eigen::Vector2d pos_before_rotation = current_pos - delta_pos;
            Eigen::Vector2d predicted_pos = R_compensation * pos_before_rotation;
            predicted_vel = R_compensation * predicted_vel;
            
            // Store predicted state
            predicted(0, i) = predicted_pos.x();
            predicted(1, i) = predicted_pos.y();
            predicted(2, i) = predicted_vel.x();
            predicted(3, i) = predicted_vel.y();
        }
    } else {
        // Simple constant velocity model
        for (int i = 0; i < 2 * STATE_DIM + 1; ++i) {
            predicted(0, i) = sigma_points(0, i) + sigma_points(2, i) * dt;  // x + vx*dt
            predicted(1, i) = sigma_points(1, i) + sigma_points(3, i) * dt;  // y + vy*dt
            predicted(2, i) = sigma_points(2, i);  // vx unchanged
            predicted(3, i) = sigma_points(3, i);  // vy unchanged
        }
    }
    
    return predicted;
}

void UKFTrack::update(const Detection& detection) {
    update(detection.x, detection.y, detection.z, detection.label);
}

void UKFTrack::update(double x, double y, double z, const std::string& label) {
    // Update z directly (no filtering)
    z_ = z;
    
    // Measurement
    Eigen::Vector2d measurement(x, y);
    
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints();
    
    // Transform sigma points to measurement space (observe only position)
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
    
    // Kalman gain
    Eigen::Matrix<double, 4, 2> K = Pxz * S.lu().solve(Eigen::Matrix2d::Identity());
    
    // Update state and covariance
    Eigen::Vector2d innovation = measurement - z_pred;
    state_ += K * innovation;
    covariance_ -= K * S * K.transpose();
    
    // Update tracking info
    time_since_update_ = 0;
    hit_count_++;
    
    // Update label history
    if (!label.empty()) {
        updateLabelHistory(label);
    }
}

void UKFTrack::updateLabelHistory(const std::string& label) {
    // Convert to lowercase for consistency
    std::string lowercase_label = label;
    std::transform(lowercase_label.begin(), lowercase_label.end(), 
                   lowercase_label.begin(), ::tolower);
    
    // Add to history
    label_history_.push_back(lowercase_label);
    if (label_history_.size() > MAX_LABEL_HISTORY) {
        std::string old_label = label_history_.front();
        label_history_.pop_front();
        if (label_counts_.find(old_label) != label_counts_.end()) {
            label_counts_[old_label] = std::max(0, label_counts_[old_label] - 1);
        }
    }
    
    label_counts_[lowercase_label]++;
    
    // Check if we should set definite label
    if (definite_label_.empty()) {
        for (const auto& [lbl, count] : label_counts_) {
            if (count >= LABEL_CONFIDENCE_THRESHOLD && lbl != "unknown") {
                definite_label_ = lbl;
                break;
            }
        }
    }
}

TrackedObject UKFTrack::getTrackedObject() const {
    TrackedObject obj;
    obj.track_id = track_id_;
    obj.x = state_(0);
    obj.y = state_(1);
    obj.z = z_;
    obj.vx = state_(2);
    obj.vy = state_(3);
    obj.label = getLabel();
    obj.age = age_;
    obj.hits = hit_count_;
    return obj;
}

Eigen::Vector3d UKFTrack::getPosition() const {
    return Eigen::Vector3d(state_(0), state_(1), z_);
}

Eigen::Vector2d UKFTrack::getVelocity() const {
    return Eigen::Vector2d(state_(2), state_(3));
}

std::string UKFTrack::getLabel() const {
    // If definite label is set, return it
    if (!definite_label_.empty()) {
        // Capitalize for output (e.g., "blue cone" -> "Blue Cone")
        std::string result = definite_label_;
        result[0] = std::toupper(result[0]);
        size_t pos = result.find(' ');
        if (pos != std::string::npos && pos + 1 < result.length()) {
            result[pos + 1] = std::toupper(result[pos + 1]);
        }
        return result;
    }
    
    // Otherwise find best label from counts
    std::string best_label = "unknown";
    int max_count = 0;
    
    for (const auto& [label, count] : label_counts_) {
        if (label != "unknown" && count > max_count) {
            max_count = count;
            best_label = label;
        }
    }
    
    // Capitalize
    if (best_label != "unknown") {
        best_label[0] = std::toupper(best_label[0]);
        size_t pos = best_label.find(' ');
        if (pos != std::string::npos && pos + 1 < best_label.length()) {
            best_label[pos + 1] = std::toupper(best_label[pos + 1]);
        }
        return best_label;
    }
    
    return "Unknown";
}

void UKFTrack::setConfig(const TrackingConfig& config) {
    config_ = config;
    
    // Update IMU transform
    R_imu_to_sensor_ = config_.imu_to_sensor_transform.block<3, 3>(0, 0);
    t_imu_to_sensor_ = config_.imu_to_sensor_transform.block<3, 1>(0, 3);
    
    // Reinitialize UKF parameters
    initializeUKF();
}

} // namespace tracking
} // namespace kalman_filters