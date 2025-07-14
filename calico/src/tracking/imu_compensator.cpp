#include "calico/tracking/imu_compensator.hpp"
#include <cmath>
#include <rclcpp/rclcpp.hpp>

namespace calico {
namespace tracking {

IMUCompensator::IMUCompensator(const IMUCompensatorConfig& config)
    : config_(config),
      filtered_acceleration_(Eigen::Vector3d::Zero()),
      angular_velocity_(Eigen::Vector3d::Zero()),
      orientation_(Eigen::Quaterniond::Identity()),
      ema_state_(Eigen::Vector3d::Zero()),
      ema_initialized_(false),
      last_timestamp_(0.0),
      first_measurement_(true) {
    
    if (config_.filter_type == IMUFilterType::BUTTERWORTH) {
        initializeButterworthCoefficients();
    }
}

void IMUCompensator::processIMU(const utils::IMUData& imu_data) {
    // Extract raw data
    Eigen::Vector3d raw_accel(imu_data.linear_accel_x, 
                             imu_data.linear_accel_y, 
                             imu_data.linear_accel_z);
    
    angular_velocity_ = Eigen::Vector3d(imu_data.angular_vel_x,
                                       imu_data.angular_vel_y,
                                       imu_data.angular_vel_z);
    
    // Update orientation
    orientation_ = Eigen::Quaterniond(imu_data.orientation.w,
                                     imu_data.orientation.x,
                                     imu_data.orientation.y,
                                     imu_data.orientation.z);
    
    // Remove gravity if enabled
    if (config_.remove_gravity) {
        raw_accel = removeGravity(raw_accel, orientation_);
    }
    
    // Apply filter based on type
    switch (config_.filter_type) {
        case IMUFilterType::EMA:
            filtered_acceleration_ = applyEMAFilter(raw_accel);
            break;
        case IMUFilterType::BUTTERWORTH:
            filtered_acceleration_ = applyButterworthFilter(raw_accel);
            break;
        default:
            filtered_acceleration_ = raw_accel;
            break;
    }
    
    // Transform to world frame
    filtered_acceleration_ = transformToWorld(filtered_acceleration_, orientation_);
    
    last_timestamp_ = imu_data.timestamp;
    first_measurement_ = false;
}

Eigen::Vector3d IMUCompensator::getCompensatedAcceleration() const {
    return filtered_acceleration_;
}

Eigen::Vector3d IMUCompensator::getAngularVelocity() const {
    return angular_velocity_;
}

Eigen::Quaterniond IMUCompensator::getOrientation() const {
    return orientation_;
}

void IMUCompensator::reset() {
    filtered_acceleration_.setZero();
    angular_velocity_.setZero();
    orientation_ = Eigen::Quaterniond::Identity();
    ema_state_.setZero();
    ema_initialized_ = false;
    input_history_.clear();
    output_history_.clear();
    last_timestamp_ = 0.0;
    first_measurement_ = true;
}

void IMUCompensator::setConfig(const IMUCompensatorConfig& config) {
    config_ = config;
    if (config_.filter_type == IMUFilterType::BUTTERWORTH) {
        initializeButterworthCoefficients();
    }
    reset();
}

Eigen::Vector3d IMUCompensator::applyEMAFilter(const Eigen::Vector3d& new_accel) {
    if (!ema_initialized_) {
        ema_state_ = new_accel;
        ema_initialized_ = true;
        return new_accel;
    }
    
    // EMA: filtered = alpha * new + (1 - alpha) * old
    ema_state_ = config_.ema_alpha * new_accel + (1.0 - config_.ema_alpha) * ema_state_;
    return ema_state_;
}

Eigen::Vector3d IMUCompensator::applyButterworthFilter(const Eigen::Vector3d& new_accel) {
    // Add new input to history
    input_history_.push_back(new_accel);
    if (input_history_.size() > butterworth_a_.size()) {
        input_history_.pop_front();
    }
    
    // Initialize output history if needed
    if (output_history_.empty()) {
        output_history_.push_back(new_accel);
        return new_accel;
    }
    
    // Apply filter
    Eigen::Vector3d output = Eigen::Vector3d::Zero();
    
    // Feed-forward
    for (size_t i = 0; i < input_history_.size() && i < butterworth_b_.size(); ++i) {
        output += butterworth_b_[i] * input_history_[input_history_.size() - 1 - i];
    }
    
    // Feed-back
    for (size_t i = 1; i < output_history_.size() + 1 && i < butterworth_a_.size(); ++i) {
        output -= butterworth_a_[i] * output_history_[output_history_.size() - i];
    }
    
    output /= butterworth_a_[0];
    
    // Update output history
    output_history_.push_back(output);
    if (output_history_.size() > butterworth_a_.size() - 1) {
        output_history_.pop_front();
    }
    
    return output;
}

Eigen::Vector3d IMUCompensator::removeGravity(const Eigen::Vector3d& accel,
                                             const Eigen::Quaterniond& orientation) {
    // Gravity in world frame (pointing down)
    Eigen::Vector3d gravity_world(0.0, 0.0, config_.gravity);
    
    // Transform gravity to body frame
    Eigen::Vector3d gravity_body = orientation.inverse() * gravity_world;
    
    // Remove gravity from acceleration
    return accel - gravity_body;
}

Eigen::Vector3d IMUCompensator::transformToWorld(const Eigen::Vector3d& accel_body,
                                                const Eigen::Quaterniond& orientation) {
    return orientation * accel_body;
}

void IMUCompensator::initializeButterworthCoefficients() {
    // Simple 2nd order Butterworth filter coefficients
    // This is a simplified implementation - for production use scipy.signal.butter equivalent
    
    double fs = 100.0;  // Assumed sampling frequency
    double nyquist = fs / 2.0;
    double normalized_cutoff = config_.butterworth_cutoff / nyquist;
    
    // Pre-computed coefficients for 2nd order Butterworth
    // These would normally be computed using filter design functions
    double a = std::tan(M_PI * normalized_cutoff);
    double a2 = a * a;
    double sqrt2 = std::sqrt(2.0);
    
    butterworth_b_.clear();
    butterworth_a_.clear();
    
    // Normalized coefficients
    butterworth_b_.push_back(a2 / (1 + sqrt2 * a + a2));
    butterworth_b_.push_back(2 * butterworth_b_[0]);
    butterworth_b_.push_back(butterworth_b_[0]);
    
    butterworth_a_.push_back(1.0);
    butterworth_a_.push_back(-2 * (a2 - 1) / (1 + sqrt2 * a + a2));
    butterworth_a_.push_back((1 - sqrt2 * a + a2) / (1 + sqrt2 * a + a2));
}

} // namespace tracking
} // namespace calico