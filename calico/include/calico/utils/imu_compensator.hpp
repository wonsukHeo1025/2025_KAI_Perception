#ifndef CALICO_UTILS_IMU_COMPENSATOR_HPP
#define CALICO_UTILS_IMU_COMPENSATOR_HPP

#include <deque>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "calico/utils/message_converter.hpp"

namespace calico {
namespace utils {

/**
 * @brief IMU filter type enumeration
 */
enum class IMUFilterType {
    NONE,
    EMA,        // Exponential Moving Average
    BUTTERWORTH // Butterworth low-pass filter
};

/**
 * @brief IMU compensator configuration
 */
struct IMUCompensatorConfig {
    IMUFilterType filter_type = IMUFilterType::EMA;
    double ema_alpha = 0.1;           // EMA smoothing factor
    double butterworth_cutoff = 5.0;  // Butterworth cutoff frequency (Hz)
    int butterworth_order = 2;        // Butterworth filter order
    double gravity = 9.81;            // Gravity constant
    bool remove_gravity = true;       // Remove gravity from acceleration
};

/**
 * @brief IMU motion compensator for tracking
 * 
 * Processes IMU data to provide motion compensation for tracking
 */
class IMUCompensator {
public:
    IMUCompensator(const IMUCompensatorConfig& config = IMUCompensatorConfig());
    ~IMUCompensator() = default;
    
    /**
     * @brief Process new IMU data
     * @param imu_data Raw IMU data
     */
    void processIMU(const utils::IMUData& imu_data);
    
    /**
     * @brief Get compensated acceleration in world frame
     * @return 3D acceleration vector (x, y, z)
     */
    Eigen::Vector3d getCompensatedAcceleration() const;
    
    /**
     * @brief Get angular velocity
     * @return 3D angular velocity vector (roll, pitch, yaw rates)
     */
    Eigen::Vector3d getAngularVelocity() const;
    
    /**
     * @brief Get current orientation estimate
     * @return Quaternion representing orientation
     */
    Eigen::Quaterniond getOrientation() const;
    
    /**
     * @brief Reset filter states
     */
    void reset();
    
    /**
     * @brief Set configuration
     * @param config New configuration
     */
    void setConfig(const IMUCompensatorConfig& config);
    
private:
    /**
     * @brief Apply EMA filter to acceleration
     * @param new_accel New acceleration measurement
     * @return Filtered acceleration
     */
    Eigen::Vector3d applyEMAFilter(const Eigen::Vector3d& new_accel);
    
    /**
     * @brief Apply Butterworth filter to acceleration
     * @param new_accel New acceleration measurement
     * @return Filtered acceleration
     */
    Eigen::Vector3d applyButterworthFilter(const Eigen::Vector3d& new_accel);
    
    /**
     * @brief Remove gravity component from acceleration
     * @param accel Acceleration in sensor frame
     * @param orientation Current orientation
     * @return Acceleration without gravity
     */
    Eigen::Vector3d removeGravity(const Eigen::Vector3d& accel,
                                 const Eigen::Quaterniond& orientation);
    
    /**
     * @brief Transform acceleration to world frame
     * @param accel_body Acceleration in body frame
     * @param orientation Body to world orientation
     * @return Acceleration in world frame
     */
    Eigen::Vector3d transformToWorld(const Eigen::Vector3d& accel_body,
                                    const Eigen::Quaterniond& orientation);
    
    /**
     * @brief Initialize Butterworth filter coefficients
     */
    void initializeButterworthCoefficients();
    
private:
    IMUCompensatorConfig config_;
    
    // Current state
    Eigen::Vector3d filtered_acceleration_;
    Eigen::Vector3d angular_velocity_;
    Eigen::Quaterniond orientation_;
    
    // EMA filter state
    Eigen::Vector3d ema_state_;
    bool ema_initialized_;
    
    // Butterworth filter state
    std::deque<Eigen::Vector3d> input_history_;
    std::deque<Eigen::Vector3d> output_history_;
    std::vector<double> butterworth_a_;  // Denominator coefficients
    std::vector<double> butterworth_b_;  // Numerator coefficients
    
    // Timestamp tracking
    double last_timestamp_;
    bool first_measurement_;
};

} // namespace utils
} // namespace calico

#endif // CALICO_UTILS_IMU_COMPENSATOR_HPP