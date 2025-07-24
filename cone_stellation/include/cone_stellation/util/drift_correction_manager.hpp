#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {

/**
 * @brief Manages drift correction between map and odom frames
 * 
 * This class maintains a history of odometry poses and calculates the
 * drift correction transform (map->odom) when SLAM optimization provides
 * corrected poses. Based on GLIM's TrajectoryManager design.
 * 
 * Usage:
 * 1. Continuously add odometry poses with add_odometry_pose()
 * 2. When SLAM optimization completes, call update_slam_pose()
 * 3. Retrieve map->odom transform with get_map_to_odom()
 */
class DriftCorrectionManager {
public:
  DriftCorrectionManager() 
    : T_map_odom_(Eigen::Isometry3d::Identity()),
      history_duration_(10.0) {  // Keep 10 seconds of history
    RCLCPP_INFO(rclcpp::get_logger("drift_correction"), 
                "DriftCorrectionManager initialized");
  }
  
  /**
   * @brief Add an odometry pose to the history buffer
   * @param timestamp Time of the odometry measurement
   * @param T_odom_base Transform from odom to base_link frame
   */
  void add_odometry_pose(double timestamp, const Eigen::Isometry3d& T_odom_base) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add to buffer
    odom_buffer_.emplace_back(timestamp, T_odom_base);
    
    // Remove old entries beyond history duration
    while (!odom_buffer_.empty() && 
           odom_buffer_.front().first < timestamp - history_duration_) {
      odom_buffer_.pop_front();
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("drift_correction"),
                 "Added odometry pose at %.3f, buffer size: %zu",
                 timestamp, odom_buffer_.size());
  }
  
  /**
   * @brief Update drift correction using SLAM optimized pose
   * @param timestamp Time of the SLAM pose
   * @param T_map_base Transform from map to base_link frame (from SLAM)
   */
  void update_slam_pose(double timestamp, const Eigen::Isometry3d& T_map_base) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find corresponding odometry pose at this timestamp
    auto T_odom_base = interpolate_odometry(timestamp);
    
    if (T_odom_base) {
      // Calculate drift: map->odom = (map->base) * (odom->base)^-1
      T_map_odom_ = T_map_base * T_odom_base->inverse();
      
      // Log drift magnitude for debugging
      double drift_translation = T_map_odom_.translation().norm();
      double drift_rotation = Eigen::AngleAxisd(T_map_odom_.rotation()).angle();
      
      RCLCPP_INFO(rclcpp::get_logger("drift_correction"),
                  "Drift correction updated: translation=%.3fm, rotation=%.3frad",
                  drift_translation, drift_rotation);
    } else {
      RCLCPP_WARN(rclcpp::get_logger("drift_correction"),
                  "No odometry data available at timestamp %.3f", timestamp);
    }
  }
  
  /**
   * @brief Get the current map->odom transform
   * @return Transform from map to odom frame
   */
  Eigen::Isometry3d get_map_to_odom() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return T_map_odom_;
  }
  
  /**
   * @brief Set history duration (how long to keep odometry poses)
   * @param duration Duration in seconds
   */
  void set_history_duration(double duration) {
    std::lock_guard<std::mutex> lock(mutex_);
    history_duration_ = duration;
  }
  
  /**
   * @brief Get current buffer size for debugging
   */
  size_t get_buffer_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return odom_buffer_.size();
  }

private:
  /**
   * @brief Interpolate odometry pose at given timestamp
   * @param timestamp Query timestamp
   * @return Interpolated pose or nullopt if insufficient data
   */
  std::optional<Eigen::Isometry3d> interpolate_odometry(double timestamp) {
    if (odom_buffer_.empty()) {
      return std::nullopt;
    }
    
    // Find the first element with timestamp >= query timestamp
    auto it = std::lower_bound(odom_buffer_.begin(), odom_buffer_.end(),
                              timestamp, 
                              [](const auto& a, double t) {
                                return a.first < t;
                              });
    
    // Case 1: Query is after all data - use latest
    if (it == odom_buffer_.end()) {
      RCLCPP_DEBUG(rclcpp::get_logger("drift_correction"),
                   "Query timestamp %.3f is after latest data %.3f",
                   timestamp, odom_buffer_.back().first);
      return odom_buffer_.back().second;
    }
    
    // Case 2: Query is before or at first data - use first
    if (it == odom_buffer_.begin()) {
      RCLCPP_DEBUG(rclcpp::get_logger("drift_correction"),
                   "Query timestamp %.3f is before earliest data %.3f",
                   timestamp, odom_buffer_.front().first);
      return it->second;
    }
    
    // Case 3: Interpolate between two poses
    auto prev = std::prev(it);
    double t0 = prev->first;
    double t1 = it->first;
    
    // Check for exact match
    if (std::abs(timestamp - t0) < 1e-6) {
      return prev->second;
    }
    if (std::abs(timestamp - t1) < 1e-6) {
      return it->second;
    }
    
    // Linear interpolation parameter
    double alpha = (timestamp - t0) / (t1 - t0);
    
    // Interpolate pose
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    
    // Linear interpolation for translation
    result.translation() = (1.0 - alpha) * prev->second.translation() + 
                          alpha * it->second.translation();
    
    // SLERP for rotation
    Eigen::Quaterniond q0(prev->second.rotation());
    Eigen::Quaterniond q1(it->second.rotation());
    Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
    result.linear() = q_interp.toRotationMatrix();
    
    RCLCPP_DEBUG(rclcpp::get_logger("drift_correction"),
                 "Interpolated pose at %.3f between %.3f and %.3f (alpha=%.3f)",
                 timestamp, t0, t1, alpha);
    
    return result;
  }
  
  // Odometry history buffer: (timestamp, T_odom_base)
  std::deque<std::pair<double, Eigen::Isometry3d>> odom_buffer_;
  
  // Current drift correction transform
  Eigen::Isometry3d T_map_odom_;
  
  // How long to keep odometry history (seconds)
  double history_duration_;
  
  // Thread safety
  mutable std::mutex mutex_;
};

}  // namespace cone_stellation