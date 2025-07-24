#pragma once

#include <memory>
#include <deque>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

#include "cone_stellation/common/cone.hpp"
#include "cone_stellation/common/estimation_frame.hpp"

namespace cone_stellation {

/**
 * @brief Preprocessing module for cone observations
 * 
 * Handles:
 * - Cone tracking across frames
 * - Outlier rejection
 * - Pattern detection
 * - Uncertainty estimation
 */
class ConePreprocessor {
public:
  using Ptr = std::shared_ptr<ConePreprocessor>;
  
  struct Config {
    // Outlier rejection
    double max_cone_distance;      // Maximum distance to consider cone
    double min_cone_confidence;     // Minimum detection confidence
    
    // Pattern detection
    bool enable_pattern_detection;
    double line_fitting_threshold;  // Max distance from line
    int min_cones_for_line;          // Minimum cones to fit line
    
    // Tracking
    double association_threshold;   // Max distance for same cone
    int max_tracking_frames;        // Track cones for N frames
    
    Config() : max_cone_distance(20.0), min_cone_confidence(0.5),
               enable_pattern_detection(true), line_fitting_threshold(0.2),
               min_cones_for_line(3), association_threshold(1.0),
               max_tracking_frames(10) {}
  };
  
  ConePreprocessor(const Config& config = Config()) : config_(config) {}
  
  /**
   * @brief Process raw cone observations
   * 
   * @param raw_observations Raw cone detections from sensor
   * @param sensor_pose Current sensor pose estimate
   * @param timestamp Observation timestamp
   * @return Processed observation set with patterns
   */
  std::shared_ptr<ConeObservationSet> process(
      const std::vector<ConeObservation>& raw_observations,
      const Eigen::Isometry3d& sensor_pose,
      double timestamp) {
    
    auto processed = std::make_shared<ConeObservationSet>();
    processed->sensor_pose = sensor_pose;
    processed->timestamp = timestamp;
    
    // Filter outliers
    for (const auto& obs : raw_observations) {
      if (is_valid_observation(obs)) {
        processed->cones.push_back(obs);
      }
    }
    
    // Detect patterns if enabled
    if (config_.enable_pattern_detection && static_cast<int>(processed->cones.size()) >= config_.min_cones_for_line) {
      detect_patterns(*processed);
    }
    
    // Update tracking
    update_tracking(*processed);
    
    return processed;
  }
  
  /**
   * @brief Get detected patterns from recent observations
   */
  std::vector<ConePattern> get_recent_patterns() const {
    std::lock_guard<std::mutex> lock(patterns_mutex_);
    return std::vector<ConePattern>(recent_patterns_.begin(), recent_patterns_.end());
  }

private:
  /**
   * @brief Check if cone observation is valid
   */
  bool is_valid_observation(const ConeObservation& obs) const {
    // Check distance
    double distance = obs.position.norm();
    if (distance > config_.max_cone_distance) {
      return false;
    }
    
    // Check confidence
    if (obs.confidence < config_.min_cone_confidence) {
      return false;
    }
    
    // Additional checks can be added here
    return true;
  }
  
  /**
   * @brief Detect geometric patterns in cone observations
   */
  void detect_patterns(ConeObservationSet& obs_set) {
    // Line detection using RANSAC or least squares
    detect_line_patterns(obs_set);
    
    // Curve detection
    // detect_curve_patterns(obs_set);
    
    // Parallel lines (track boundaries)
    // detect_parallel_patterns(obs_set);
  }
  
  /**
   * @brief Detect straight line patterns
   */
  void detect_line_patterns(ConeObservationSet& obs_set) {
    const auto& cones = obs_set.cones;
    if (static_cast<int>(cones.size()) < config_.min_cones_for_line) {
      return;
    }
    
    // Simple approach: try all combinations of 3+ cones
    // In practice, use RANSAC or Hough transform
    for (size_t i = 0; i < cones.size() - 2; ++i) {
      for (size_t j = i + 1; j < cones.size() - 1; ++j) {
        for (size_t k = j + 1; k < cones.size(); ++k) {
          // Check if cones i, j, k are collinear
          if (are_cones_collinear(cones[i], cones[j], cones[k])) {
            ConePattern pattern;
            pattern.type = ConePattern::LINE;
            pattern.cone_ids = {cones[i].id, cones[j].id, cones[k].id};
            pattern.confidence = 0.8; // Simplified
            
            // Store line parameters (e.g., direction, offset)
            pattern.parameters = fit_line(cones[i].position, cones[j].position, cones[k].position);
            
            // Check if more cones lie on this line
            for (size_t m = 0; m < cones.size(); ++m) {
              if (m != i && m != j && m != k) {
                if (distance_to_line(cones[m].position, pattern.parameters) < config_.line_fitting_threshold) {
                  pattern.cone_ids.push_back(cones[m].id);
                }
              }
            }
            
            // Add pattern if it has enough cones
            if (static_cast<int>(pattern.cone_ids.size()) >= config_.min_cones_for_line) {
              obs_set.detected_patterns.push_back(pattern);
            }
          }
        }
      }
    }
  }
  
  /**
   * @brief Check if three cones are approximately collinear
   */
  bool are_cones_collinear(const ConeObservation& c1, const ConeObservation& c2, 
                          const ConeObservation& c3) const {
    Eigen::Vector2d v1 = c2.position - c1.position;
    Eigen::Vector2d v2 = c3.position - c1.position;
    
    // Cross product
    double cross = v1.x() * v2.y() - v1.y() * v2.x();
    double area = std::abs(cross) / 2.0;
    
    // Check if area is small relative to distances
    double max_dist = std::max({v1.norm(), v2.norm(), (c3.position - c2.position).norm()});
    return (area / max_dist) < config_.line_fitting_threshold;
  }
  
  /**
   * @brief Fit line to points and return parameters
   */
  Eigen::VectorXd fit_line(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, 
                          const Eigen::Vector2d& p3) const {
    // Simple line representation: ax + by + c = 0
    // For now, just use two points (p3 reserved for future least-squares fitting)
    (void)p3; // Suppress unused parameter warning
    Eigen::Vector2d dir = (p2 - p1).normalized();
    Eigen::Vector2d normal(-dir.y(), dir.x());
    double d = -normal.dot(p1);
    
    Eigen::VectorXd params(3);
    params << normal.x(), normal.y(), d;
    return params;
  }
  
  /**
   * @brief Calculate distance from point to line
   */
  double distance_to_line(const Eigen::Vector2d& point, const Eigen::VectorXd& line_params) const {
    // Line: ax + by + c = 0
    double a = line_params(0);
    double b = line_params(1);
    double c = line_params(2);
    
    return std::abs(a * point.x() + b * point.y() + c) / std::sqrt(a * a + b * b);
  }
  
  /**
   * @brief Update cone tracking across frames
   */
  void update_tracking(ConeObservationSet& obs_set) {
    // Simple tracking: assign consistent IDs based on position
    // In practice, use more sophisticated tracking (e.g., Hungarian algorithm)
    
    static int next_track_id = 0;
    
    // For now, just assign incremental IDs
    for (auto& cone : obs_set.cones) {
      if (cone.id < 0) {
        cone.id = next_track_id++;
      }
    }
  }
  
  Config config_;
  
  // Pattern detection history
  mutable std::mutex patterns_mutex_;
  std::deque<ConePattern> recent_patterns_;
  
  // Tracking state
  std::unordered_map<int, Eigen::Vector2d> tracked_positions_;
  int next_track_id = 0;
};

} // namespace cone_stellation