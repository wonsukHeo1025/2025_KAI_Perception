#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>

#include "cone_stellation/common/cone.hpp"

namespace cone_stellation {

/**
 * @brief Simple data association for cone observations to landmarks
 * 
 * Uses nearest neighbor matching with color constraints and gating.
 * This is a basic implementation to get SLAM working before implementing
 * more sophisticated methods like JCBB.
 */
class DataAssociation {
public:
  using Ptr = std::shared_ptr<DataAssociation>;
  
  struct Config {
    double max_association_distance;  // Maximum distance for association
    bool use_color_constraint;       // Enforce color matching
    double gating_threshold;          // Chi-squared threshold for gating
    double track_id_weight;          // Weight for track ID matching
    bool use_track_id;               // Enable track ID in association
    
    Config() : max_association_distance(2.0), use_color_constraint(true), 
               gating_threshold(5.0), track_id_weight(5.0), use_track_id(true) {}
  };
  
  DataAssociation(const Config& config = Config()) : config_(config) {}
  
  /**
   * @brief Associate cone observations with existing landmarks
   * 
   * @param observations Current cone observations in world frame
   * @param landmarks Existing landmarks
   * @param pose_covariance Current pose uncertainty (3x3: x,y,theta)
   * @return Map from observation index to landmark ID (-1 if new landmark)
   */
  std::unordered_map<int, int> associate(
      const std::vector<ConeObservation>& observations,
      const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
      const Eigen::Matrix3d& /*pose_covariance*/ = Eigen::Matrix3d::Identity()) {
    
    std::unordered_map<int, int> associations;
    std::unordered_set<int> used_landmarks;
    
    // For each observation, find best matching landmark
    for (size_t obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
      const auto& obs = observations[obs_idx];
      
      int best_landmark_id = -1;
      double best_score = std::numeric_limits<double>::max();
      
      // Search through all landmarks
      for (const auto& [landmark_id, landmark] : landmarks) {
        // Skip if already associated in this frame
        if (used_landmarks.count(landmark_id) > 0) continue;
        
        // Color constraint
        if (config_.use_color_constraint && 
            obs.color != ConeColor::UNKNOWN &&
            landmark->color() != ConeColor::UNKNOWN &&
            obs.color != landmark->color()) {
          continue;
        }
        
        // Calculate distance
        double distance = (obs.position - landmark->position()).norm();
        
        // Skip if too far
        if (distance > config_.max_association_distance) continue;
        
        // Calculate association score (lower is better)
        double score = distance;
        
        // Track ID bonus (negative score for matching track IDs)
        if (config_.use_track_id && obs.id >= 0 && landmark->track_id() >= 0) {
          if (obs.id == landmark->track_id()) {
            score -= config_.track_id_weight;  // Prioritize track ID matches
          }
        }
        
        // Update best match
        if (score < best_score) {
          best_score = score;
          best_landmark_id = landmark_id;
        }
      }
      
      // Record association
      associations[obs_idx] = best_landmark_id;
      if (best_landmark_id >= 0) {
        used_landmarks.insert(best_landmark_id);
        
        RCLCPP_DEBUG(rclcpp::get_logger("data_association"),
                     "Associated obs %zu (%.2f,%.2f) track:%d color:%d with landmark %d (%.2f,%.2f) track:%d color:%d, score:%.2f",
                     obs_idx, obs.position.x(), obs.position.y(), obs.id, static_cast<int>(obs.color),
                     best_landmark_id, 
                     landmarks.at(best_landmark_id)->position().x(),
                     landmarks.at(best_landmark_id)->position().y(),
                     landmarks.at(best_landmark_id)->track_id(),
                     static_cast<int>(landmarks.at(best_landmark_id)->color()),
                     best_score);
      } else {
        RCLCPP_DEBUG(rclcpp::get_logger("data_association"),
                     "No association for observation %zu (%.2f, %.2f) color %d",
                     obs_idx, obs.position.x(), obs.position.y(), static_cast<int>(obs.color));
      }
    }
    
    return associations;
  }
  
  /**
   * @brief Mahalanobis distance for more sophisticated gating
   * 
   * @param obs_position Observation position in world frame
   * @param landmark_position Landmark position
   * @param innovation_covariance Combined uncertainty (S = H*P*H' + R)
   * @return Mahalanobis distance squared
   */
  double mahalanobis_distance(
      const Eigen::Vector2d& obs_position,
      const Eigen::Vector2d& landmark_position,
      const Eigen::Matrix2d& innovation_covariance) {
    
    Eigen::Vector2d innovation = obs_position - landmark_position;
    return innovation.transpose() * innovation_covariance.inverse() * innovation;
  }

private:
  Config config_;
};

} // namespace cone_stellation