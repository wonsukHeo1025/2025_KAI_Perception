#pragma once

#include <memory>
#include <unordered_map>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include "cone_stellation/mapping/data_association.hpp"
#include <unordered_set>
#include <rclcpp/rclcpp.hpp>

#include "cone_stellation/common/cone.hpp"
#include "cone_stellation/common/estimation_frame.hpp"
#include "cone_stellation/factors/cone_observation_factor.hpp"

namespace cone_stellation {

/**
 * @brief Simplified cone mapping for debugging
 * Start with the most basic SLAM and gradually add features
 */
class SimpleConeMapping {
public:
  using Ptr = std::shared_ptr<SimpleConeMapping>;
  
  SimpleConeMapping() {
    // Very conservative ISAM2 parameters
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    params.enableRelinearization = true;
    params.evaluateNonlinearError = false;  // Skip error evaluation for speed
    params.factorization = gtsam::ISAM2Params::QR;  // More stable than CHOLESKY
    
    isam2_ = std::make_shared<gtsam::ISAM2>(params);
    
    // Initialize data association
    DataAssociation::Config da_config;
    da_config.max_association_distance = 2.0;
    da_config.use_color_constraint = true;
    data_association_ = std::make_shared<DataAssociation>(da_config);
    
    RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), 
                "SimpleConeMapping initialized with conservative parameters");
  }
  
  void add_keyframe(const EstimationFrame::Ptr& frame) {
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
    
    // Create pose symbol
    gtsam::Symbol pose_key('x', pose_count_);
    
    // Convert to 2D pose
    const auto& T = frame->T_world_sensor;
    double yaw = std::atan2(T.rotation()(1,0), T.rotation()(0,0));
    gtsam::Pose2 pose2d(T.translation().x(), T.translation().y(), yaw);
    
    // Add pose to values
    new_values.insert(pose_key, pose2d);
    
    // First pose gets a prior
    if (pose_count_ == 0) {
      auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
          gtsam::Vector3(0.01, 0.01, 0.01));  // Very tight prior
      new_factors.add(gtsam::PriorFactor<gtsam::Pose2>(pose_key, pose2d, prior_noise));
      
      RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), 
                  "Added prior to first pose at (%.2f, %.2f, %.2f)", 
                  pose2d.x(), pose2d.y(), pose2d.theta());
    } else {
      // Add odometry factor from previous pose
      gtsam::Symbol prev_pose_key('x', pose_count_ - 1);
      
      // Get previous pose from current estimate
      gtsam::Pose2 prev_pose;
      if (pose_count_ == 1) {
        // Use the prior value we just set
        prev_pose = last_pose_;
      } else {
        try {
          auto estimate = isam2_->calculateEstimate();
          prev_pose = estimate.at<gtsam::Pose2>(prev_pose_key);
        } catch (...) {
          RCLCPP_WARN(rclcpp::get_logger("simple_mapping"), 
                      "Failed to get previous pose, using last known pose");
          prev_pose = last_pose_;
        }
      }
      
      // Calculate odometry
      gtsam::Pose2 odom = prev_pose.between(pose2d);
      
      auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
          gtsam::Vector3(0.1, 0.1, 0.05));
      new_factors.add(gtsam::BetweenFactor<gtsam::Pose2>(
          prev_pose_key, pose_key, odom, odom_noise));
      
      RCLCPP_DEBUG(rclcpp::get_logger("simple_mapping"), 
                   "Added odometry factor: delta = (%.2f, %.2f, %.2f)", 
                   odom.x(), odom.y(), odom.theta());
    }
    
    // Process cone observations with data association
    if (frame->cone_observations) {
      const auto& obs_set = *frame->cone_observations;
      
      // Transform observations to world frame for association
      std::vector<ConeObservation> world_observations;
      for (const auto& obs : obs_set.cones) {
        ConeObservation world_obs = obs;
        world_obs.position = frame->transform_to_world(obs);
        world_observations.push_back(world_obs);
      }
      
      // Convert landmarks to format expected by data association
      std::unordered_map<int, ConeLandmark::Ptr> landmark_map;
      for (const auto& [id, simple_lm] : landmarks_) {
        auto lm = std::make_shared<ConeLandmark>(id, simple_lm.position, simple_lm.color);
        landmark_map[id] = lm;
      }
      
      // Perform data association
      auto associations = data_association_->associate(world_observations, landmark_map);
      
      // Process associations
      int new_landmarks_added = 0;
      for (const auto& [obs_idx, landmark_id] : associations) {
        if (obs_idx >= world_observations.size()) continue;
        if (obs_idx >= obs_set.cones.size()) continue;
        
        const auto& obs = obs_set.cones[obs_idx];
        const auto& world_obs = world_observations[obs_idx];
        
        if (landmark_id < 0 && new_landmarks_added < 3) {  // New landmark (limit to 3 per frame)
          // Create new landmark
          int new_id = landmark_count_++;
          gtsam::Symbol landmark_key('l', new_id);
          
          // Add to values
          new_values.insert(landmark_key, gtsam::Point2(world_obs.position.x(), world_obs.position.y()));
          
          // Add observation factor
          auto obs_noise = gtsam::noiseModel::Diagonal::Sigmas(
              gtsam::Vector2(0.2, 0.2));
          new_factors.add(ConeObservationFactor(
              pose_key, landmark_key, gtsam::Point2(obs.position.x(), obs.position.y()), obs_noise));
          
          // Add prior to first few landmarks
          if (new_id < 3) {
            auto landmark_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
                gtsam::Vector2(0.05, 0.05));
            new_factors.add(gtsam::PriorFactor<gtsam::Point2>(
                landmark_key, gtsam::Point2(world_obs.position.x(), world_obs.position.y()), landmark_prior_noise));
            
            RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), 
                        "Added landmark %d with prior at (%.2f, %.2f) color %d", 
                        new_id, world_obs.position.x(), world_obs.position.y(), 
                        static_cast<int>(obs.color));
          }
          
          landmarks_[new_id] = {world_obs.position, obs.color};
          new_landmarks_added++;
          
        } else if (landmark_id >= 0) {  // Existing landmark
          // Add observation factor to existing landmark
          gtsam::Symbol landmark_key('l', landmark_id);
          
          auto obs_noise = gtsam::noiseModel::Diagonal::Sigmas(
              gtsam::Vector2(0.2, 0.2));
          new_factors.add(ConeObservationFactor(
              pose_key, landmark_key, gtsam::Point2(obs.position.x(), obs.position.y()), obs_noise));
          
          RCLCPP_DEBUG(rclcpp::get_logger("simple_mapping"), 
                       "Associated observation to landmark %d", landmark_id);
        }
      }
    }
    
    // Update ISAM2
    try {
      RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), 
                  "Updating ISAM2 with %zu factors and %zu values", 
                  new_factors.size(), new_values.size());
      
      // Debug: Print all factors
      RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), "Factors:");
      for (size_t i = 0; i < new_factors.size(); i++) {
        RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), "  Factor %zu", i);
      }
      
      // Debug: Print all values
      RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), "Values:");
      for (const auto& key_value : new_values) {
        RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), "  Key: %s", 
                    gtsam::DefaultKeyFormatter(key_value.key).c_str());
      }
      
      gtsam::ISAM2Result result = isam2_->update(new_factors, new_values);
      
      RCLCPP_INFO(rclcpp::get_logger("simple_mapping"), 
                  "ISAM2 update successful: %zu cliques updated", 
                  result.cliques);
      
      // Store last pose
      last_pose_ = pose2d;
      
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
      RCLCPP_ERROR(rclcpp::get_logger("simple_mapping"), 
                   "IndeterminantLinearSystemException: %s", e.what());
      throw;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("simple_mapping"), 
                   "ISAM2 update failed: %s", e.what());
      throw;
    }
    
    pose_count_++;
  }
  
  gtsam::Values get_current_estimate() const {
    try {
      return isam2_->calculateEstimate();
    } catch (...) {
      return gtsam::Values();
    }
  }
  
  struct SimpleLandmark {
    Eigen::Vector2d position;
    ConeColor color;
  };
  
  std::unordered_map<int, SimpleLandmark> get_landmarks() const {
    return landmarks_;
  }
  
  gtsam::NonlinearFactorGraph get_factor_graph() const {
    return isam2_->getFactorsUnsafe();
  }

private:
  std::shared_ptr<gtsam::ISAM2> isam2_;
  std::shared_ptr<DataAssociation> data_association_;
  std::unordered_map<int, SimpleLandmark> landmarks_;
  
  int pose_count_ = 0;
  int landmark_count_ = 0;
  gtsam::Pose2 last_pose_;
};

} // namespace cone_stellation