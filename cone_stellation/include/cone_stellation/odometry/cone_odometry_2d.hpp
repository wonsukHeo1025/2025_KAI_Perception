#pragma once

#include <unordered_map>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/numericalDerivative.h>

#include "cone_stellation/odometry/cone_odometry_base.hpp"
#include "cone_stellation/factors/cone_observation_factor.hpp"

namespace cone_stellation {

/**
 * @brief 2D cone-based odometry estimation using factor graph optimization
 * 
 * Estimates 2D motion (x, y, theta) by matching cone observations between frames
 * and optimizing the relative pose using GTSAM.
 * 
 * Algorithm:
 * 1. Find cone correspondences using nearest neighbor + color constraint
 * 2. Create factor graph with cone observation factors
 * 3. Optimize relative pose with robust kernels
 * 4. Reject outliers and re-optimize if needed
 */
class ConeOdometry2D : public ConeOdometryBase {
public:
  using Ptr = std::shared_ptr<ConeOdometry2D>;
  
  ConeOdometry2D(const Config& config = Config()) 
    : config_(config) {}
  
  /**
   * @brief Estimate 2D relative transformation between cone observations
   */
  Eigen::Isometry3d estimate(
      const EstimationFrame::ConstPtr& prev_frame,
      const EstimationFrame::ConstPtr& curr_frame) override {
    
    if (!prev_frame->cone_observations || !curr_frame->cone_observations) {
      RCLCPP_WARN(rclcpp::get_logger("cone_odometry"), 
                  "No cone observations in one of the frames");
      return Eigen::Isometry3d::Identity();
    }
    
    const auto& prev_cones = prev_frame->cone_observations->cones;
    const auto& curr_cones = curr_frame->cone_observations->cones;
    
    // Initial guess: no motion
    Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();
    
    // Find correspondences
    correspondences_ = find_correspondences(prev_cones, curr_cones, initial_guess);
    
    if (correspondences_.size() < config_.min_correspondences) {
      RCLCPP_WARN(rclcpp::get_logger("cone_odometry"), 
                  "Not enough correspondences: %zu < %d", 
                  correspondences_.size(), config_.min_correspondences);
      return Eigen::Isometry3d::Identity();
    }
    
    // Create factor graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_values;
    
    // Add pose variables
    gtsam::Symbol prev_pose('x', 0);
    gtsam::Symbol curr_pose('x', 1);
    
    // Fix previous pose at origin
    initial_values.insert(prev_pose, gtsam::Pose2(0, 0, 0));
    initial_values.insert(curr_pose, gtsam::Pose2(0, 0, 0)); // Initial guess
    
    // Add prior on previous pose
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector3(1e-6, 1e-6, 1e-6));
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(prev_pose, gtsam::Pose2(0, 0, 0), prior_noise));
    
    // Add cone observation factors for each correspondence
    for (const auto& [prev_idx, curr_idx] : correspondences_) {
      const auto& prev_cone = prev_cones[prev_idx];
      const auto& curr_cone = curr_cones[curr_idx];
      
      // Create temporary landmarks for optimization
      gtsam::Symbol landmark('l', prev_idx);
      
      // Initialize landmark at observed position from previous frame
      initial_values.insert(landmark, gtsam::Point2(prev_cone.position.x(), prev_cone.position.y()));
      
      // Observation from previous frame
      auto obs_noise = gtsam::noiseModel::Diagonal::Sigmas(
          gtsam::Vector2(0.1, 0.1)); // 10cm observation noise
      
      // Use robust kernel for outlier rejection
      auto robust_noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(config_.outlier_threshold),
          obs_noise);
      
      graph.emplace_shared<ConeObservationFactor>(
          prev_pose, landmark, gtsam::Point2(prev_cone.position.x(), prev_cone.position.y()), robust_noise);
      
      // Observation from current frame  
      graph.emplace_shared<ConeObservationFactor>(
          curr_pose, landmark, gtsam::Point2(curr_cone.position.x(), curr_cone.position.y()), robust_noise);
    }
    
    // Optimize
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = config_.max_iterations;
    params.relativeErrorTol = config_.convergence_threshold;
    // params.verbosity = gtsam::LevenbergMarquardtParams::SILENT; // Commented out due to type mismatch
    
    gtsam::Values result;
    try {
      gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);
      result = optimizer.optimize();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("cone_odometry"), 
                   "Optimization failed: %s", e.what());
      return Eigen::Isometry3d::Identity();
    }
    
    // Extract relative pose
    gtsam::Pose2 prev_pose_2d, curr_pose_2d;
    try {
      prev_pose_2d = result.at<gtsam::Pose2>(prev_pose);
      curr_pose_2d = result.at<gtsam::Pose2>(curr_pose);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("cone_odometry"), 
                   "Failed to extract poses: %s", e.what());
      return Eigen::Isometry3d::Identity();
    }
    
    gtsam::Pose2 relative_pose_2d = prev_pose_2d.between(curr_pose_2d);
    
    // Convert to 3D
    Eigen::Isometry3d relative_pose = Eigen::Isometry3d::Identity();
    relative_pose.translation().x() = relative_pose_2d.x();
    relative_pose.translation().y() = relative_pose_2d.y();
    relative_pose.rotate(Eigen::AngleAxisd(relative_pose_2d.theta(), Eigen::Vector3d::UnitZ()));
    
    // Count inliers (cones with low residual error)
    num_inliers_ = 0;
    for (const auto& [prev_idx, curr_idx] : correspondences_) {
      gtsam::Symbol landmark('l', prev_idx);
      gtsam::Point2 landmark_pos = result.at<gtsam::Point2>(landmark);
      
      // Check residuals
      gtsam::Point2 prev_expected = prev_pose_2d.transformTo(landmark_pos);
      gtsam::Point2 curr_expected = curr_pose_2d.transformTo(landmark_pos);
      
      double prev_error = (prev_expected - gtsam::Point2(prev_cones[prev_idx].position.x(), prev_cones[prev_idx].position.y())).norm();
      double curr_error = (curr_expected - gtsam::Point2(curr_cones[curr_idx].position.x(), curr_cones[curr_idx].position.y())).norm();
      
      if (prev_error < config_.outlier_threshold && curr_error < config_.outlier_threshold) {
        num_inliers_++;
      }
    }
    
    RCLCPP_INFO(rclcpp::get_logger("cone_odometry"), 
                "Odometry: %zu correspondences, %d inliers, delta: (%.3f, %.3f, %.3f rad)",
                correspondences_.size(), num_inliers_,
                relative_pose.translation().x(), 
                relative_pose.translation().y(),
                relative_pose_2d.theta());
    
    return relative_pose;
  }
  
  int num_inliers() const override { return num_inliers_; }
  
  std::vector<std::pair<int, int>> get_correspondences() const override {
    return correspondences_;
  }
  
  std::string name() const override { return "ConeOdometry2D"; }
  
protected:
  /**
   * @brief Find cone correspondences using nearest neighbor with color constraint
   */
  std::vector<std::pair<int, int>> find_correspondences(
      const std::vector<ConeObservation>& prev_cones,
      const std::vector<ConeObservation>& curr_cones,
      const Eigen::Isometry3d& initial_guess) override {
    
    std::vector<std::pair<int, int>> correspondences;
    
    // Convert initial guess to 2D
    double theta = std::atan2(initial_guess.rotation()(1,0), initial_guess.rotation()(0,0));
    gtsam::Pose2 T_prev_curr(initial_guess.translation().x(), 
                            initial_guess.translation().y(), 
                            theta);
    
    // For each cone in current frame, find best match in previous frame
    std::unordered_map<int, int> prev_used; // Track used previous cones
    
    for (size_t curr_idx = 0; curr_idx < curr_cones.size(); curr_idx++) {
      const auto& curr_cone = curr_cones[curr_idx];
      
      // Transform current cone to previous frame using initial guess
      gtsam::Point2 curr_in_prev = T_prev_curr.inverse().transformFrom(
          gtsam::Point2(curr_cone.position.x(), curr_cone.position.y()));
      
      // Find nearest cone in previous frame
      double best_dist = config_.max_correspondence_distance;
      int best_prev_idx = -1;
      
      for (size_t prev_idx = 0; prev_idx < prev_cones.size(); prev_idx++) {
        // Skip if already used
        if (prev_used.count(prev_idx) > 0) continue;
        
        const auto& prev_cone = prev_cones[prev_idx];
        
        // Check color constraint
        if (config_.use_color_constraint && 
            prev_cone.color != ConeColor::UNKNOWN &&
            curr_cone.color != ConeColor::UNKNOWN &&
            prev_cone.color != curr_cone.color) {
          continue;
        }
        
        // Calculate distance
        double dist = (gtsam::Point2(prev_cone.position.x(), prev_cone.position.y()) - curr_in_prev).norm();
        
        // Apply color mismatch penalty if colors don't match
        if (prev_cone.color != curr_cone.color) {
          dist += config_.color_mismatch_penalty;
        }
        
        if (dist < best_dist) {
          best_dist = dist;
          best_prev_idx = prev_idx;
        }
      }
      
      // Add correspondence if found
      if (best_prev_idx >= 0) {
        correspondences.push_back({best_prev_idx, curr_idx});
        prev_used[best_prev_idx] = curr_idx;
      }
    }
    
    return correspondences;
  }
  
private:
  Config config_;
  std::vector<std::pair<int, int>> correspondences_;
  int num_inliers_ = 0;
};

} // namespace cone_stellation