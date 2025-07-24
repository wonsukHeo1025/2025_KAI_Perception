#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cone_stellation/common/cone.hpp"
#include "cone_stellation/common/estimation_frame.hpp"

namespace cone_stellation {

/**
 * @brief Abstract base class for cone-based odometry estimation
 * 
 * Estimates vehicle motion by matching cone observations between frames.
 * Similar to GLIM's OdometryEstimationBase but for sparse cone features
 * instead of dense point clouds.
 */
class ConeOdometryBase {
public:
  using Ptr = std::shared_ptr<ConeOdometryBase>;
  
  struct Config {
    // Matching parameters
    double max_correspondence_distance = 3.0;  // Max distance for cone matching
    bool use_color_constraint = true;         // Use color in matching
    double color_mismatch_penalty = 10.0;     // Penalty for color mismatch
    
    // Optimization parameters
    int max_iterations = 50;
    double convergence_threshold = 1e-6;
    
    // Outlier rejection
    double outlier_threshold = 2.0;           // Robust kernel threshold
    
    // Minimum requirements
    int min_correspondences = 3;              // Minimum matched cones
  };
  
  ConeOdometryBase() = default;
  virtual ~ConeOdometryBase() = default;
  
  /**
   * @brief Estimate relative transformation between two cone observation sets
   * 
   * @param prev_frame Previous frame with cone observations
   * @param curr_frame Current frame with cone observations  
   * @return Estimated transformation from previous to current frame
   */
  virtual Eigen::Isometry3d estimate(
      const EstimationFrame::ConstPtr& prev_frame,
      const EstimationFrame::ConstPtr& curr_frame) = 0;
  
  /**
   * @brief Get the number of inlier correspondences from last estimation
   */
  virtual int num_inliers() const = 0;
  
  /**
   * @brief Get correspondence pairs from last estimation
   * @return Vector of (prev_idx, curr_idx) pairs
   */
  virtual std::vector<std::pair<int, int>> get_correspondences() const = 0;
  
  /**
   * @brief Get the name of this odometry method
   */
  virtual std::string name() const = 0;
  
protected:
  /**
   * @brief Find correspondences between cone sets
   * 
   * @param prev_cones Previous frame cones in vehicle frame
   * @param curr_cones Current frame cones in vehicle frame
   * @param initial_guess Initial transformation guess
   * @return Correspondence pairs (prev_idx, curr_idx)
   */
  virtual std::vector<std::pair<int, int>> find_correspondences(
      const std::vector<ConeObservation>& prev_cones,
      const std::vector<ConeObservation>& curr_cones,
      const Eigen::Isometry3d& initial_guess) = 0;
};

} // namespace cone_stellation