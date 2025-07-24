#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

namespace cone_stellation {

/**
 * @brief Factor constraining the distance between two cone landmarks
 * 
 * This factor encodes that two cones observed from the same pose
 * have a known relative distance that should be preserved
 */
class ConeDistanceFactor : public gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Point2> {
public:
  ConeDistanceFactor(gtsam::Key landmark1_key, gtsam::Key landmark2_key,
                     double measured_distance, const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Point2>(model, landmark1_key, landmark2_key),
      measured_distance_(measured_distance) {}
  
  gtsam::Vector evaluateError(const gtsam::Point2& l1, const gtsam::Point2& l2,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    gtsam::Vector2 diff = l2 - l1;
    double distance = diff.norm();
    
    // Residual is difference from measured distance
    gtsam::Vector1 residual;
    residual << distance - measured_distance_;
    
    // Jacobians if requested
    if (H1) {
      *H1 = -diff.transpose() / distance;  // 1x2
    }
    if (H2) {
      *H2 = diff.transpose() / distance;   // 1x2
    }
    
    return residual;
  }
  
  double measured_distance() const { return measured_distance_; }

private:
  double measured_distance_;
};

/**
 * @brief Factor constraining three cones to lie on a straight line
 * 
 * This factor is activated when cone detection identifies a line pattern
 */
class ConeLineFactor : public gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2> {
public:
  ConeLineFactor(gtsam::Key landmark1_key, gtsam::Key landmark2_key, gtsam::Key landmark3_key,
                 const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2>(
        model, landmark1_key, landmark2_key, landmark3_key) {}
  
  gtsam::Vector evaluateError(const gtsam::Point2& l1, const gtsam::Point2& l2, const gtsam::Point2& l3,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none) const override {
    // Calculate cross product to measure deviation from line
    // If points are collinear, cross product should be zero
    gtsam::Vector2 v1 = l2 - l1;
    gtsam::Vector2 v2 = l3 - l1;
    
    // 2D cross product (scalar)
    double cross = v1.x() * v2.y() - v1.y() * v2.x();
    
    // Normalize by distance to make scale-invariant
    double norm_factor = v1.norm() * v2.norm();
    if (norm_factor > 1e-6) {
      cross /= norm_factor;
    }
    
    gtsam::Vector1 residual;
    residual << cross;
    
    // Jacobians (derived from cross product)
    if (H1 || H2 || H3) {
      double n1 = v1.norm();
      double n2 = v2.norm();
      double n12 = n1 * n2;
      
      if (n12 > 1e-6) {
        if (H1) {
          *H1 = gtsam::Matrix12();
          (*H1)(0, 0) = (v2.y() - v1.y()) / n12;
          (*H1)(0, 1) = (v1.x() - v2.x()) / n12;
        }
        if (H2) {
          *H2 = gtsam::Matrix12();
          (*H2)(0, 0) = -v2.y() / n12;
          (*H2)(0, 1) = v2.x() / n12;
        }
        if (H3) {
          *H3 = gtsam::Matrix12();
          (*H3)(0, 0) = v1.y() / n12;
          (*H3)(0, 1) = -v1.x() / n12;
        }
      } else {
        if (H1) *H1 = gtsam::Matrix12::Zero();
        if (H2) *H2 = gtsam::Matrix12::Zero();
        if (H3) *H3 = gtsam::Matrix12::Zero();
      }
    }
    
    return residual;
  }
};

/**
 * @brief Factor constraining relative angle between three cones
 * 
 * Useful for maintaining geometric patterns like corners or curves
 */
class ConeAngleFactor : public gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2> {
public:
  ConeAngleFactor(gtsam::Key landmark1_key, gtsam::Key landmark2_key, gtsam::Key landmark3_key,
                  double measured_angle, const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor3<gtsam::Point2, gtsam::Point2, gtsam::Point2>(
        model, landmark1_key, landmark2_key, landmark3_key),
      measured_angle_(measured_angle) {}
  
  gtsam::Vector evaluateError(const gtsam::Point2& l1, const gtsam::Point2& l2, const gtsam::Point2& l3,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none) const override {
    // Calculate angle at l2 between l1-l2-l3
    gtsam::Vector2 v1 = l1 - l2;  // Vector from l2 to l1
    gtsam::Vector2 v2 = l3 - l2;  // Vector from l2 to l3
    
    double n1 = v1.norm();
    double n2 = v2.norm();
    
    if (n1 < 1e-6 || n2 < 1e-6) {
      if (H1) *H1 = gtsam::Matrix12::Zero();
      if (H2) *H2 = gtsam::Matrix12::Zero();
      if (H3) *H3 = gtsam::Matrix12::Zero();
      return gtsam::Vector1::Zero();
    }
    
    // Angle between vectors
    double cos_angle = v1.dot(v2) / (n1 * n2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // Clamp to avoid numerical issues
    double angle = std::acos(cos_angle);
    
    gtsam::Vector1 residual;
    residual << angle - measured_angle_;
    
    // TODO: Implement Jacobians for angle constraint
    if (H1) *H1 = gtsam::Matrix12::Zero();
    if (H2) *H2 = gtsam::Matrix12::Zero();
    if (H3) *H3 = gtsam::Matrix12::Zero();
    
    return residual;
  }
  
  double measured_angle() const { return measured_angle_; }

private:
  double measured_angle_;
};

/**
 * @brief Factor for parallel line constraint between two sets of cones
 * 
 * Enforces that cones form parallel track boundaries
 */
class ConeParallelLinesFactor : public gtsam::NoiseModelFactor4<
    gtsam::Point2, gtsam::Point2, gtsam::Point2, gtsam::Point2> {
public:
  ConeParallelLinesFactor(gtsam::Key l1_key, gtsam::Key l2_key, 
                         gtsam::Key l3_key, gtsam::Key l4_key,
                         const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor4<gtsam::Point2, gtsam::Point2, gtsam::Point2, gtsam::Point2>(
        model, l1_key, l2_key, l3_key, l4_key) {}
  
  gtsam::Vector evaluateError(const gtsam::Point2& l1, const gtsam::Point2& l2,
                              const gtsam::Point2& l3, const gtsam::Point2& l4,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none,
                              boost::optional<gtsam::Matrix&> H4 = boost::none) const override {
    // Line 1: l1-l2, Line 2: l3-l4
    gtsam::Vector2 v1 = l2 - l1;
    gtsam::Vector2 v2 = l4 - l3;
    
    // Normalize directions
    double n1 = v1.norm();
    double n2 = v2.norm();
    
    if (n1 < 1e-6 || n2 < 1e-6) {
      if (H1) *H1 = gtsam::Matrix12::Zero();
      if (H2) *H2 = gtsam::Matrix12::Zero();
      if (H3) *H3 = gtsam::Matrix12::Zero();
      if (H4) *H4 = gtsam::Matrix12::Zero();
      return gtsam::Vector1::Zero();
    }
    
    v1 /= n1;
    v2 /= n2;
    
    // Cross product for parallel check (should be zero)
    double cross = v1.x() * v2.y() - v1.y() * v2.x();
    
    gtsam::Vector1 residual;
    residual << cross;
    
    // TODO: Implement Jacobians
    if (H1) *H1 = gtsam::Matrix12::Zero();
    if (H2) *H2 = gtsam::Matrix12::Zero();
    if (H3) *H3 = gtsam::Matrix12::Zero();
    if (H4) *H4 = gtsam::Matrix12::Zero();
    
    return residual;
  }
};

} // namespace cone_stellation