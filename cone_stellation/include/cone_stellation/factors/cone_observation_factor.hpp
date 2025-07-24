#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

namespace cone_stellation {

/**
 * @brief Factor for cone observation from a pose
 * 
 * This factor models the observation of a cone landmark from a vehicle pose.
 * The measurement is in the vehicle frame (relative position).
 */
class ConeObservationFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2> {
public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2>;
  using shared_ptr = boost::shared_ptr<ConeObservationFactor>;

private:
  gtsam::Point2 measured_;  // Measured position in vehicle frame

public:
  ConeObservationFactor(gtsam::Key pose_key, gtsam::Key landmark_key,
                        const gtsam::Point2& measured,
                        const gtsam::SharedNoiseModel& noise_model)
      : Base(noise_model, pose_key, landmark_key), measured_(measured) {}

  virtual ~ConeObservationFactor() {}

  /**
   * @brief Evaluate the error between predicted and measured cone position
   */
  gtsam::Vector evaluateError(const gtsam::Pose2& pose,
                             const gtsam::Point2& landmark,
                             boost::optional<gtsam::Matrix&> H_pose = boost::none,
                             boost::optional<gtsam::Matrix&> H_landmark = boost::none) const override {
    
    // Transform landmark to vehicle frame
    gtsam::Point2 predicted = pose.transformTo(landmark, H_pose ? &(*H_pose) : nullptr, 
                                                          H_landmark ? &(*H_landmark) : nullptr);
    
    // Error is predicted - measured
    gtsam::Vector2 error = predicted - measured_;
    
    return error;
  }

  /**
   * @brief Clone the factor
   */
  virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ConeObservationFactor(*this)));
  }

  /**
   * @brief Print the factor
   */
  virtual void print(const std::string& s = "", 
                    const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "ConeObservationFactor on " << keyFormatter(this->key1()) 
              << " and " << keyFormatter(this->key2()) << std::endl;
    std::cout << "  measured: " << measured_.transpose() << std::endl;
    this->noiseModel_->print("  noise model: ");
  }

  /**
   * @brief Check equality
   */
  virtual bool equals(const gtsam::NonlinearFactor& expected, double tol = 1e-9) const override {
    const ConeObservationFactor* e = dynamic_cast<const ConeObservationFactor*>(&expected);
    return e != nullptr && Base::equals(*e, tol) && 
           gtsam::traits<gtsam::Point2>::Equals(measured_, e->measured_, tol);
  }
};

/**
 * @brief Alternative: Range-bearing factor for cone observation
 * 
 * This factor models the observation as range and bearing measurements
 */
class ConeRangeBearingFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2> {
public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2>;
  using shared_ptr = boost::shared_ptr<ConeRangeBearingFactor>;

private:
  double range_;    // Measured range to cone
  double bearing_;  // Measured bearing to cone (in vehicle frame)

public:
  ConeRangeBearingFactor(gtsam::Key pose_key, gtsam::Key landmark_key,
                         double range, double bearing,
                         const gtsam::SharedNoiseModel& noise_model)
      : Base(noise_model, pose_key, landmark_key), range_(range), bearing_(bearing) {}

  virtual ~ConeRangeBearingFactor() {}

  /**
   * @brief Evaluate the error between predicted and measured range/bearing
   */
  gtsam::Vector evaluateError(const gtsam::Pose2& pose,
                             const gtsam::Point2& landmark,
                             boost::optional<gtsam::Matrix&> H_pose = boost::none,
                             boost::optional<gtsam::Matrix&> H_landmark = boost::none) const override {
    
    // Get relative position
    gtsam::Point2 relative = pose.transformTo(landmark);
    
    // Predicted measurements
    double predicted_range = relative.norm();
    double predicted_bearing = std::atan2(relative.y(), relative.x());
    
    // Error vector [range_error, bearing_error]
    gtsam::Vector2 error;
    error(0) = predicted_range - range_;
    error(1) = gtsam::Pose2::Logmap(gtsam::Pose2(0, 0, predicted_bearing - bearing_))(2);
    
    // Compute Jacobians if requested
    if (H_pose || H_landmark) {
      // Jacobian of measurements w.r.t. relative position
      gtsam::Matrix22 H_rel;
      H_rel(0, 0) = relative.x() / predicted_range;  // d(range)/dx
      H_rel(0, 1) = relative.y() / predicted_range;  // d(range)/dy
      double range_sq = predicted_range * predicted_range;
      H_rel(1, 0) = -relative.y() / range_sq;        // d(bearing)/dx
      H_rel(1, 1) = relative.x() / range_sq;         // d(bearing)/dy
      
      // Chain rule for full Jacobians
      gtsam::Matrix23 H_pose_transform;
      gtsam::Matrix22 H_landmark_transform;
      pose.transformTo(landmark, H_pose ? &H_pose_transform : nullptr,
                                 H_landmark ? &H_landmark_transform : nullptr);
      
      if (H_pose) {
        *H_pose = H_rel * H_pose_transform;
      }
      if (H_landmark) {
        *H_landmark = H_rel * H_landmark_transform;
      }
    }
    
    return error;
  }

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ConeRangeBearingFactor(*this)));
  }

  virtual void print(const std::string& s = "", 
                    const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "ConeRangeBearingFactor on " << keyFormatter(this->key1()) 
              << " and " << keyFormatter(this->key2()) << std::endl;
    std::cout << "  range: " << range_ << ", bearing: " << bearing_ << std::endl;
    this->noiseModel_->print("  noise model: ");
  }

  virtual bool equals(const gtsam::NonlinearFactor& expected, double tol = 1e-9) const override {
    const ConeRangeBearingFactor* e = dynamic_cast<const ConeRangeBearingFactor*>(&expected);
    return e != nullptr && Base::equals(*e, tol) && 
           std::abs(range_ - e->range_) < tol &&
           std::abs(bearing_ - e->bearing_) < tol;
  }
};

} // namespace cone_stellation