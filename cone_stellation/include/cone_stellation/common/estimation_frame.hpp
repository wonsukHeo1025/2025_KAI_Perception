#pragma once

#include <memory>
#include <unordered_map>
#include <boost/any.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cone_stellation/common/cone.hpp"

namespace cone_stellation {

/**
 * @brief Frame containing all data for a single timestamp
 * Adapted from GLIM's EstimationFrame for cone-based SLAM
 */
class EstimationFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<EstimationFrame>;
  using ConstPtr = std::shared_ptr<const EstimationFrame>;
  
  EstimationFrame() : id(-1), timestamp(0.0) {
    T_world_sensor = Eigen::Isometry3d::Identity();
    T_sensor_base = Eigen::Isometry3d::Identity();
    v_world = Eigen::Vector3d::Zero();
  }
  
  // Core data
  int id;                                    // Frame ID
  double timestamp;                          // Timestamp in seconds
  
  // Poses
  Eigen::Isometry3d T_world_sensor;         // Sensor pose in world frame
  Eigen::Isometry3d T_sensor_base;          // Sensor to base_link transform
  
  // Motion state
  Eigen::Vector3d v_world;                  // Velocity in world frame
  Eigen::Vector3d imu_bias_acc;             // IMU accelerometer bias
  Eigen::Vector3d imu_bias_gyro;            // IMU gyroscope bias
  
  // Cone observations
  std::shared_ptr<ConeObservationSet> cone_observations;
  
  // Detected patterns in this frame
  std::vector<ConePattern> detected_patterns;
  
  // Data association results
  std::unordered_map<int, int> observation_to_landmark; // local_id -> global_id
  
  // Get world pose of base_link
  Eigen::Isometry3d T_world_base() const {
    return T_world_sensor * T_sensor_base;
  }
  
  // Transform cone observation to world frame
  Eigen::Vector2d transform_to_world(const ConeObservation& obs) const {
    Eigen::Vector3d pos_sensor(obs.position.x(), obs.position.y(), 0.0);
    Eigen::Vector3d pos_world = T_world_sensor * pos_sensor;
    return pos_world.head<2>();
  }
  
  // Check if this is a keyframe
  bool is_keyframe = false;
  
  // Custom data storage (following GLIM pattern)
  template<typename T>
  void add_custom_data(const std::string& key, const T& data) {
    custom_data[key] = data;
  }
  
  template<typename T>
  T* get_custom_data(const std::string& key) {
    auto found = custom_data.find(key);
    if(found == custom_data.end()) {
      return nullptr;
    }
    return boost::any_cast<T>(&found->second);
  }
  
  template<typename T>
  const T* get_custom_data(const std::string& key) const {
    auto found = custom_data.find(key);
    if(found == custom_data.end()) {
      return nullptr;
    }
    return boost::any_cast<T>(&found->second);
  }

private:
  std::unordered_map<std::string, boost::any> custom_data;
};

/**
 * @brief Submap containing multiple frames
 * Used for hierarchical mapping like GLIM
 */
class SubMap {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<SubMap>;
  using ConstPtr = std::shared_ptr<const SubMap>;
  
  SubMap() : id(-1) {
    T_world_origin = Eigen::Isometry3d::Identity();
  }
  
  int id;                                    // Submap ID
  Eigen::Isometry3d T_world_origin;         // Origin pose of this submap
  
  std::vector<EstimationFrame::Ptr> frames; // Frames in this submap
  std::unordered_map<int, ConeLandmark::Ptr> local_landmarks; // Local cone map
  
  // Get all cone observations in this submap
  std::vector<ConeObservationSet> get_all_observations() const {
    std::vector<ConeObservationSet> all_obs;
    for(const auto& frame : frames) {
      if(frame->cone_observations) {
        all_obs.push_back(*frame->cone_observations);
      }
    }
    return all_obs;
  }
  
  // Merge landmarks from frames
  void build_local_map() {
    local_landmarks.clear();
    // Implementation depends on data association strategy
  }
};

} // namespace cone_stellation