#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <set>

namespace cone_stellation {

/**
 * @brief Cone color enumeration matching Formula Student standards
 */
enum class ConeColor {
  UNKNOWN = 0,
  YELLOW = 1,   // Right side (outer track boundary, Y < 0)
  BLUE = 2,     // Left side (inner track boundary, Y > 0)
  ORANGE = 3,   // Small orange
  RED = 4       // Large red (start/finish)
};

/**
 * @brief Single cone observation from a specific pose
 */
struct ConeObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  int id;                         // Temporary ID within this observation
  Eigen::Vector2d position;       // Position in sensor frame
  Eigen::Matrix2d covariance;     // Measurement uncertainty
  ConeColor color;                // Detected color
  double confidence;              // Detection confidence [0,1]
  
  ConeObservation() : id(-1), position(Eigen::Vector2d::Zero()), 
                      covariance(Eigen::Matrix2d::Identity()), 
                      color(ConeColor::UNKNOWN), confidence(1.0) {}
};

/**
 * @brief Cone landmark in the global map
 */
class ConeLandmark {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<ConeLandmark>;
  
  ConeLandmark(int id, const Eigen::Vector2d& position, ConeColor color)
    : id_(id), position_(position), color_(color), 
      observations_(0), confidence_(1.0), primary_track_id_(-1) {}
  
  // Getters
  int id() const { return id_; }
  const Eigen::Vector2d& position() const { return position_; }
  ConeColor color() const { return color_; }
  double confidence() const { return confidence_; }
  int observations() const { return observations_; }
  const std::set<int>& co_observed_cones() const { return co_observed_cones_; }
  
  // Update methods
  void update_position(const Eigen::Vector2d& pos) { position_ = pos; }
  void increment_observations() { observations_++; }
  void update_confidence(double conf) { confidence_ = conf; }
  
  // Co-observation tracking
  void add_co_observed(int cone_id) { 
    co_observed_cones_.insert(cone_id);
    co_observation_counts_[cone_id]++;
  }
  void add_co_observed(const std::vector<int>& cone_ids) {
    for (int id : cone_ids) {
      add_co_observed(id);
    }
  }
  
  // Check if this cone has been observed together with another
  bool is_co_observed_with(int cone_id) const {
    return co_observed_cones_.find(cone_id) != co_observed_cones_.end();
  }
  
  // Calculate co-observation strength (number of times seen together)
  int co_observation_count(int cone_id) const {
    auto it = co_observation_counts_.find(cone_id);
    return (it != co_observation_counts_.end()) ? it->second : 0;
  }

  // Track ID management
  int track_id() const { return primary_track_id_; }
  void set_track_id(int id) { 
    if (id >= 0) {
      primary_track_id_ = id;
      track_id_scores_[id] = 1.0;  // Initialize score
    }
  }
  void update_track_id(int id, float score = 1.0) {
    if (id < 0) return;  // Ignore invalid track IDs
    
    track_id_scores_[id] += score;
    
    // Update primary track ID if this one has higher score
    if (primary_track_id_ < 0 || 
        track_id_scores_[id] > track_id_scores_[primary_track_id_]) {
      primary_track_id_ = id;
    }
  }
  const std::map<int, float>& track_id_history() const { return track_id_scores_; }

private:
  int id_;                        // Unique landmark ID
  Eigen::Vector2d position_;      // Position in map frame
  ConeColor color_;               // Cone color
  int observations_;              // Number of times observed
  double confidence_;             // Landmark confidence
  std::set<int> co_observed_cones_; // IDs of cones observed together
  std::map<int, int> co_observation_counts_; // Count of co-observations per cone
  int primary_track_id_;          // Primary track ID from sensor
  std::map<int, float> track_id_scores_; // Track ID observation history with scores
};

/**
 * @brief Detected geometric pattern from cone observations
 */
struct ConePattern {
  enum Type {
    NONE = 0,
    LINE = 1,      // Straight line of cones
    CURVE = 2,     // Curved section
    PARALLEL = 3,  // Parallel lines (track boundaries)
    CORNER = 4     // Corner pattern
  };
  
  Type type;
  std::vector<int> cone_ids;  // IDs of cones in this pattern
  Eigen::VectorXd parameters;  // Pattern-specific parameters
  double confidence;           // Pattern detection confidence
  
  ConePattern() : type(NONE), confidence(0.0) {}
};

/**
 * @brief Collection of cones observed from a single pose
 */
struct ConeObservationSet {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  std::vector<ConeObservation> cones;
  Eigen::Isometry3d sensor_pose;    // Pose where observations were made
  double timestamp;
  std::vector<ConePattern> detected_patterns; // Patterns detected in this observation
  
  // Get cone IDs for co-observation tracking
  std::vector<int> get_cone_ids() const {
    std::vector<int> ids;
    ids.reserve(cones.size());
    for (const auto& cone : cones) {
      if (cone.id >= 0) ids.push_back(cone.id);
    }
    return ids;
  }
  
  // Check if this set forms a valid pattern (minimum cones)
  bool has_valid_pattern(size_t min_cones = 2) const {
    return cones.size() >= min_cones;
  }
};

} // namespace cone_stellation