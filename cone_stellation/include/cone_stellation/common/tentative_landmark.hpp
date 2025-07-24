#pragma once

#include <deque>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "cone_stellation/common/cone.hpp"

namespace cone_stellation {

/**
 * @brief Observation record for tentative landmark
 */
struct LandmarkObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Eigen::Vector2d world_position;     // Position in world frame
  Eigen::Vector2d sensor_position;    // Position in sensor frame  
  ConeColor color;
  int track_id;
  double timestamp;
  double confidence;
  int frame_id;                       // Which frame this observation came from
};

/**
 * @brief Tentative landmark that accumulates observations before being added to map
 */
class TentativeLandmark {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<TentativeLandmark>;
  
  TentativeLandmark(int id) : id_(id), primary_track_id_(-1) {
    position_sum_ = Eigen::Vector2d::Zero();
    position_squared_sum_ = Eigen::Vector2d::Zero();
  }
  
  /**
   * @brief Add a new observation to this tentative landmark
   */
  void add_observation(const LandmarkObservation& obs) {
    observations_.push_back(obs);
    
    // Update position statistics
    position_sum_ += obs.world_position;
    position_squared_sum_ += obs.world_position.cwiseProduct(obs.world_position);
    
    // Update color voting
    color_votes_[obs.color]++;
    
    // Update track ID scores
    track_id_scores_[obs.track_id] += 1.0;
    
    // Decay old track ID scores
    for (auto& [id, score] : track_id_scores_) {
      if (id != obs.track_id) {
        score *= 0.95;  // Decay factor
      }
    }
    
    // Update primary track ID if needed
    update_primary_track_id();
    
    // Limit buffer size
    while (observations_.size() > max_observations_) {
      const auto& old_obs = observations_.front();
      position_sum_ -= old_obs.world_position;
      position_squared_sum_ -= old_obs.world_position.cwiseProduct(old_obs.world_position);
      color_votes_[old_obs.color]--;
      observations_.pop_front();
    }
  }
  
  /**
   * @brief Get the mean position of all observations
   */
  Eigen::Vector2d get_mean_position() const {
    if (observations_.empty()) return Eigen::Vector2d::Zero();
    return position_sum_ / observations_.size();
  }
  
  /**
   * @brief Get position covariance
   */
  Eigen::Matrix2d get_position_covariance() const {
    if (observations_.size() < 2) return Eigen::Matrix2d::Identity() * 1e6;
    
    Eigen::Vector2d mean = get_mean_position();
    Eigen::Vector2d var = (position_squared_sum_ / observations_.size()) - mean.cwiseProduct(mean);
    return var.asDiagonal();
  }
  
  /**
   * @brief Get the most voted color
   */
  ConeColor get_primary_color() const {
    ConeColor best_color = ConeColor::UNKNOWN;
    int max_votes = 0;
    
    for (const auto& [color, votes] : color_votes_) {
      if (votes > max_votes) {
        max_votes = votes;
        best_color = color;
      }
    }
    
    return best_color;
  }
  
  /**
   * @brief Get color confidence (ratio of primary color votes)
   */
  double get_color_confidence() const {
    if (observations_.empty()) return 0.0;
    
    int total_votes = 0;
    int max_votes = 0;
    
    for (const auto& [color, votes] : color_votes_) {
      total_votes += votes;
      max_votes = std::max(max_votes, votes);
    }
    
    return total_votes > 0 ? static_cast<double>(max_votes) / total_votes : 0.0;
  }
  
  /**
   * @brief Check if this tentative landmark is ready to be promoted
   */
  bool is_ready_for_promotion() const {
    // Criteria 1: Minimum number of observations
    if (observations_.size() < min_observations_) return false;
    
    // Criteria 2: Time span
    if (!observations_.empty()) {
      double time_span = observations_.back().timestamp - observations_.front().timestamp;
      if (time_span < min_time_span_) return false;
    }
    
    // Criteria 3: Position covariance
    Eigen::Matrix2d cov = get_position_covariance();
    double max_variance = std::max(cov(0,0), cov(1,1));
    if (max_variance > max_position_variance_) return false;
    
    // Criteria 4: Color consistency
    if (get_color_confidence() < min_color_confidence_) return false;
    
    return true;
  }
  
  /**
   * @brief Get all observations
   */
  const std::deque<LandmarkObservation>& get_observations() const {
    return observations_;
  }
  
  /**
   * @brief Get unique frame IDs that observed this landmark
   */
  std::vector<int> get_observing_frames() const {
    std::set<int> unique_frames;
    for (const auto& obs : observations_) {
      unique_frames.insert(obs.frame_id);
    }
    return std::vector<int>(unique_frames.begin(), unique_frames.end());
  }
  
  int get_id() const { return id_; }
  int get_primary_track_id() const { return primary_track_id_; }
  size_t get_observation_count() const { return observations_.size(); }
  
  // Configuration parameters
  static size_t min_observations_;
  static double min_time_span_;
  static double max_position_variance_;
  static double min_color_confidence_;
  static size_t max_observations_;
  
private:
  void update_primary_track_id() {
    int best_id = -1;
    double best_score = 0.0;
    
    for (const auto& [id, score] : track_id_scores_) {
      if (score > best_score) {
        best_score = score;
        best_id = id;
      }
    }
    
    // Hysteresis: only change if new ID has significantly higher score
    if (best_id != primary_track_id_ && best_score > track_id_scores_[primary_track_id_] * 1.5) {
      primary_track_id_ = best_id;
    }
  }
  
  int id_;
  int primary_track_id_;
  std::deque<LandmarkObservation> observations_;
  std::unordered_map<ConeColor, int> color_votes_;
  std::unordered_map<int, double> track_id_scores_;
  
  // Running statistics for efficiency
  Eigen::Vector2d position_sum_;
  Eigen::Vector2d position_squared_sum_;
};

// Initialize static members (should be done in cpp file)
inline size_t TentativeLandmark::min_observations_ = 3;
inline double TentativeLandmark::min_time_span_ = 0.5;  // seconds
inline double TentativeLandmark::max_position_variance_ = 0.2;  // m^2 (stricter)
inline double TentativeLandmark::min_color_confidence_ = 0.8;  // (higher)
inline size_t TentativeLandmark::max_observations_ = 15;

} // namespace cone_stellation