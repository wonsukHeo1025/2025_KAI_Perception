#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <gtsam/geometry/Pose2.h>
#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>

#include "cone_stellation/common/cone.hpp"
#include "cone_stellation/common/estimation_frame.hpp"

namespace cone_stellation {

/**
 * @brief Loop closure candidate with validation score
 */
struct LoopCandidate {
  int query_frame_id;      // Current frame ID
  int reference_frame_id;  // Potential loop closure frame ID
  double score;            // Similarity score (lower is better)
  gtsam::Pose2 relative_pose;  // Estimated relative pose
  std::vector<std::pair<int, int>> cone_matches;  // Matched cone IDs
  
  bool operator<(const LoopCandidate& other) const {
    return score > other.score;  // For priority queue (min-heap)
  }
};

/**
 * @brief Path segment for trajectory-based loop detection
 */
struct PathSegment {
  std::vector<gtsam::Pose2> poses;      // Recent poses leading to this keyframe
  double total_length;                   // Total path length
  double avg_curvature;                  // Average curvature
  std::vector<double> curvature_profile; // Curvature at each pose
};

/**
 * @brief Geometric features for sparse landmark environments
 */
struct GeometricFeature {
  enum Type {
    STRAIGHT,           // Straight section
    TURN_LEFT,          // Left turn
    TURN_RIGHT,         // Right turn
    STRAIGHT_TO_TURN,   // Transition from straight to turn
    TURN_TO_STRAIGHT,   // Transition from turn to straight
    CHICANE,            // S-curve
    HAIRPIN             // Sharp turn > 90 degrees
  };
  
  Type type;
  double angle_change;     // Total angle change in radians
  double segment_length;   // Length of the feature
  Eigen::Vector2d entry_direction;  // Direction vector at entry
  Eigen::Vector2d exit_direction;   // Direction vector at exit
};

/**
 * @brief Enhanced constellation descriptor for sparse environments
 * 
 * Combines cone arrangements, odometry path, and geometric features
 * for robust place recognition in sparse landmark environments.
 */
struct ConstellationDescriptor {
  struct ConeInfo {
    Eigen::Vector2d relative_position;  // Position relative to constellation center
    ConeColor color;
    double distance_to_center;
    double angle_from_north;  // Angle in local constellation frame
  };
  
  int frame_id;
  Eigen::Vector2d center;  // Center of constellation in world frame
  std::vector<ConeInfo> cones;
  Eigen::Matrix2d covariance;  // Spatial spread of constellation
  
  // Histogram-based features for fast matching
  std::vector<double> distance_histogram;  // Distances between cone pairs
  std::vector<double> angle_histogram;     // Angles in cone triplets
  std::array<int, 4> color_counts;        // Count per color type
  
  // Path-based features for sparse environments
  PathSegment path_segment;              // Odometry path leading to this keyframe
  std::vector<GeometricFeature> geometric_features; // Detected geometric patterns
  
  /**
   * @brief Compute similarity to another descriptor
   * @return Distance metric (lower is more similar)
   */
  double distance_to(const ConstellationDescriptor& other) const;
  
  /**
   * @brief Check if descriptors could potentially match
   * Fast rejection based on color counts and size
   */
  bool is_compatible_with(const ConstellationDescriptor& other) const;
  
  /**
   * @brief Compute path similarity using curvature profile
   */
  double path_similarity(const ConstellationDescriptor& other) const;
  
  /**
   * @brief Check if geometric features match
   */
  bool geometric_features_match(const ConstellationDescriptor& other) const;
};

/**
 * @brief Loop closure detector using cone constellations
 * 
 * Inspired by GLIM's loop detection but adapted for sparse cone observations.
 * Uses geometric arrangements of cones (constellations) as place descriptors.
 */
class LoopClosureDetector {
public:
  using Ptr = std::shared_ptr<LoopClosureDetector>;
  
  struct Config {
    // Descriptor parameters
    double max_constellation_radius;  // Maximum radius for constellation
    int min_cones_per_constellation;     // Minimum cones to form descriptor
    int max_cones_per_constellation;    // Maximum cones (for efficiency)
    
    // Histogram parameters
    int distance_histogram_bins;
    double max_inter_cone_distance;
    int angle_histogram_bins;  // 30-degree bins
    
    // Path segment parameters
    int path_segment_size;  // Number of poses to store per segment
    double curvature_threshold;  // Threshold for detecting turns (rad/m)
    double straight_threshold;   // Threshold for straight sections
    
    // Geometric feature parameters
    double min_feature_length;  // Minimum length to detect a feature
    double turn_angle_threshold;  // Minimum angle change for turn
    double hairpin_angle_threshold;  // Angle for hairpin detection
    
    // Matching parameters
    int min_keyframes_apart;  // Temporal constraint
    double max_distance_for_loop;  // Spatial constraint
    double descriptor_match_threshold;  // Descriptor similarity threshold
    double path_match_weight;  // Weight for path similarity (0-1)
    double geometric_feature_weight;  // Weight for geometric features (0-1)
    
    // Validation parameters
    int min_matched_cones;  // For RANSAC validation
    double ransac_inlier_threshold;  // meters
    int ransac_iterations;
    double geometric_consistency_threshold;  // Final validation
    
    // Performance parameters
    int max_candidates_per_query;  // Limit candidates for efficiency
    bool use_parallel_matching;   // Use TBB for parallel processing
    
    // Constructor with default values
    Config() :
      max_constellation_radius(10.0),
      min_cones_per_constellation(3),  // Reduced for sparse environments
      max_cones_per_constellation(20),
      distance_histogram_bins(10),
      max_inter_cone_distance(15.0),
      angle_histogram_bins(12),
      path_segment_size(20),  // Store last 20 poses
      curvature_threshold(0.1),  // rad/m
      straight_threshold(0.02),  // rad/m
      min_feature_length(5.0),  // meters
      turn_angle_threshold(0.3),  // ~17 degrees
      hairpin_angle_threshold(1.57),  // 90 degrees
      min_keyframes_apart(20),
      max_distance_for_loop(5.0),
      descriptor_match_threshold(0.3),
      path_match_weight(0.3),  // 30% weight on path similarity
      geometric_feature_weight(0.4),  // 40% weight on geometric features
      min_matched_cones(3),  // Reduced for sparse
      ransac_inlier_threshold(0.5),
      ransac_iterations(100),
      geometric_consistency_threshold(0.8),
      max_candidates_per_query(10),
      use_parallel_matching(true) {}
  };
  
  LoopClosureDetector(const Config& config = Config()) 
    : config_(config), next_descriptor_id_(0) {
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "LoopClosureDetector initialized with constellation radius %.1f m",
                config_.max_constellation_radius);
  }
  
  /**
   * @brief Add a new keyframe for loop closure detection
   * @param frame The keyframe with cone observations
   * @param landmarks Current landmark positions (for constellation building)
   * @param recent_poses Recent pose history for path segment (optional)
   */
  void add_keyframe(const EstimationFrame::Ptr& frame,
                   const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
                   const std::vector<gtsam::Pose2>& recent_poses = {});
  
  /**
   * @brief Detect loop closure candidates for a query frame
   * @param query_frame The current frame to find loops for
   * @param landmarks Current landmark map
   * @return Validated loop closure candidates sorted by score
   */
  std::vector<LoopCandidate> detect_loop_closures(
      const EstimationFrame::Ptr& query_frame,
      const std::unordered_map<int, ConeLandmark::Ptr>& landmarks);
  
  /**
   * @brief Get descriptor for visualization/debugging
   */
  const ConstellationDescriptor* get_descriptor(int frame_id) const {
    auto it = frame_descriptors_.find(frame_id);
    return it != frame_descriptors_.end() ? &it->second : nullptr;
  }
  
  /**
   * @brief Clear old descriptors to manage memory
   * @param keep_recent_n Keep only the N most recent descriptors
   */
  void prune_old_descriptors(size_t keep_recent_n);

private:
  /**
   * @brief Build constellation descriptor from frame observations
   */
  ConstellationDescriptor build_descriptor(
      const EstimationFrame::Ptr& frame,
      const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
      const std::vector<gtsam::Pose2>& recent_poses = {});
  
  /**
   * @brief Extract geometric features for descriptor
   */
  void compute_geometric_features(ConstellationDescriptor& descriptor);
  
  /**
   * @brief Find potential loop candidates using descriptors
   */
  std::vector<int> find_candidates(const ConstellationDescriptor& query);
  
  /**
   * @brief Validate loop closure with geometric verification
   */
  bool validate_loop_closure(const EstimationFrame::Ptr& query_frame,
                           const EstimationFrame::Ptr& reference_frame,
                           const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
                           LoopCandidate& candidate);
  
  /**
   * @brief RANSAC-based relative pose estimation
   */
  bool estimate_relative_pose(const std::vector<Eigen::Vector2d>& query_cones,
                            const std::vector<Eigen::Vector2d>& reference_cones,
                            const std::vector<std::pair<int, int>>& matches,
                            gtsam::Pose2& relative_pose,
                            std::vector<int>& inliers);
  
  /**
   * @brief Compute 2D rigid transformation from point correspondences
   */
  gtsam::Pose2 compute_transform_svd(const std::vector<Eigen::Vector2d>& src,
                                   const std::vector<Eigen::Vector2d>& dst);
  
  /**
   * @brief Build path segment from recent poses
   */
  PathSegment build_path_segment(const std::vector<gtsam::Pose2>& poses);
  
  /**
   * @brief Detect geometric features from path segment
   */
  std::vector<GeometricFeature> detect_geometric_features(const PathSegment& path);
  
  /**
   * @brief Compute curvature at each pose
   */
  std::vector<double> compute_curvature_profile(const std::vector<gtsam::Pose2>& poses);
  
  /**
   * @brief Classify geometric feature type based on curvature
   */
  GeometricFeature::Type classify_feature(const std::vector<double>& curvatures,
                                         double total_angle_change);
  
  Config config_;
  
  // Storage
  std::unordered_map<int, ConstellationDescriptor> frame_descriptors_;
  std::unordered_map<int, EstimationFrame::Ptr> keyframes_;  // Keep references for validation
  std::vector<int> frame_id_sequence_;  // Maintain temporal order
  
  // Indexing for fast search (could be replaced with KD-tree)
  std::unordered_map<int, std::vector<int>> color_index_;  // Frames by dominant color
  
  int next_descriptor_id_;
  mutable std::mutex mutex_;  // Thread safety
};

} // namespace cone_stellation