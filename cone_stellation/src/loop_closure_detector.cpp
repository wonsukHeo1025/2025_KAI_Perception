#include "cone_stellation/mapping/loop_closure_detector.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <execution>  // For parallel algorithms

namespace cone_stellation {

// Implementation of LoopClosureDetector member functions

void LoopClosureDetector::add_keyframe(const EstimationFrame::Ptr& frame,
                                      const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
                                      const std::vector<gtsam::Pose2>& recent_poses) {
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[ADD_KEYFRAME] START for frame %d", frame ? frame->id : -1);
  
  if (!frame) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[ADD_KEYFRAME] Null frame pointer!");
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[ADD_KEYFRAME] Building descriptor...");
  // Build descriptor for this frame
  auto descriptor = build_descriptor(frame, landmarks, recent_poses);
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[ADD_KEYFRAME] Storing descriptor and frame...");
  // Store descriptor and frame
  frame_descriptors_[frame->id] = descriptor;
  keyframes_[frame->id] = frame;
  frame_id_sequence_.push_back(frame->id);
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[ADD_KEYFRAME] END - Added keyframe %d with %zu cones", 
              frame->id, descriptor.cones.size());
}

std::vector<LoopCandidate> LoopClosureDetector::detect_loop_closures(
    const EstimationFrame::Ptr& query_frame,
    const std::unordered_map<int, ConeLandmark::Ptr>& landmarks) {
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] START for frame %d", query_frame ? query_frame->id : -1);
  
  if (!query_frame) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[DETECT_LOOP] Null query frame!");
    return {};
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::vector<LoopCandidate> candidates;
  
  // Need at least min_keyframes_apart frames in database
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] Database has %zu frames (need %d)",
              frame_id_sequence_.size(), config_.min_keyframes_apart);
  
  if (frame_id_sequence_.size() < static_cast<size_t>(config_.min_keyframes_apart)) {
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[DETECT_LOOP] Not enough frames yet");
    return candidates;
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] Building query descriptor...");
  // Build query descriptor
  auto query_descriptor = build_descriptor(query_frame, landmarks);
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] Finding candidates...");
  // Find candidate frames
  auto candidate_ids = find_candidates(query_descriptor);
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] Found %zu potential candidates", candidate_ids.size());
  
  // Validate each candidate
  for (int ref_id : candidate_ids) {
    if (keyframes_.find(ref_id) == keyframes_.end()) continue;
    
    LoopCandidate candidate;
    candidate.query_frame_id = query_frame->id;
    candidate.reference_frame_id = ref_id;
    
    if (validate_loop_closure(query_frame, keyframes_[ref_id], landmarks, candidate)) {
      candidates.push_back(candidate);
    }
  }
  
  // Sort by score (lower is better)
  std::sort(candidates.begin(), candidates.end());
  
  // Keep only top candidates
  if (candidates.size() > static_cast<size_t>(config_.max_candidates_per_query)) {
    candidates.resize(config_.max_candidates_per_query);
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[DETECT_LOOP] END - Detected %zu validated candidates", 
              candidates.size());
  
  return candidates;
}

void LoopClosureDetector::prune_old_descriptors(size_t keep_recent_n) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (frame_id_sequence_.size() <= keep_recent_n) {
    return;
  }
  
  size_t to_remove = frame_id_sequence_.size() - keep_recent_n;
  for (size_t i = 0; i < to_remove; ++i) {
    int frame_id = frame_id_sequence_[i];
    frame_descriptors_.erase(frame_id);
    keyframes_.erase(frame_id);
  }
  
  frame_id_sequence_.erase(frame_id_sequence_.begin(), 
                          frame_id_sequence_.begin() + to_remove);
}

// Helper function to compute angle between three points
double compute_angle(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3) {
  Eigen::Vector2d v1 = p1 - p2;
  Eigen::Vector2d v2 = p3 - p2;
  
  // Check for zero-length vectors
  if (v1.norm() < 1e-6 || v2.norm() < 1e-6) {
    return 0.0;
  }
  
  v1.normalize();
  v2.normalize();
  double angle = std::acos(std::clamp(v1.dot(v2), -1.0, 1.0));
  return angle;
}

double ConstellationDescriptor::distance_to(const ConstellationDescriptor& other) const {
  RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
              "[DISTANCE_TO] Computing distance between frames %d and %d",
              frame_id, other.frame_id);
  
  // Quick rejection based on size difference
  if (std::abs(static_cast<int>(cones.size()) - static_cast<int>(other.cones.size())) > 5) {
    return std::numeric_limits<double>::max();
  }
  
  // Compare histograms using chi-squared distance
  double constellation_distance = 0.0;
  
  // Safety check histogram sizes
  if (distance_histogram.size() != other.distance_histogram.size() ||
      angle_histogram.size() != other.angle_histogram.size()) {
    RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                "[DISTANCE_TO] Histogram size mismatch!");
    return std::numeric_limits<double>::max();
  }
  
  // Distance histogram comparison
  for (size_t i = 0; i < distance_histogram.size(); ++i) {
    double diff = distance_histogram[i] - other.distance_histogram[i];
    double sum = distance_histogram[i] + other.distance_histogram[i];
    if (sum > 0) {
      constellation_distance += (diff * diff) / sum;
    }
  }
  
  // Angle histogram comparison
  for (size_t i = 0; i < angle_histogram.size(); ++i) {
    double diff = angle_histogram[i] - other.angle_histogram[i];
    double sum = angle_histogram[i] + other.angle_histogram[i];
    if (sum > 0) {
      constellation_distance += (diff * diff) / sum;
    }
  }
  
  // Color distribution comparison
  for (size_t i = 0; i < color_counts.size(); ++i) {
    double diff = color_counts[i] - other.color_counts[i];
    double sum = color_counts[i] + other.color_counts[i];
    if (sum > 0) {
      constellation_distance += (diff * diff) / sum;
    }
  }
  
  constellation_distance /= 3.0;  // Normalize
  
  // Path similarity (if available)
  double path_distance = 1.0;  // Default to no match
  if (!path_segment.poses.empty() && !other.path_segment.poses.empty()) {
    path_distance = 1.0 - path_similarity(other);
  }
  
  // Geometric feature matching
  double geometric_distance = 1.0;  // Default to no match
  if (!geometric_features.empty() && !other.geometric_features.empty()) {
    geometric_distance = geometric_features_match(other) ? 0.0 : 1.0;
  }
  
  // Weighted combination (constellation: 30%, path: 30%, geometric: 40%)
  return 0.3 * constellation_distance + 0.3 * path_distance + 0.4 * geometric_distance;
}

bool ConstellationDescriptor::is_compatible_with(const ConstellationDescriptor& other) const {
  // Check cone count similarity
  int size_diff = std::abs(static_cast<int>(cones.size()) - static_cast<int>(other.cones.size()));
  if (size_diff > 5) {
    return false;
  }
  
  // Check color distribution - at least 50% overlap
  int color_overlap = 0;
  int total_cones = 0;
  for (size_t i = 0; i < color_counts.size(); ++i) {
    color_overlap += std::min(color_counts[i], other.color_counts[i]);
    total_cones += color_counts[i];
  }
  
  if (total_cones > 0 && color_overlap < total_cones / 2) {
    return false;
  }
  
  return true;
}

double ConstellationDescriptor::path_similarity(const ConstellationDescriptor& other) const {
  if (path_segment.poses.empty() || other.path_segment.poses.empty()) {
    return 0.0;
  }
  
  // Compare path lengths
  double length_diff = std::abs(path_segment.total_length - other.path_segment.total_length);
  double length_similarity = 1.0 - std::min(length_diff / std::max(path_segment.total_length, other.path_segment.total_length), 1.0);
  
  // Compare average curvatures
  double curvature_diff = std::abs(path_segment.avg_curvature - other.path_segment.avg_curvature);
  double curvature_similarity = 1.0 - std::min(curvature_diff / 1.0, 1.0); // Normalize by max expected curvature
  
  // Compare curvature profiles using normalized cross-correlation
  double profile_similarity = 0.0;
  if (!path_segment.curvature_profile.empty() && !other.path_segment.curvature_profile.empty()) {
    size_t min_size = std::min(path_segment.curvature_profile.size(), other.path_segment.curvature_profile.size());
    
    double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    for (size_t i = 0; i < min_size; ++i) {
      double x = path_segment.curvature_profile[i];
      double y = other.path_segment.curvature_profile[i];
      sum_xy += x * y;
      sum_x2 += x * x;
      sum_y2 += y * y;
    }
    
    if (sum_x2 > 0 && sum_y2 > 0) {
      profile_similarity = sum_xy / (std::sqrt(sum_x2) * std::sqrt(sum_y2));
      profile_similarity = std::max(0.0, profile_similarity); // Ensure non-negative
    }
  }
  
  // Weighted combination
  return 0.3 * length_similarity + 0.3 * curvature_similarity + 0.4 * profile_similarity;
}

bool ConstellationDescriptor::geometric_features_match(const ConstellationDescriptor& other) const {
  if (geometric_features.empty() || other.geometric_features.empty()) {
    return false;
  }
  
  // Check if key features match
  int matched_features = 0;
  for (const auto& feature : geometric_features) {
    for (const auto& other_feature : other.geometric_features) {
      if (feature.type == other_feature.type) {
        // Check if angle changes are similar (within 20%)
        double angle_diff = std::abs(feature.angle_change - other_feature.angle_change);
        double angle_threshold = 0.2 * std::max(std::abs(feature.angle_change), std::abs(other_feature.angle_change));
        
        if (angle_diff <= angle_threshold) {
          matched_features++;
          break;
        }
      }
    }
  }
  
  // Require at least 50% of features to match
  return matched_features >= static_cast<int>(geometric_features.size() * 0.5);
}

ConstellationDescriptor LoopClosureDetector::build_descriptor(
    const EstimationFrame::Ptr& frame,
    const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
    const std::vector<gtsam::Pose2>& recent_poses) {
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] START for frame %d", frame ? frame->id : -1);
  
  ConstellationDescriptor descriptor;
  
  if (!frame) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[BUILD_DESC] Null frame!");
    return descriptor;
  }
  
  descriptor.frame_id = frame->id;
  
  // Initialize histograms
  descriptor.distance_histogram.resize(config_.distance_histogram_bins, 0);
  descriptor.angle_histogram.resize(config_.angle_histogram_bins, 0);
  descriptor.color_counts.fill(0);
  
  // Safety check
  if (!frame->cone_observations) {
    RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                "[BUILD_DESC] No cone observations in frame %d", frame->id);
    return descriptor;
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] Frame has %zu cone observations",
              frame->cone_observations->cones.size());
  
  // Collect visible landmarks
  std::vector<Eigen::Vector2d> cone_positions;
  std::vector<ConeColor> cone_colors;
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] observation_to_landmark map size: %zu",
              frame->observation_to_landmark.size());
  
  if (frame->cone_observations) {
    for (size_t i = 0; i < frame->cone_observations->cones.size(); ++i) {
      if (frame->observation_to_landmark.count(i) > 0) {
        int landmark_id = frame->observation_to_landmark.at(i);
        RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                    "[BUILD_DESC] Obs %zu -> Landmark %d", i, landmark_id);
        
        if (landmark_id >= 0 && landmarks.count(landmark_id) > 0) {
          const auto& landmark = landmarks.at(landmark_id);
          if (landmark) {
            cone_positions.push_back(landmark->position());
            cone_colors.push_back(landmark->color());
          } else {
            RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                       "[BUILD_DESC] Null landmark pointer for ID %d", landmark_id);
          }
        } else {
          RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                      "[BUILD_DESC] Landmark %d not found in map", landmark_id);
        }
      }
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] Collected %zu cone positions", cone_positions.size());
  
  if (cone_positions.empty()) {
    return descriptor;
  }
  
  // Compute center of constellation
  descriptor.center = Eigen::Vector2d::Zero();
  for (const auto& pos : cone_positions) {
    descriptor.center += pos;
  }
  descriptor.center /= cone_positions.size();
  
  // Build cone info relative to center
  for (size_t i = 0; i < cone_positions.size(); ++i) {
    ConstellationDescriptor::ConeInfo info;
    info.relative_position = cone_positions[i] - descriptor.center;
    info.color = cone_colors[i];
    info.distance_to_center = info.relative_position.norm();
    info.angle_from_north = std::atan2(info.relative_position.y(), 
                                      info.relative_position.x());
    
    // Only include cones within constellation radius
    if (info.distance_to_center <= config_.max_constellation_radius) {
      descriptor.cones.push_back(info);
      descriptor.color_counts[static_cast<int>(info.color)]++;
    }
  }
  
  // Limit constellation size for efficiency
  if (descriptor.cones.size() > config_.max_cones_per_constellation) {
    // Sort by distance and keep closest cones
    std::sort(descriptor.cones.begin(), descriptor.cones.end(),
              [](const auto& a, const auto& b) {
                return a.distance_to_center < b.distance_to_center;
              });
    descriptor.cones.resize(config_.max_cones_per_constellation);
    
    // Recompute color counts
    descriptor.color_counts.fill(0);
    for (const auto& cone : descriptor.cones) {
      descriptor.color_counts[static_cast<int>(cone.color)]++;
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] Computing geometric features...");
  // Compute geometric features
  try {
    compute_geometric_features(descriptor);
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_DESC] Geometric features computed successfully");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[BUILD_DESC] Failed to compute geometric features: %s", e.what());
  }
  
  // Build path segment if poses available
  if (!recent_poses.empty()) {
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_DESC] Building path segment from %zu poses...", recent_poses.size());
    try {
      descriptor.path_segment = build_path_segment(recent_poses);
      
      // TEMPORARILY DISABLED geometric features to isolate segfault
      // descriptor.geometric_features = detect_geometric_features(descriptor.path_segment);
      RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                  "[BUILD_DESC] Path segment built (geometric features disabled)");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                  "[BUILD_DESC] Failed to build path segment: %s", e.what());
    } catch (...) {
      RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                  "[BUILD_DESC] Failed to build path segment: unknown exception");
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_DESC] END - Descriptor built successfully");
  return descriptor;
}

void LoopClosureDetector::compute_geometric_features(ConstellationDescriptor& descriptor) {
  const auto& cones = descriptor.cones;
  
  if (cones.size() < 2) {
    return;
  }
  
  // Compute pairwise distances
  for (size_t i = 0; i < cones.size(); ++i) {
    for (size_t j = i + 1; j < cones.size(); ++j) {
      double dist = (cones[i].relative_position - cones[j].relative_position).norm();
      if (dist <= config_.max_inter_cone_distance) {
        int bin = std::min(static_cast<int>(dist / config_.max_inter_cone_distance * 
                                           config_.distance_histogram_bins),
                          config_.distance_histogram_bins - 1);
        descriptor.distance_histogram[bin]++;
      }
    }
  }
  
  // Normalize distance histogram
  double dist_sum = std::accumulate(descriptor.distance_histogram.begin(), 
                                   descriptor.distance_histogram.end(), 0.0);
  if (dist_sum > 0) {
    for (auto& val : descriptor.distance_histogram) {
      val /= dist_sum;
    }
  }
  
  // Compute angles in triplets
  if (cones.size() >= 3) {
    for (size_t i = 0; i < cones.size(); ++i) {
      for (size_t j = i + 1; j < cones.size(); ++j) {
        for (size_t k = j + 1; k < cones.size(); ++k) {
          // Angle at cone j
          double angle = compute_angle(cones[i].relative_position,
                                     cones[j].relative_position,
                                     cones[k].relative_position);
          int bin = std::min(static_cast<int>(angle / M_PI * config_.angle_histogram_bins),
                           config_.angle_histogram_bins - 1);
          descriptor.angle_histogram[bin]++;
        }
      }
    }
    
    // Normalize angle histogram
    double angle_sum = std::accumulate(descriptor.angle_histogram.begin(), 
                                     descriptor.angle_histogram.end(), 0.0);
    if (angle_sum > 0) {
      for (auto& val : descriptor.angle_histogram) {
        val /= angle_sum;
      }
    }
  }
  
  // Compute spatial covariance
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto& cone : cones) {
    cov += cone.relative_position * cone.relative_position.transpose();
  }
  descriptor.covariance = cov / cones.size();
}

std::vector<int> LoopClosureDetector::find_candidates(const ConstellationDescriptor& query) {
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[FIND_CANDIDATES] START");
  
  // NOTE: Mutex is already locked by the calling function (detect_loop_closures)
  // So we don't need to lock it again here
  
  std::vector<std::pair<int, double>> scored_candidates;
  
  // Get current frame position in sequence
  int query_seq_pos = frame_id_sequence_.size();
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[FIND_CANDIDATES] Checking %zu stored descriptors, query_seq_pos=%d",
              frame_descriptors_.size(), query_seq_pos);
  
  // Check all stored descriptors
  for (const auto& [frame_id, descriptor] : frame_descriptors_) {
    // Skip recent frames
    auto it = std::find(frame_id_sequence_.begin(), frame_id_sequence_.end(), frame_id);
    int ref_seq_pos = std::distance(frame_id_sequence_.begin(), it);
    
    RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                "[FIND_CANDIDATES] Checking frame %d (seq_pos %d)",
                frame_id, ref_seq_pos);
    
    if (query_seq_pos - ref_seq_pos < config_.min_keyframes_apart) {
      RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                  "[FIND_CANDIDATES] Skip - too recent");
      continue;
    }
    
    // Quick compatibility check
    if (!query.is_compatible_with(descriptor)) {
      RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                  "[FIND_CANDIDATES] Skip - not compatible");
      continue;
    }
    
    // Spatial constraint - check if we're near a previously visited location
    double spatial_distance = (query.center - descriptor.center).norm();
    
    // More aggressive loop closure detection for sparse environments
    if (spatial_distance > config_.max_distance_for_loop * 2.0) {
      RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                  "[FIND_CANDIDATES] Skip - too far (%.2f > %.2f)",
                  spatial_distance, config_.max_distance_for_loop * 2.0);
      continue;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[FIND_CANDIDATES] Computing distance for frame %d", frame_id);
    
    // Compute descriptor distance
    double desc_distance = query.distance_to(descriptor);
    
    // Apply spatial bonus for close locations
    if (spatial_distance < config_.max_distance_for_loop) {
      double spatial_bonus = 0.2 * (1.0 - spatial_distance / config_.max_distance_for_loop);
      desc_distance -= spatial_bonus; // Lower is better
    }
    
    // More lenient threshold for sparse environments
    if (desc_distance < config_.descriptor_match_threshold * 1.5) {
      scored_candidates.emplace_back(frame_id, desc_distance);
      RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                  "[FIND_CANDIDATES] Candidate frame %d: spatial_dist=%.2f, desc_dist=%.3f",
                  frame_id, spatial_distance, desc_distance);
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[FIND_CANDIDATES] Found %zu scored candidates", scored_candidates.size());
  
  // Sort by score and return top candidates
  std::sort(scored_candidates.begin(), scored_candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  
  std::vector<int> result;
  for (size_t i = 0; i < std::min(scored_candidates.size(), 
                                 static_cast<size_t>(config_.max_candidates_per_query)); ++i) {
    result.push_back(scored_candidates[i].first);
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[FIND_CANDIDATES] END - Returning %zu candidates", result.size());
  
  return result;
}

bool LoopClosureDetector::validate_loop_closure(
    const EstimationFrame::Ptr& query_frame,
    const EstimationFrame::Ptr& reference_frame,
    const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
    LoopCandidate& candidate) {
  
  // Safety checks
  if (!query_frame || !reference_frame) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "Null frame in validate_loop_closure");
    return false;
  }
  
  if (!query_frame->cone_observations || !reference_frame->cone_observations) {
    RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                "No cone observations in frames");
    return false;
  }
  
  // Collect cone positions in both frames
  std::vector<Eigen::Vector2d> query_cones;
  std::vector<Eigen::Vector2d> reference_cones;
  std::vector<int> query_landmark_ids;
  std::vector<int> reference_landmark_ids;
  
  // Get query frame cones
  if (query_frame->cone_observations) {
    for (size_t i = 0; i < query_frame->cone_observations->cones.size(); ++i) {
      if (query_frame->observation_to_landmark.count(i) > 0) {
        int landmark_id = query_frame->observation_to_landmark.at(i);
        if (landmark_id >= 0 && landmarks.count(landmark_id) > 0) {
          query_cones.push_back(landmarks.at(landmark_id)->position());
          query_landmark_ids.push_back(landmark_id);
        }
      }
    }
  }
  
  // Get reference frame cones
  if (reference_frame->cone_observations) {
    for (size_t i = 0; i < reference_frame->cone_observations->cones.size(); ++i) {
      if (reference_frame->observation_to_landmark.count(i) > 0) {
        int landmark_id = reference_frame->observation_to_landmark.at(i);
        if (landmark_id >= 0 && landmarks.count(landmark_id) > 0) {
          reference_cones.push_back(landmarks.at(landmark_id)->position());
          reference_landmark_ids.push_back(landmark_id);
        }
      }
    }
  }
  
  // Find matches based on appearance and geometry
  std::vector<std::pair<int, int>> tentative_matches;
  
  for (size_t i = 0; i < query_cones.size(); ++i) {
    for (size_t j = 0; j < reference_cones.size(); ++j) {
      // Check if they could be the same cone
      if (landmarks.at(query_landmark_ids[i])->color() == 
          landmarks.at(reference_landmark_ids[j])->color() ||
          landmarks.at(query_landmark_ids[i])->color() == ConeColor::UNKNOWN ||
          landmarks.at(reference_landmark_ids[j])->color() == ConeColor::UNKNOWN) {
        tentative_matches.emplace_back(i, j);
      }
    }
  }
  
  if (tentative_matches.size() < config_.min_matched_cones) {
    RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                "Too few tentative matches: %zu < %d",
                tentative_matches.size(), config_.min_matched_cones);
    return false;
  }
  
  // RANSAC to find best transformation
  std::vector<int> best_inliers;
  gtsam::Pose2 best_transform;
  
  if (!estimate_relative_pose(query_cones, reference_cones, tentative_matches,
                            best_transform, best_inliers)) {
    RCLCPP_DEBUG(rclcpp::get_logger("loop_closure"), 
                "RANSAC failed to find valid transformation");
    return false;
  }
  
  // Fill in candidate information
  candidate.relative_pose = best_transform;
  candidate.score = 1.0 - (double)best_inliers.size() / tentative_matches.size();
  
  for (int idx : best_inliers) {
    const auto& match = tentative_matches[idx];
    candidate.cone_matches.emplace_back(query_landmark_ids[match.first],
                                      reference_landmark_ids[match.second]);
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "Loop closure validated with %zu/%zu inliers",
              best_inliers.size(), tentative_matches.size());
  
  return true;
}

bool LoopClosureDetector::estimate_relative_pose(
    const std::vector<Eigen::Vector2d>& query_cones,
    const std::vector<Eigen::Vector2d>& reference_cones,
    const std::vector<std::pair<int, int>>& matches,
    gtsam::Pose2& relative_pose,
    std::vector<int>& inliers) {
  
  if (matches.size() < 3) {
    return false;
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, matches.size() - 1);
  
  int best_inlier_count = 0;
  gtsam::Pose2 best_pose;
  std::vector<int> best_inlier_indices;
  
  for (int iter = 0; iter < config_.ransac_iterations; ++iter) {
    // Sample 3 matches
    std::vector<int> sample_indices;
    while (sample_indices.size() < 3) {
      int idx = dis(gen);
      if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
        sample_indices.push_back(idx);
      }
    }
    
    // Get corresponding points
    std::vector<Eigen::Vector2d> src_points, dst_points;
    for (int idx : sample_indices) {
      const auto& match = matches[idx];
      src_points.push_back(query_cones[match.first]);
      dst_points.push_back(reference_cones[match.second]);
    }
    
    // Compute transformation
    gtsam::Pose2 transform = compute_transform_svd(src_points, dst_points);
    
    // Count inliers
    std::vector<int> current_inliers;
    for (size_t i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];
      Eigen::Vector2d transformed = transform.transformFrom(
          gtsam::Point2(query_cones[match.first]));
      double error = (transformed - reference_cones[match.second]).norm();
      
      if (error < config_.ransac_inlier_threshold) {
        current_inliers.push_back(i);
      }
    }
    
    if (current_inliers.size() > best_inlier_count) {
      best_inlier_count = current_inliers.size();
      best_pose = transform;
      best_inlier_indices = current_inliers;
    }
    
    // Early termination if we have enough inliers
    if (best_inlier_count > matches.size() * 0.8) {
      break;
    }
  }
  
  if (best_inlier_count < config_.min_matched_cones) {
    return false;
  }
  
  // Refine with all inliers
  std::vector<Eigen::Vector2d> src_inliers, dst_inliers;
  for (int idx : best_inlier_indices) {
    const auto& match = matches[idx];
    src_inliers.push_back(query_cones[match.first]);
    dst_inliers.push_back(reference_cones[match.second]);
  }
  
  relative_pose = compute_transform_svd(src_inliers, dst_inliers);
  inliers = best_inlier_indices;
  
  return true;
}

gtsam::Pose2 LoopClosureDetector::compute_transform_svd(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& dst) {
  
  if (src.size() != dst.size() || src.size() < 2) {
    return gtsam::Pose2();
  }
  
  // Compute centroids
  Eigen::Vector2d src_centroid = Eigen::Vector2d::Zero();
  Eigen::Vector2d dst_centroid = Eigen::Vector2d::Zero();
  
  for (size_t i = 0; i < src.size(); ++i) {
    src_centroid += src[i];
    dst_centroid += dst[i];
  }
  src_centroid /= src.size();
  dst_centroid /= dst.size();
  
  // Build covariance matrix
  Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    H += (src[i] - src_centroid) * (dst[i] - dst_centroid).transpose();
  }
  
  // SVD
  Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
  
  // Ensure proper rotation (det(R) = 1)
  if (R.determinant() < 0) {
    Eigen::Matrix2d V = svd.matrixV();
    V.col(1) *= -1;
    R = V * svd.matrixU().transpose();
  }
  
  // Compute translation
  Eigen::Vector2d t = dst_centroid - R * src_centroid;
  
  // Convert to gtsam::Pose2
  double theta = std::atan2(R(1, 0), R(0, 0));
  return gtsam::Pose2(t.x(), t.y(), theta);
}

PathSegment LoopClosureDetector::build_path_segment(const std::vector<gtsam::Pose2>& poses) {
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_PATH_SEGMENT] START with %zu poses", poses.size());
  
  PathSegment segment;
  
  if (poses.size() < 3) {
    RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Too few poses (%zu < 3)", poses.size());
    return segment;
  }
  
  try {
    // Store poses
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Storing poses...");
    segment.poses = poses;
    
    // Compute total length
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Computing total length...");
    segment.total_length = 0.0;
    for (size_t i = 1; i < poses.size(); ++i) {
      gtsam::Point2 p1 = poses[i-1].translation();
      gtsam::Point2 p2 = poses[i].translation();
      double dist = (p2 - p1).norm();
      segment.total_length += dist;
      
      if (std::isnan(dist) || std::isinf(dist)) {
        RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                    "[BUILD_PATH_SEGMENT] Invalid distance at i=%zu: %.3f", i, dist);
        return PathSegment();
      }
    }
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Total length: %.2f", segment.total_length);
    
    // Compute curvature profile
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Computing curvature profile...");
    segment.curvature_profile = compute_curvature_profile(poses);
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Curvature profile size: %zu", 
                segment.curvature_profile.size());
    
    // Compute average curvature
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Computing average curvature...");
    segment.avg_curvature = 0.0;
    if (!segment.curvature_profile.empty()) {
      for (double c : segment.curvature_profile) {
        segment.avg_curvature += std::abs(c);
      }
      segment.avg_curvature /= segment.curvature_profile.size();
    }
    RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Average curvature: %.3f", segment.avg_curvature);
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Exception: %s", e.what());
    return PathSegment();
  } catch (...) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[BUILD_PATH_SEGMENT] Unknown exception");
    return PathSegment();
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[BUILD_PATH_SEGMENT] END - Success");
  return segment;
}

std::vector<double> LoopClosureDetector::compute_curvature_profile(const std::vector<gtsam::Pose2>& poses) {
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[COMPUTE_CURVATURE] START with %zu poses", poses.size());
  
  std::vector<double> curvatures;
  
  if (poses.size() < 3) {
    RCLCPP_WARN(rclcpp::get_logger("loop_closure"), 
                "[COMPUTE_CURVATURE] Too few poses");
    return curvatures;
  }
  
  try {
    // Reserve space to avoid reallocation
    curvatures.reserve(poses.size() - 2);
    
    // Compute curvature at each intermediate pose
    for (size_t i = 1; i < poses.size() - 1; ++i) {
      // Get three consecutive positions
      gtsam::Point2 p0 = poses[i-1].translation();
      gtsam::Point2 p1 = poses[i].translation();
      gtsam::Point2 p2 = poses[i+1].translation();
      
      // Compute vectors
      Eigen::Vector2d diff1 = p1 - p0;
      Eigen::Vector2d diff2 = p2 - p1;
      
      // Skip if segments are too short
      if (diff1.norm() < 0.001 || diff2.norm() < 0.001) {
        curvatures.push_back(0.0);
        continue;
      }
      
      Eigen::Vector2d v1 = diff1.normalized();
      Eigen::Vector2d v2 = diff2.normalized();
      
      // Compute angle change
      double cross = v1.x() * v2.y() - v1.y() * v2.x();
      double dot = v1.dot(v2);
      double angle = std::atan2(cross, dot);
      
      // Compute distance
      double dist = 0.5 * (diff1.norm() + diff2.norm());
      
      // Curvature = angle change / distance
      double curvature = (dist > 0.01) ? angle / dist : 0.0;
      
      if (std::isnan(curvature) || std::isinf(curvature)) {
        RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                    "[COMPUTE_CURVATURE] Invalid curvature at i=%zu", i);
        curvatures.push_back(0.0);
      } else {
        curvatures.push_back(curvature);
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("loop_closure"), 
                "[COMPUTE_CURVATURE] Exception: %s", e.what());
    return std::vector<double>();
  }
  
  RCLCPP_INFO(rclcpp::get_logger("loop_closure"), 
              "[COMPUTE_CURVATURE] END - Computed %zu curvatures", curvatures.size());
  return curvatures;
}

std::vector<GeometricFeature> LoopClosureDetector::detect_geometric_features(const PathSegment& path) {
  std::vector<GeometricFeature> features;
  
  if (path.poses.size() < 3 || path.curvature_profile.empty()) {
    return features;
  }
  
  // Sliding window to detect features
  size_t window_size = std::min(size_t(5), path.curvature_profile.size());
  
  // Check if we have enough data for window
  if (window_size < 2 || path.curvature_profile.size() < window_size) {
    return features;
  }
  
  for (size_t i = 0; i + window_size <= path.curvature_profile.size(); ++i) {
    GeometricFeature feature;
    feature.segment_length = 0.0;
    feature.angle_change = 0.0;
    
    // Compute feature properties over window
    double max_curvature = 0.0;
    double min_curvature = 0.0;
    double sum_curvature = 0.0;
    
    for (size_t j = i; j < i + window_size; ++j) {
      double c = path.curvature_profile[j];
      sum_curvature += c;
      max_curvature = std::max(max_curvature, c);
      min_curvature = std::min(min_curvature, c);
      
      // The curvature profile has size poses.size()-2, so we need to adjust index
      size_t pose_idx = j + 1; // curvature[j] corresponds to pose[j+1]
      if (pose_idx + 1 < path.poses.size()) {
        gtsam::Point2 p1 = path.poses[pose_idx].translation();
        gtsam::Point2 p2 = path.poses[pose_idx + 1].translation();
        feature.segment_length += (p2 - p1).norm();
      }
    }
    
    // Skip if segment too short
    if (feature.segment_length < config_.min_feature_length) {
      continue;
    }
    
    // Compute total angle change
    // Adjust indices since curvature profile is shorter than poses
    size_t start_pose_idx = i + 1;
    size_t end_pose_idx = std::min(i + window_size + 1, path.poses.size() - 1);
    
    if (start_pose_idx >= path.poses.size() || end_pose_idx >= path.poses.size() || 
        start_pose_idx >= end_pose_idx) {
      continue; // Skip if indices are out of bounds
    }
    
    gtsam::Pose2 start_pose = path.poses[start_pose_idx];
    gtsam::Pose2 end_pose = path.poses[end_pose_idx];
    feature.angle_change = end_pose.theta() - start_pose.theta();
    
    // Normalize angle to [-pi, pi]
    while (feature.angle_change > M_PI) feature.angle_change -= 2 * M_PI;
    while (feature.angle_change < -M_PI) feature.angle_change += 2 * M_PI;
    
    // Entry and exit directions
    feature.entry_direction = Eigen::Vector2d(std::cos(start_pose.theta()), std::sin(start_pose.theta()));
    feature.exit_direction = Eigen::Vector2d(std::cos(end_pose.theta()), std::sin(end_pose.theta()));
    feature.type = classify_feature(std::vector<double>(
        path.curvature_profile.begin() + i,
        path.curvature_profile.begin() + i + window_size),
        feature.angle_change);
    
    // Only add significant features
    if (feature.type != GeometricFeature::STRAIGHT || 
        std::abs(feature.angle_change) > config_.turn_angle_threshold) {
      features.push_back(feature);
    }
  }
  
  return features;
}

GeometricFeature::Type LoopClosureDetector::classify_feature(
    const std::vector<double>& curvatures,
    double total_angle_change) {
  
  // Compute statistics
  double avg_curvature = 0.0;
  double max_abs_curvature = 0.0;
  int sign_changes = 0;
  int last_sign = 0;
  
  for (double c : curvatures) {
    avg_curvature += c;
    max_abs_curvature = std::max(max_abs_curvature, std::abs(c));
    
    int sign = (c > config_.straight_threshold) ? 1 : (c < -config_.straight_threshold) ? -1 : 0;
    if (sign != 0 && last_sign != 0 && sign != last_sign) {
      sign_changes++;
    }
    if (sign != 0) {
      last_sign = sign;
    }
  }
  avg_curvature /= curvatures.size();
  
  // Classify based on patterns
  if (std::abs(total_angle_change) > config_.hairpin_angle_threshold) {
    return GeometricFeature::HAIRPIN;
  }
  
  if (sign_changes > 0) {
    return GeometricFeature::CHICANE;
  }
  
  if (max_abs_curvature < config_.straight_threshold) {
    return GeometricFeature::STRAIGHT;
  }
  
  if (avg_curvature > config_.curvature_threshold) {
    // Check for transitions
    if (std::abs(curvatures.front()) < config_.straight_threshold) {
      return GeometricFeature::STRAIGHT_TO_TURN;
    } else if (std::abs(curvatures.back()) < config_.straight_threshold) {
      return GeometricFeature::TURN_TO_STRAIGHT;
    }
    return GeometricFeature::TURN_LEFT;
  }
  
  if (avg_curvature < -config_.curvature_threshold) {
    // Check for transitions
    if (std::abs(curvatures.front()) < config_.straight_threshold) {
      return GeometricFeature::STRAIGHT_TO_TURN;
    } else if (std::abs(curvatures.back()) < config_.straight_threshold) {
      return GeometricFeature::TURN_TO_STRAIGHT;
    }
    return GeometricFeature::TURN_RIGHT;
  }
  
  return GeometricFeature::STRAIGHT;
}

} // namespace cone_stellation