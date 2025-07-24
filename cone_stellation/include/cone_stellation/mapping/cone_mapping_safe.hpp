#pragma once

// Safe version of promote_tentative_landmarks function
// This is a replacement for the original function in cone_mapping.hpp

/**
 * @brief Promote ready tentative landmarks to confirmed landmarks (SAFE VERSION)
 * 
 * Key changes:
 * 1. Don't add observation factors from old frames
 * 2. Only create the landmark and let future observations add factors
 * 3. Better error handling
 */
void promote_tentative_landmarks_safe() {
  std::vector<int> promoted_ids;
  
  for (const auto& [id, tentative] : tentative_landmarks_) {
    if (!tentative->is_ready_for_promotion()) {
      continue;
    }
    
    try {
      // Create confirmed landmark
      Eigen::Vector2d position = tentative->get_mean_position();
      ConeColor color = tentative->get_primary_color();
      
      int landmark_id = next_landmark_id_++;
      landmarks_[landmark_id] = std::make_shared<ConeLandmark>(landmark_id, position, color);
      landmarks_[landmark_id]->set_track_id(tentative->get_primary_track_id());
      
      // Add to GTSAM
      gtsam::Symbol landmark_key('l', landmark_id);
      initial_values_.insert(landmark_key, gtsam::Point2(position.x(), position.y()));
      
      // Add prior only for first few landmarks
      if (landmarks_.size() <= 3) {
        auto landmark_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            gtsam::Vector2(0.1, 0.1));  // Strong prior
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Point2>>(
            landmark_key, gtsam::Point2(position.x(), position.y()), landmark_prior_noise);
      }
      
      // Don't add observation factors from old frames - let future observations handle it
      // This avoids referencing frames that may have been marginalized
      
      promoted_ids.push_back(id);
      
      RCLCPP_INFO(rclcpp::get_logger("cone_mapping"), 
                  "Promoted tentative landmark %d to confirmed landmark %d (color: %d, track: %d, observations: %d)",
                  id, landmark_id, static_cast<int>(color), 
                  tentative->get_primary_track_id(),
                  tentative->get_observation_count());
                  
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("cone_mapping"), 
                   "Error promoting tentative landmark %d: %s", id, e.what());
    }
  }
  
  // Remove promoted tentative landmarks
  for (int id : promoted_ids) {
    tentative_landmarks_.erase(id);
  }
}