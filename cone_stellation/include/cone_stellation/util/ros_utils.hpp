#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cctype>

#include "cone_stellation/common/cone.hpp"
#include "custom_interface/msg/tracked_cone_array.hpp"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>

namespace cone_stellation {

/**
 * @brief Convert TrackedConeArray message to ConeObservationSet
 */
inline std::vector<ConeObservation> from_ros_msg(const custom_interface::msg::TrackedConeArray& msg) {
  std::vector<ConeObservation> observations;
  observations.reserve(msg.cones.size());
  
  for (const auto& cone_msg : msg.cones) {
    ConeObservation obs;
    obs.id = cone_msg.track_id;
    obs.position = Eigen::Vector2d(cone_msg.position.x, cone_msg.position.y);
    obs.confidence = 1.0; // Default confidence since not in message
    
    // Parse color from string (case-insensitive)
    std::string color_lower = cone_msg.color;
    std::transform(color_lower.begin(), color_lower.end(), color_lower.begin(), ::tolower);
    
    if (color_lower.find("yellow") != std::string::npos) {
      obs.color = ConeColor::YELLOW;
    } else if (color_lower.find("blue") != std::string::npos) {
      obs.color = ConeColor::BLUE;
    } else if (color_lower.find("red") != std::string::npos) {
      obs.color = ConeColor::RED;
    } else if (color_lower.find("orange") != std::string::npos) {
      obs.color = ConeColor::ORANGE;
    } else {
      obs.color = ConeColor::UNKNOWN;
      RCLCPP_DEBUG(rclcpp::get_logger("ros_utils"), 
                   "Unknown color string: '%s'", cone_msg.color.c_str());
    }
    
    // Simple covariance model based on distance
    double range = obs.position.norm();
    double sigma = 0.1 + 0.02 * range; // 10cm + 2% of range
    obs.covariance = Eigen::Matrix2d::Identity() * sigma * sigma;
    
    observations.push_back(obs);
  }
  
  return observations;
}

/**
 * @brief Create visualization markers for cones
 */
inline visualization_msgs::msg::MarkerArray create_cone_markers(
    const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
    const std::string& frame_id = "map",
    const rclcpp::Time& timestamp = rclcpp::Time()) {
  
  visualization_msgs::msg::MarkerArray markers;
  
  // Delete all marker
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.header.frame_id = frame_id;
  delete_marker.header.stamp = timestamp;
  delete_marker.ns = "cone_landmarks";
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(delete_marker);
  
  // Add cone markers
  for (const auto& [id, landmark] : landmarks) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "cone_landmarks";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position
    marker.pose.position.x = landmark->position().x();
    marker.pose.position.y = landmark->position().y();
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    // Scale
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.5;
    
    // Color based on cone type
    switch (landmark->color()) {
      case ConeColor::YELLOW:
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        break;
      case ConeColor::BLUE:
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        break;
      case ConeColor::RED:
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        break;
      case ConeColor::ORANGE:
        marker.color.r = 1.0;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        break;
      default:
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
    }
    marker.color.a = 0.8;
    
    markers.markers.push_back(marker);
    
    // Add text label
    visualization_msgs::msg::Marker text_marker = marker;
    text_marker.ns = "cone_ids";
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.pose.position.z = 0.7;
    text_marker.scale.x = 0.0;
    text_marker.scale.y = 0.0;
    text_marker.scale.z = 0.3;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    text_marker.text = std::to_string(id);
    
    markers.markers.push_back(text_marker);
  }
  
  return markers;
}

/**
 * @brief Create factor graph visualization
 */
inline visualization_msgs::msg::MarkerArray create_factor_markers(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& values,
    const std::string& frame_id = "map",
    const rclcpp::Time& timestamp = rclcpp::Time()) {
  
  visualization_msgs::msg::MarkerArray markers;
  
  // Delete all marker
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.header.frame_id = frame_id;
  delete_marker.header.stamp = timestamp;
  delete_marker.ns = "factors";
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(delete_marker);
  
  int marker_id = 0;
  
  // Iterate through factors
  for (const auto& factor : graph) {
    if (!factor) continue;
    
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = timestamp;
    line_marker.ns = "factors";
    line_marker.id = marker_id++;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Get factor keys to determine type
    const auto& keys = factor->keys();
    if (keys.size() < 2) continue;
    
    // Determine factor type and set color/width accordingly
    char type1 = gtsam::Symbol(keys[0]).chr();
    char type2 = gtsam::Symbol(keys[1]).chr();
    
    if (type1 == 'x' && type2 == 'x') {
      // Odometry factor (pose-to-pose) - thick green line
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 0.8;
      line_marker.scale.x = 0.05;
      line_marker.ns = "odometry_factors";
    } else if ((type1 == 'x' && type2 == 'l') || (type1 == 'l' && type2 == 'x')) {
      // Observation factor (pose-to-landmark) - thin blue line
      line_marker.color.r = 0.0;
      line_marker.color.g = 0.5;
      line_marker.color.b = 1.0;
      line_marker.color.a = 0.6;
      line_marker.scale.x = 0.02;
      line_marker.ns = "observation_factors";
    } else if (type1 == 'l' && type2 == 'l') {
      // Inter-landmark factor - red dashed line
      line_marker.color.r = 1.0;
      line_marker.color.g = 0.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 0.8;
      line_marker.scale.x = 0.03;
      line_marker.ns = "inter_landmark_factors";
    } else {
      // Other factors - gray
      line_marker.color.r = 0.5;
      line_marker.color.g = 0.5;
      line_marker.color.b = 0.5;
      line_marker.color.a = 0.5;
      line_marker.scale.x = 0.02;
    }
    
    // Create line between factor nodes
    if (keys.size() == 2) {
      // Binary factor
      try {
        geometry_msgs::msg::Point p1, p2;
        
        // Try to get positions
        if (values.exists(keys[0]) && values.exists(keys[1])) {
          // Handle both Pose2 and Point2
          if (gtsam::Symbol(keys[0]).chr() == 'x') {
            // Pose
            auto pose = values.at<gtsam::Pose2>(keys[0]);
            p1.x = pose.x();
            p1.y = pose.y();
            p1.z = 0.0;
          } else {
            // Landmark
            auto point = values.at<gtsam::Point2>(keys[0]);
            p1.x = point.x();
            p1.y = point.y();
            p1.z = 0.0;
          }
          
          if (gtsam::Symbol(keys[1]).chr() == 'x') {
            // Pose
            auto pose = values.at<gtsam::Pose2>(keys[1]);
            p2.x = pose.x();
            p2.y = pose.y();
            p2.z = 0.0;
          } else {
            // Landmark
            auto point = values.at<gtsam::Point2>(keys[1]);
            p2.x = point.x();
            p2.y = point.y();
            p2.z = 0.0;
          }
          
          line_marker.points.push_back(p1);
          line_marker.points.push_back(p2);
          markers.markers.push_back(line_marker);
        }
      } catch (...) {
        // Skip if values not available
      }
    }
  }
  
  return markers;
}

/**
 * @brief Create keyframe visualization markers
 */
inline visualization_msgs::msg::MarkerArray create_keyframe_markers(
    const std::unordered_map<int, gtsam::Pose2>& keyframe_poses,
    const std::string& frame_id = "map",
    const rclcpp::Time& timestamp = rclcpp::Time()) {
  
  visualization_msgs::msg::MarkerArray markers;
  
  // Delete all marker
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.header.frame_id = frame_id;
  delete_marker.header.stamp = timestamp;
  delete_marker.ns = "keyframes";
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(delete_marker);
  
  // Add keyframe markers
  for (const auto& [id, pose] : keyframe_poses) {
    // Create arrow marker for pose
    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = frame_id;
    arrow.header.stamp = timestamp;
    arrow.ns = "keyframes";
    arrow.id = id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    
    // Position
    arrow.pose.position.x = pose.x();
    arrow.pose.position.y = pose.y();
    arrow.pose.position.z = 0.1;  // Slightly above ground
    
    // Orientation from yaw angle
    tf2::Quaternion q;
    q.setRPY(0, 0, pose.theta());
    arrow.pose.orientation.x = q.x();
    arrow.pose.orientation.y = q.y();
    arrow.pose.orientation.z = q.z();
    arrow.pose.orientation.w = q.w();
    
    // Scale
    arrow.scale.x = 0.5;  // Arrow length
    arrow.scale.y = 0.1;  // Arrow width
    arrow.scale.z = 0.1;  // Arrow height
    
    // Color - cyan for keyframes
    arrow.color.r = 0.0;
    arrow.color.g = 1.0;
    arrow.color.b = 1.0;
    arrow.color.a = 0.8;
    
    markers.markers.push_back(arrow);
    
    // Add text label for keyframe ID
    visualization_msgs::msg::Marker text;
    text.header = arrow.header;
    text.ns = "keyframe_ids";
    text.id = id;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    
    text.pose.position.x = pose.x();
    text.pose.position.y = pose.y();
    text.pose.position.z = 0.8;  // Above arrow
    text.pose.orientation.w = 1.0;
    
    text.scale.z = 0.3;  // Text height
    
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.color.a = 1.0;
    
    text.text = "KF" + std::to_string(id);
    
    markers.markers.push_back(text);
  }
  
  return markers;
}

} // namespace cone_stellation