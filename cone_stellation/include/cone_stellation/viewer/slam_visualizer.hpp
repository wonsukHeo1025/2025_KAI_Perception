#pragma once

#include <memory>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "cone_stellation/viewer/viewer_base.hpp"
#include "cone_stellation/common/cone.hpp"
#include "cone_stellation/mapping/cone_mapping.hpp"

namespace cone_stellation {
namespace viewer {

/**
 * @brief SLAM-specific visualization for cone mapping results
 * 
 * Handles visualization of:
 * - Mapped cone landmarks with proper colors
 * - Factor graph edges (pose-pose, pose-landmark, inter-landmark)
 * - Keyframe poses
 * - Optimized trajectory
 */
class SLAMVisualizer : public ViewerBase {
public:
  using Ptr = std::shared_ptr<SLAMVisualizer>;
  
  SLAMVisualizer(rclcpp::Node* node) : node_(node) {
    setName("SLAMVisualizer");
  }
  
  bool initialize() override {
    // Create publishers with volatile QoS for real-time visualization
    rclcpp::QoS viz_qos(10);
    viz_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    viz_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    viz_qos.history(rclcpp::HistoryPolicy::KeepLast);
    
    landmark_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/slam/landmarks", viz_qos);
    factor_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/slam/factor_graph", viz_qos);
    keyframe_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/slam/keyframes", viz_qos);
    path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("/slam/path", viz_qos);
    
    initialized_ = true;
    return true;
  }
  
  void shutdown() override {
    clear();
    initialized_ = false;
  }
  
  bool isInitialized() const override { return initialized_; }
  
  void update() override {
    // Update is triggered by specific visualization calls
  }
  
  void clear() override {
    publishDeleteAll("/slam/landmarks");
    publishDeleteAll("/slam/factor_graph");
    publishDeleteAll("/slam/keyframes");
  }
  
  /**
   * @brief Visualize cone landmarks from mapping
   * @param landmarks Map of landmark ID to landmark pointer
   * @param timestamp Optional timestamp for markers (uses current time if not provided)
   */
  void visualizeLandmarks(const std::unordered_map<int, ConeLandmark::Ptr>& landmarks,
                         const rclcpp::Time& timestamp = rclcpp::Time()) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    visualization_msgs::msg::MarkerArray markers;
    
    // Use provided timestamp or current time
    rclcpp::Time marker_time = timestamp.nanoseconds() > 0 ? timestamp : node_->now();
    
    // Delete all marker
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = marker_time;
    delete_marker.ns = "cone_landmarks";
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_marker);
    
    // Add cone markers
    for (const auto& [id, landmark] : landmarks) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = marker_time;
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
      setMarkerColor(marker, landmark->color());
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
    
    landmark_pub_->publish(markers);
  }
  
  /**
   * @brief Visualize factor graph structure showing most recent factors
   * @param graph GTSAM factor graph
   * @param values Current estimates
   * @param timestamp Optional timestamp for markers
   */
  virtual void visualizeFactorGraph(const gtsam::NonlinearFactorGraph& graph, 
                           const gtsam::Values& values,
                           const rclcpp::Time& timestamp = rclcpp::Time()) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    visualization_msgs::msg::MarkerArray markers;
    
    // Limits for different factor types to prevent overload
    const size_t max_observation_factors = 100;   // Most numerous, limit heavily
    const size_t max_odometry_factors = 200;      // Show more odometry
    const size_t max_inter_landmark_factors = 50; // Limit these
    const size_t max_loop_closure_factors = 20;   // Show all loop closures if possible
    
    // Store factors by type with their indices for reverse iteration
    std::vector<std::pair<size_t, gtsam::NonlinearFactor::shared_ptr>> observation_factors;
    std::vector<std::pair<size_t, gtsam::NonlinearFactor::shared_ptr>> odometry_factors;
    std::vector<std::pair<size_t, gtsam::NonlinearFactor::shared_ptr>> inter_landmark_factors;
    std::vector<std::pair<size_t, gtsam::NonlinearFactor::shared_ptr>> loop_closure_factors;
    
    // First pass: categorize all factors
    size_t factor_index = 0;
    for (const auto& factor : graph) {
      if (!factor) {
        factor_index++;
        continue;
      }
      
      const auto& keys = factor->keys();
      if (keys.size() < 2) {
        factor_index++;
        continue;
      }
      
      char type1 = gtsam::Symbol(keys[0]).chr();
      char type2 = gtsam::Symbol(keys[1]).chr();
      
      if (type1 == 'x' && type2 == 'x') {
        // Check if this is a loop closure factor
        int id1 = gtsam::Symbol(keys[0]).index();
        int id2 = gtsam::Symbol(keys[1]).index();
        bool is_loop_closure = std::abs(id2 - id1) > 5;
        
        if (is_loop_closure) {
          loop_closure_factors.emplace_back(factor_index, factor);
        } else {
          odometry_factors.emplace_back(factor_index, factor);
        }
      } else if ((type1 == 'x' && type2 == 'l') || (type1 == 'l' && type2 == 'x')) {
        observation_factors.emplace_back(factor_index, factor);
      } else if (type1 == 'l' && type2 == 'l') {
        inter_landmark_factors.emplace_back(factor_index, factor);
      }
      
      factor_index++;
    }
    
    // Delete all markers periodically
    static auto last_delete_time = node_->now();
    if ((node_->now() - last_delete_time).seconds() > 30.0) {
      visualization_msgs::msg::Marker delete_marker;
      delete_marker.header.frame_id = "map";
      delete_marker.header.stamp = timestamp.nanoseconds() > 0 ? timestamp : node_->now();
      delete_marker.ns = "factors";
      delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
      markers.markers.push_back(delete_marker);
      last_delete_time = node_->now();
    }
    
    int marker_id = 0;
    
    // Lambda to visualize a single factor
    auto visualize_factor = [&](const gtsam::NonlinearFactor::shared_ptr& factor, 
                               const std::string& ns, 
                               double r, double g, double b, double a,
                               double scale, double lifetime) {
      visualization_msgs::msg::Marker line_marker;
      line_marker.header.frame_id = "map";
      line_marker.header.stamp = timestamp.nanoseconds() > 0 ? timestamp : node_->now();
      line_marker.ns = ns;
      line_marker.id = marker_id++;
      line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::msg::Marker::ADD;
      line_marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
      
      line_marker.color.r = r;
      line_marker.color.g = g;
      line_marker.color.b = b;
      line_marker.color.a = a;
      line_marker.scale.x = scale;
      
      const auto& keys = factor->keys();
      if (keys.size() == 2) {
        try {
          geometry_msgs::msg::Point p1, p2;
          if (values.exists(keys[0]) && values.exists(keys[1])) {
            extractPosition(values, keys[0], p1);
            extractPosition(values, keys[1], p2);
            line_marker.points.push_back(p1);
            line_marker.points.push_back(p2);
            markers.markers.push_back(line_marker);
          }
        } catch (...) {
          // Skip if values not available
        }
      }
    };
    
    // Visualize most recent factors first (reverse iteration)
    
    // Observation factors - show last N
    size_t obs_start = observation_factors.size() > max_observation_factors ? 
                      observation_factors.size() - max_observation_factors : 0;
    for (size_t i = obs_start; i < observation_factors.size(); ++i) {
      visualize_factor(observation_factors[i].second, "observation_factors",
                      0.0, 0.5, 1.0, 0.6, 0.02, 5.0);
    }
    
    // Odometry factors - show last N
    size_t odom_start = odometry_factors.size() > max_odometry_factors ?
                       odometry_factors.size() - max_odometry_factors : 0;
    for (size_t i = odom_start; i < odometry_factors.size(); ++i) {
      visualize_factor(odometry_factors[i].second, "odometry_factors",
                      0.0, 1.0, 0.0, 0.8, 0.05, 10.0);
    }
    
    // Inter-landmark factors - show last N
    size_t inter_start = inter_landmark_factors.size() > max_inter_landmark_factors ?
                        inter_landmark_factors.size() - max_inter_landmark_factors : 0;
    for (size_t i = inter_start; i < inter_landmark_factors.size(); ++i) {
      visualize_factor(inter_landmark_factors[i].second, "inter_landmark_factors",
                      1.0, 0.0, 0.0, 0.8, 0.03, 30.0);
    }
    
    // Loop closure factors - show last N (usually want to see all)
    size_t loop_start = loop_closure_factors.size() > max_loop_closure_factors ?
                       loop_closure_factors.size() - max_loop_closure_factors : 0;
    for (size_t i = loop_start; i < loop_closure_factors.size(); ++i) {
      visualize_factor(loop_closure_factors[i].second, "loop_closure_factors",
                      0.7, 0.0, 0.7, 0.9, 0.06, 60.0);
    }
    
    // Log visualization stats periodically
    static auto last_log_time = node_->now();
    if ((node_->now() - last_log_time).seconds() > 5.0) {
      size_t obs_shown = std::min(observation_factors.size(), max_observation_factors);
      size_t odom_shown = std::min(odometry_factors.size(), max_odometry_factors);
      size_t inter_shown = std::min(inter_landmark_factors.size(), max_inter_landmark_factors);
      size_t loop_shown = std::min(loop_closure_factors.size(), max_loop_closure_factors);
      
      RCLCPP_INFO(node_->get_logger(), 
                  "Factor visualization: %zu/%zu obs (total %zu), %zu/%zu odom (total %zu), "
                  "%zu/%zu inter (total %zu), %zu/%zu loop (total %zu)",
                  obs_shown, max_observation_factors, observation_factors.size(),
                  odom_shown, max_odometry_factors, odometry_factors.size(),
                  inter_shown, max_inter_landmark_factors, inter_landmark_factors.size(),
                  loop_shown, max_loop_closure_factors, loop_closure_factors.size());
      last_log_time = node_->now();
    }
    
    factor_pub_->publish(markers);
  }
  
  /**
   * @brief Visualize keyframe poses
   * @param keyframe_poses Map of keyframe ID to pose
   * @param timestamp Optional timestamp for markers
   */
  void visualizeKeyframes(const std::unordered_map<int, gtsam::Pose2>& keyframe_poses,
                         const rclcpp::Time& timestamp = rclcpp::Time()) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    visualization_msgs::msg::MarkerArray markers;
    
    // Use provided timestamp or current time
    rclcpp::Time marker_time = timestamp.nanoseconds() > 0 ? timestamp : node_->now();
    
    // Delete all marker
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = marker_time;
    delete_marker.ns = "keyframes";
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_marker);
    
    // Add keyframe markers
    for (const auto& [id, pose] : keyframe_poses) {
      // Create arrow marker for pose
      visualization_msgs::msg::Marker arrow;
      arrow.header.frame_id = "map";
      arrow.header.stamp = marker_time;
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
    
    keyframe_pub_->publish(markers);
  }
  
  /**
   * @brief Update and publish SLAM path
   */
  void updatePath(const nav_msgs::msg::Path& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_pub_->publish(path);
  }

private:
  void publishDeleteAll(const std::string& topic) {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = node_->now();
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_marker);
    
    if (topic == "/slam/landmarks") {
      landmark_pub_->publish(markers);
    } else if (topic == "/slam/factor_graph") {
      factor_pub_->publish(markers);
    } else if (topic == "/slam/keyframes") {
      keyframe_pub_->publish(markers);
    }
  }
  
  void setMarkerColor(visualization_msgs::msg::Marker& marker, ConeColor color) {
    switch (color) {
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
  }
  
  void extractPosition(const gtsam::Values& values, gtsam::Key key, 
                      geometry_msgs::msg::Point& point) {
    if (gtsam::Symbol(key).chr() == 'x') {
      // Pose
      auto pose = values.at<gtsam::Pose2>(key);
      point.x = pose.x();
      point.y = pose.y();
      point.z = 0.0;
    } else {
      // Landmark
      auto landmark = values.at<gtsam::Point2>(key);
      point.x = landmark.x();
      point.y = landmark.y();
      point.z = 0.0;
    }
  }
  
  rclcpp::Node* node_;
  bool initialized_ = false;
  
  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr factor_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr keyframe_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
};

} // namespace viewer
} // namespace cone_stellation