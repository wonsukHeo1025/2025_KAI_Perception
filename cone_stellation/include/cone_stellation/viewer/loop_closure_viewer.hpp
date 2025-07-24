#pragma once

#include <memory>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/rclcpp.hpp>
#include <gtsam/geometry/Pose2.h>

#include "cone_stellation/viewer/viewer_base.hpp"
#include "cone_stellation/mapping/loop_closure_detector.hpp"

namespace cone_stellation {

/**
 * @brief Viewer for loop closure detection visualization
 * 
 * Displays:
 * - Cone constellations as transparent spheres
 * - Loop closure connections as thick lines
 * - Descriptor similarity scores as colors
 */
class LoopClosureViewer : public ViewerBase {
public:
  using Ptr = std::shared_ptr<LoopClosureViewer>;
  
  struct Config {
    double constellation_radius = 10.0;
    double loop_line_width = 0.1;
    double constellation_alpha = 0.3;
    bool show_constellations = true;
    bool show_loop_connections = true;
  };
  
  LoopClosureViewer(const Config& config = Config()) 
    : ViewerBase("loop_closure_markers"),
      config_(config) {
    marker_pub_ = std::make_shared<MarkerPublisher>("loop_closure_markers");
  }
  
  /**
   * @brief Visualize loop closure detections
   */
  void visualize_loop_closures(
      const std::vector<LoopCandidate>& candidates,
      const std::unordered_map<int, gtsam::Pose2>& poses,
      const LoopClosureDetector* detector) {
    
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;
    
    // Visualize each loop closure
    for (const auto& candidate : candidates) {
      // Get poses
      if (poses.count(candidate.query_frame_id) == 0 || 
          poses.count(candidate.reference_frame_id) == 0) {
        continue;
      }
      
      const auto& query_pose = poses.at(candidate.query_frame_id);
      const auto& ref_pose = poses.at(candidate.reference_frame_id);
      
      // Loop closure connection line
      if (config_.show_loop_connections) {
        auto line = create_loop_line(query_pose, ref_pose, candidate.score);
        line.id = marker_id++;
        markers.markers.push_back(line);
        
        // Add arrow showing relative transformation
        auto arrow = create_transform_arrow(ref_pose, candidate.relative_pose);
        arrow.id = marker_id++;
        markers.markers.push_back(arrow);
      }
      
      // Constellations
      if (config_.show_constellations && detector) {
        // Query constellation
        auto query_desc = detector->get_descriptor(candidate.query_frame_id);
        if (query_desc) {
          auto sphere = create_constellation_sphere(
              query_pose, query_desc, 
              std_msgs::msg::ColorRGBA{0.0, 1.0, 0.0, config_.constellation_alpha});
          sphere.id = marker_id++;
          markers.markers.push_back(sphere);
        }
        
        // Reference constellation
        auto ref_desc = detector->get_descriptor(candidate.reference_frame_id);
        if (ref_desc) {
          auto sphere = create_constellation_sphere(
              ref_pose, ref_desc,
              std_msgs::msg::ColorRGBA{1.0, 0.0, 0.0, config_.constellation_alpha});
          sphere.id = marker_id++;
          markers.markers.push_back(sphere);
        }
      }
      
      // Matched cone connections
      for (const auto& [query_cone, ref_cone] : candidate.cone_matches) {
        // This would require landmark positions, skipping for now
      }
    }
    
    publish_markers(markers);
  }
  
  /**
   * @brief Visualize constellation descriptors for debugging
   */
  void visualize_descriptors(
      const std::unordered_map<int, ConstellationDescriptor>& descriptors,
      const std::unordered_map<int, gtsam::Pose2>& poses) {
    
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;
    
    for (const auto& [frame_id, desc] : descriptors) {
      if (poses.count(frame_id) == 0) continue;
      
      const auto& pose = poses.at(frame_id);
      
      // Color based on number of cones
      double intensity = std::min(1.0, desc.cones.size() / 20.0);
      auto color = std_msgs::msg::ColorRGBA{
        intensity, 0.5, 1.0 - intensity, config_.constellation_alpha
      };
      
      auto sphere = create_constellation_sphere(pose, &desc, color);
      sphere.id = marker_id++;
      markers.markers.push_back(sphere);
    }
    
    publish_markers(markers);
  }

private:
  Config config_;
  
  visualization_msgs::msg::Marker create_loop_line(
      const gtsam::Pose2& pose1,
      const gtsam::Pose2& pose2,
      double score) {
    
    visualization_msgs::msg::Marker line;
    line.header = create_header();
    line.ns = "loop_closures";
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    
    // Points
    geometry_msgs::msg::Point p1, p2;
    p1.x = pose1.x();
    p1.y = pose1.y();
    p1.z = 0.5;  // Slightly elevated
    
    p2.x = pose2.x();
    p2.y = pose2.y();
    p2.z = 0.5;
    
    line.points.push_back(p1);
    line.points.push_back(p2);
    
    // Color based on score (green = good, red = poor)
    line.color.r = score;
    line.color.g = 1.0 - score;
    line.color.b = 0.0;
    line.color.a = 0.8;
    
    line.scale.x = config_.loop_line_width;
    
    return line;
  }
  
  visualization_msgs::msg::Marker create_constellation_sphere(
      const gtsam::Pose2& pose,
      const ConstellationDescriptor* desc,
      const std_msgs::msg::ColorRGBA& color) {
    
    visualization_msgs::msg::Marker sphere;
    sphere.header = create_header();
    sphere.ns = "constellations";
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    
    sphere.pose.position.x = pose.x();
    sphere.pose.position.y = pose.y();
    sphere.pose.position.z = 0.0;
    
    // Scale based on constellation spread
    double scale = config_.constellation_radius;
    if (desc) {
      // Could use covariance to determine scale
      scale = std::min(config_.constellation_radius, 
                      std::sqrt(desc->covariance.trace()) * 2.0);
    }
    
    sphere.scale.x = scale;
    sphere.scale.y = scale;
    sphere.scale.z = 2.0;  // Flatter sphere
    
    sphere.color = color;
    
    return sphere;
  }
  
  visualization_msgs::msg::Marker create_transform_arrow(
      const gtsam::Pose2& base_pose,
      const gtsam::Pose2& relative_pose) {
    
    visualization_msgs::msg::Marker arrow;
    arrow.header = create_header();
    arrow.ns = "transform_arrows";
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    
    // Start at base pose
    geometry_msgs::msg::Point start;
    start.x = base_pose.x();
    start.y = base_pose.y();
    start.z = 1.0;
    
    // End at transformed pose
    gtsam::Pose2 end_pose = base_pose * relative_pose;
    geometry_msgs::msg::Point end;
    end.x = end_pose.x();
    end.y = end_pose.y();
    end.z = 1.0;
    
    arrow.points.push_back(start);
    arrow.points.push_back(end);
    
    arrow.scale.x = 0.2;  // Shaft diameter
    arrow.scale.y = 0.3;  // Head diameter
    arrow.scale.z = 0.3;  // Head length
    
    arrow.color.r = 1.0;
    arrow.color.g = 1.0;
    arrow.color.b = 0.0;
    arrow.color.a = 0.8;
    
    return arrow;
  }
};

} // namespace cone_stellation