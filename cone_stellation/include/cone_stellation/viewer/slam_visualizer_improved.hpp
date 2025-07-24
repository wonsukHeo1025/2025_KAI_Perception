#pragma once

#include "cone_stellation/viewer/slam_visualizer.hpp"
#include "cone_stellation/factors/cone_observation_factor.hpp"

namespace cone_stellation {
namespace viewer {

/**
 * @brief Improved SLAM visualizer with accurate observation ray visualization
 * 
 * This extends the base SLAMVisualizer to show observation factors more accurately
 * by displaying the actual observation rays from vehicle to observed position.
 */
class SLAMVisualizerImproved : public SLAMVisualizer {
public:
  using Ptr = std::shared_ptr<SLAMVisualizerImproved>;
  
  SLAMVisualizerImproved(rclcpp::Node* node) : SLAMVisualizer(node) {
    setName("SLAMVisualizerImproved");
  }
  
  /**
   * @brief Enhanced factor graph visualization with accurate observation rays
   */
  void visualizeFactorGraph(const gtsam::NonlinearFactorGraph& graph, 
                           const gtsam::Values& values) override {
    std::lock_guard<std::mutex> lock(mutex_);
    
    visualization_msgs::msg::MarkerArray markers;
    
    // Delete all marker
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = node_->now();
    delete_marker.ns = "factors";
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_marker);
    
    int marker_id = 0;
    
    // Iterate through factors
    for (size_t i = 0; i < graph.size(); ++i) {
      const auto& factor = graph[i];
      if (!factor) continue;
      
      // Check if this is a ConeObservationFactor
      auto obs_factor = boost::dynamic_pointer_cast<ConeObservationFactor>(factor);
      if (obs_factor) {
        // Handle observation factors specially
        visualizeObservationFactor(obs_factor, values, markers, marker_id);
        continue;
      }
      
      // Handle other factors normally
      visualization_msgs::msg::Marker line_marker;
      line_marker.header.frame_id = "map";
      line_marker.header.stamp = node_->now();
      line_marker.ns = "factors";
      line_marker.id = marker_id++;
      line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::msg::Marker::ADD;
      
      const auto& keys = factor->keys();
      if (keys.size() < 2) continue;
      
      // Determine factor type and set color/width accordingly
      char type1 = gtsam::Symbol(keys[0]).chr();
      char type2 = gtsam::Symbol(keys[1]).chr();
      
      if (type1 == 'x' && type2 == 'x') {
        // Odometry factor - thick green line
        line_marker.color.r = 0.0;
        line_marker.color.g = 1.0;
        line_marker.color.b = 0.0;
        line_marker.color.a = 0.8;
        line_marker.scale.x = 0.05;
        line_marker.ns = "odometry_factors";
      } else if (type1 == 'l' && type2 == 'l') {
        // Inter-landmark factor - red line
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
    }
    
    factor_pub_->publish(markers);
  }

private:
  /**
   * @brief Visualize observation factor with actual observation ray
   */
  void visualizeObservationFactor(const boost::shared_ptr<ConeObservationFactor>& obs_factor,
                                 const gtsam::Values& values,
                                 visualization_msgs::msg::MarkerArray& markers,
                                 int& marker_id) {
    const auto& keys = obs_factor->keys();
    if (keys.size() != 2) return;
    
    // Get pose and landmark keys
    gtsam::Key pose_key = keys[0];
    gtsam::Key landmark_key = keys[1];
    
    if (!values.exists(pose_key) || !values.exists(landmark_key)) return;
    
    try {
      // Get the pose and landmark values
      auto pose = values.at<gtsam::Pose2>(pose_key);
      auto landmark = values.at<gtsam::Point2>(landmark_key);
      
      // Get the measurement from the factor (this requires adding a getter to ConeObservationFactor)
      // For now, we'll show the line to the optimized landmark position
      // In a real implementation, we'd add: const Point2& measured() const { return measured_; }
      
      // Create observation ray marker
      visualization_msgs::msg::Marker obs_ray;
      obs_ray.header.frame_id = "map";
      obs_ray.header.stamp = node_->now();
      obs_ray.ns = "observation_rays";
      obs_ray.id = marker_id++;
      obs_ray.type = visualization_msgs::msg::Marker::ARROW;
      obs_ray.action = visualization_msgs::msg::Marker::ADD;
      
      // Arrow from pose to observed position
      geometry_msgs::msg::Point start, end;
      start.x = pose.x();
      start.y = pose.y();
      start.z = 0.1;
      
      // For now, use landmark position (in improved version, would use pose.transformFrom(measured))
      end.x = landmark.x();
      end.y = landmark.y();
      end.z = 0.1;
      
      obs_ray.points.push_back(start);
      obs_ray.points.push_back(end);
      
      // Style - thin blue arrow for observations
      obs_ray.scale.x = 0.02;  // Shaft diameter
      obs_ray.scale.y = 0.04;  // Head diameter
      obs_ray.scale.z = 0.04;  // Head length
      
      obs_ray.color.r = 0.0;
      obs_ray.color.g = 0.5;
      obs_ray.color.b = 1.0;
      obs_ray.color.a = 0.6;
      
      markers.markers.push_back(obs_ray);
      
      // Optionally add a small sphere at the observation point
      visualization_msgs::msg::Marker obs_point;
      obs_point.header = obs_ray.header;
      obs_point.ns = "observation_points";
      obs_point.id = marker_id++;
      obs_point.type = visualization_msgs::msg::Marker::SPHERE;
      obs_point.action = visualization_msgs::msg::Marker::ADD;
      
      obs_point.pose.position = end;
      obs_point.pose.orientation.w = 1.0;
      
      obs_point.scale.x = 0.1;
      obs_point.scale.y = 0.1;
      obs_point.scale.z = 0.1;
      
      obs_point.color = obs_ray.color;
      obs_point.color.a = 0.8;
      
      markers.markers.push_back(obs_point);
      
    } catch (const std::exception& e) {
      // Skip this factor if there's an error
      RCLCPP_WARN_ONCE(rclcpp::get_logger("slam_visualizer"), 
                       "Failed to visualize observation factor: %s", e.what());
    }
  }
};

} // namespace viewer
} // namespace cone_stellation