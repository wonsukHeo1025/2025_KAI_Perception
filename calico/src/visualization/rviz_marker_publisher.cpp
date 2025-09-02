#include "calico/visualization/rviz_marker_publisher.hpp"
#include <rclcpp/rclcpp.hpp>
#include <set>
#include <algorithm>
#include <geometry_msgs/msg/point.hpp>

namespace calico {
namespace visualization {

RVizMarkerPublisher::RVizMarkerPublisher()
    : cone_height_(0.5),
      cone_radius_(0.15),
      show_track_ids_(true),
      show_color_labels_(false),
      previous_cone_count_(0),
      previous_velocity_count_(0),
      previous_text_count_(0) {
    
    // Initialize default color mappings
    // Lowercase versions (Python fusion output)
    color_map_["blue cone"] = ConeColor(0.0f, 0.0f, 1.0f, 1.0f);
    color_map_["yellow cone"] = ConeColor(1.0f, 1.0f, 0.0f, 1.0f);
    color_map_["red cone"] = ConeColor(1.0f, 0.0f, 0.0f, 1.0f);
    color_map_["unknown"] = ConeColor(0.5f, 0.5f, 0.5f, 0.8f);
    
    // Capitalized versions (Python/C++ UKF output)
    color_map_["Blue Cone"] = ConeColor(0.0f, 0.0f, 1.0f, 1.0f);
    color_map_["Yellow Cone"] = ConeColor(1.0f, 1.0f, 0.0f, 1.0f);
    color_map_["Red Cone"] = ConeColor(1.0f, 0.0f, 0.0f, 1.0f);
    color_map_["Unknown"] = ConeColor(0.5f, 0.5f, 0.5f, 0.8f);
    
    // Single word versions for compatibility
    color_map_["Blue"] = ConeColor(0.0f, 0.0f, 1.0f, 1.0f);
    color_map_["Yellow"] = ConeColor(1.0f, 1.0f, 0.0f, 1.0f);
    color_map_["Red"] = ConeColor(1.0f, 0.0f, 0.0f, 1.0f);
}

visualization_msgs::msg::MarkerArray RVizMarkerPublisher::createMarkerArray(
    const std::vector<utils::Cone>& cones,
    const std::string& frame_id,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 1. Delete previous markers (Python-style: delete ALL previous markers)
    // Delete cone markers
    for (int i = 0; i < previous_cone_count_; ++i) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id;
        delete_marker.header.stamp = timestamp;
        delete_marker.ns = "fused_cones_colored";
        delete_marker.id = i;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(delete_marker);
    }
    
    // Delete velocity arrow markers
    for (int i = 0; i < previous_velocity_count_; ++i) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id;
        delete_marker.header.stamp = timestamp;
        delete_marker.ns = "velocity_arrows";
        delete_marker.id = i;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(delete_marker);
    }
    
    // Delete text markers
    for (int i = 0; i < previous_text_count_; ++i) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id;
        delete_marker.header.stamp = timestamp;
        delete_marker.ns = "track_id_text";
        delete_marker.id = i;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(delete_marker);
    }
    
    // 2. Create new markers with consecutive IDs (matching Python)
    int current_cone_count = 0;
    int current_text_count = 0;
    
    for (size_t i = 0; i < cones.size(); ++i) {
        const auto& cone = cones[i];
        
        // Cone sphere marker (using index as ID like Python)
        marker_array.markers.push_back(
            createConeMarker(cone, i, frame_id, timestamp));
        current_cone_count++;
        
        // Track ID text
        if (show_track_ids_ && cone.id >= 0) {
            marker_array.markers.push_back(
                createTrackIdMarker(cone, current_text_count++, frame_id, timestamp));
        }
    }
    
    // 3. Add velocity arrows with separate counter
    int current_velocity_count = 0;
    for (const auto& cone : cones) {
        if (cone.has_velocity && cone.velocity.norm() > 0.1) {  // Only show arrows for speeds > 0.1 m/s
            marker_array.markers.push_back(
                createVelocityArrowMarker(cone, current_velocity_count++, frame_id, timestamp));
        }
    }
    
    // 4. Update counts for next callback (Python pattern)
    previous_cone_count_ = current_cone_count;
    previous_velocity_count_ = current_velocity_count;
    previous_text_count_ = current_text_count;
    
    return marker_array;
}

void RVizMarkerPublisher::setConeParameters(double height, double radius) {
    cone_height_ = height;
    cone_radius_ = radius;
}

void RVizMarkerPublisher::setColorMapping(const std::string& color_name, const ConeColor& color) {
    color_map_[color_name] = color;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createConeMarker(
    const utils::Cone& cone,
    int marker_id,
    const std::string& frame_id,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "fused_cones_colored";  // Match Python namespace
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position
    marker.pose.position.x = cone.x;
    marker.pose.position.y = cone.y;
    marker.pose.position.z = cone.z;  // Center of sphere at cone position
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    // Scale (for sphere, all dimensions should be equal)
    marker.scale.x = cone_radius_ * 2.0;
    marker.scale.y = cone_radius_ * 2.0;
    marker.scale.z = cone_radius_ * 2.0;
    
    // Color
    marker.color = getColor(cone.color);
    
    // Lifetime
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);  // Infinite lifetime
    
    return marker;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createTrackIdMarker(
    const utils::Cone& cone,
    int marker_id,
    const std::string& frame_id,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "track_id_text";  // Match Python namespace
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position (above cone)
    marker.pose.position.x = cone.x;
    marker.pose.position.y = cone.y;
    marker.pose.position.z = cone.z + cone_height_ + 0.2;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    // Text
    marker.text = std::to_string(cone.id);
    
    // Scale
    marker.scale.z = 0.2;  // Text height
    
    // Color (white)
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    
    // Lifetime
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);  // Infinite lifetime
    
    return marker;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createColorLabelMarker(
    const utils::Cone& cone,
    int marker_id,
    const std::string& frame_id,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "color_labels";
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position (to the side of cone)
    marker.pose.position.x = cone.x + 0.3;
    marker.pose.position.y = cone.y;
    marker.pose.position.z = cone.z + cone_height_ / 2.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    // Text
    marker.text = cone.color;
    
    // Scale
    marker.scale.z = 0.15;  // Text height
    
    // Color (match cone color)
    marker.color = getColor(cone.color);
    
    // Lifetime
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);  // Infinite lifetime
    
    return marker;
}

std_msgs::msg::ColorRGBA RVizMarkerPublisher::getColor(const std::string& color_name) {
    std_msgs::msg::ColorRGBA color;
    
    auto it = color_map_.find(color_name);
    if (it != color_map_.end()) {
        color.r = it->second.r;
        color.g = it->second.g;
        color.b = it->second.b;
        color.a = it->second.a;
    } else {
        // Log unrecognized color for debugging
        static std::set<std::string> logged_colors;
        if (logged_colors.find(color_name) == logged_colors.end()) {
            RCLCPP_WARN(rclcpp::get_logger("rviz_marker_publisher"),
                       "Unrecognized color: '%s', using gray", color_name.c_str());
            logged_colors.insert(color_name);
        }
        
        // Default to gray for unknown colors
        color.r = 0.5;
        color.g = 0.5;
        color.b = 0.5;
        color.a = 0.8;
    }
    
    return color;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createDeleteMarker(
    const std::string& ns, int id) {
    visualization_msgs::msg::Marker marker;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createVelocityArrowMarker(
    const utils::Cone& cone,
    int marker_id,
    const std::string& frame_id,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "velocity_arrows";
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Arrow points
    geometry_msgs::msg::Point start_point;
    start_point.x = cone.x;
    start_point.y = cone.y;
    start_point.z = cone.z;
    
    // Predict position 0.5s ahead
    double prediction_time = 0.5;
    geometry_msgs::msg::Point end_point;
    end_point.x = cone.x + cone.velocity.x() * prediction_time;
    end_point.y = cone.y + cone.velocity.y() * prediction_time;
    end_point.z = cone.z + cone.velocity.z() * prediction_time;
    
    marker.points.push_back(start_point);
    marker.points.push_back(end_point);
    
    // Arrow size (proportional to speed)
    double speed = cone.velocity.norm();
    marker.scale.x = 0.05 * (1.0 + speed * 0.167);  // Shaft diameter
    marker.scale.y = 0.08 * (1.0 + speed * 0.167);  // Head diameter
    marker.scale.z = 0.1;  // Head length
    
    // Color based on speed
    marker.color.a = 0.8;
    if (speed < 1.0) {
        // Slow: green
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
    } else if (speed < 3.0) {
        // Medium: yellow
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
    } else {
        // Fast: red
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
    }
    
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);  // Infinite lifetime
    
    return marker;
}

visualization_msgs::msg::Marker RVizMarkerPublisher::createDeleteAllMarker() {
    visualization_msgs::msg::Marker marker;
    marker.ns = "cones";
    marker.id = 0;
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    return marker;
}

void RVizMarkerPublisher::updateVelocityEstimates(
    const std::vector<utils::Cone>& cones, double current_time) {
    
    for (const auto& cone : cones) {
        if (cone.id < 0) continue;  // Skip cones without track ID
        
        Eigen::Vector3d current_pos(cone.x, cone.y, cone.z);
        
        auto it = previous_positions_.find(cone.id);
        if (it != previous_positions_.end()) {
            double dt = current_time - it->second.first;
            if (dt > 0.01) {  // Minimum time interval
                Eigen::Vector3d velocity = (current_pos - it->second.second) / dt;
                velocity.z() = 0.0;  // Only 2D velocity
                velocity_estimates_[cone.id] = velocity;
            }
        } else {
            velocity_estimates_[cone.id] = Eigen::Vector3d::Zero();
        }
        
        previous_positions_[cone.id] = std::make_pair(current_time, current_pos);
    }
}

void RVizMarkerPublisher::setVelocityForCones(std::vector<utils::Cone>& cones) {
    for (auto& cone : cones) {
        if (cone.id >= 0) {
            auto it = velocity_estimates_.find(cone.id);
            if (it != velocity_estimates_.end()) {
                cone.has_velocity = true;
                cone.velocity = it->second;
            }
        }
    }
}

} // namespace visualization
} // namespace calico