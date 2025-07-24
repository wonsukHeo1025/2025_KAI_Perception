#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "cone_stellation/preprocessing/cone_preprocessor.hpp"
#include "cone_stellation/odometry/cone_odometry_2d.hpp"
#include "cone_stellation/odometry/async_cone_odometry.hpp"
#include "cone_stellation/mapping/cone_mapping.hpp"
#include "cone_stellation/mapping/simple_cone_mapping.hpp"
#include "cone_stellation/common/tentative_landmark.hpp"
#include "cone_stellation/util/ros_utils.hpp"
#include "cone_stellation/util/drift_correction_manager.hpp"
#include "cone_stellation/viewer/slam_visualizer.hpp"
#include "custom_interface/msg/tracked_cone_array.hpp"

namespace cone_stellation {

class ConeSLAMNode : public rclcpp::Node {
public:
  ConeSLAMNode() 
    : Node("cone_slam"), 
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_),
      tf_broadcaster_(this) {
    
    // Load configuration
    load_config();
    
    // Initialize components
    preprocessor_ = std::make_shared<ConePreprocessor>(preprocess_config_);
    
    // Initialize odometry
    ConeOdometryBase::Config odometry_config;
    odometry_config.max_correspondence_distance = 
        this->declare_parameter("odometry.max_correspondence_distance", 3.0);
    odometry_config.use_color_constraint = 
        this->declare_parameter("odometry.use_color_constraint", true);
    odometry_config.min_correspondences = 
        this->declare_parameter("odometry.min_correspondences", 3);
    
    auto cone_odometry = std::make_shared<ConeOdometry2D>(odometry_config);
    async_odometry_ = std::make_shared<AsyncConeOdometry>(cone_odometry);
    async_odometry_->start();
    
    // Check if we should use simple mapping for debugging
    bool use_simple_mapping = this->declare_parameter("mapping.use_simple_mapping", false);
    
    RCLCPP_INFO(this->get_logger(), "use_simple_mapping parameter value: %s", 
                use_simple_mapping ? "true" : "false");
    
    // FORCE USE OF CONEMAPPING FOR TESTING INTER-LANDMARK FACTORS
    use_simple_mapping = false;
    RCLCPP_INFO(this->get_logger(), "FORCING use_simple_mapping to false for testing");
    
    // Debug: Double check the value
    RCLCPP_INFO(this->get_logger(), "After forcing, use_simple_mapping = %s", 
                use_simple_mapping ? "true" : "false");
    
    if (use_simple_mapping) {
      RCLCPP_WARN(this->get_logger(), "Using SimpleConeMapping for debugging");
      simple_mapping_ = std::make_shared<SimpleConeMapping>();
      use_simple_mapping_ = true;
    } else {
      RCLCPP_INFO(this->get_logger(), "Using ConeMapping with inter-landmark factors support");
      mapping_ = std::make_shared<ConeMapping>(mapping_config_);
      use_simple_mapping_ = false;
    }
    
    // Subscribers
    cone_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
        "/fused_sorted_cones_ukf_sim", 10,
        std::bind(&ConeSLAMNode::cone_callback, this, std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 100,
        std::bind(&ConeSLAMNode::odom_callback, this, std::placeholders::_1));
    
    // Initialize visualizer
    slam_visualizer_ = std::make_shared<viewer::SLAMVisualizer>(this);
    slam_visualizer_->initialize();
    
    // Initialize drift correction manager
    drift_manager_ = std::make_shared<DriftCorrectionManager>();
    
    // Publishers
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 10);
    
    // Timers
    visualization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&ConeSLAMNode::visualization_callback, this));
    
    // Initialize map->odom transform as identity
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = this->now();
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = "odom";
    map_to_odom.transform.translation.x = 0.0;
    map_to_odom.transform.translation.y = 0.0;
    map_to_odom.transform.translation.z = 0.0;
    map_to_odom.transform.rotation.w = 1.0;
    tf_broadcaster_.sendTransform(map_to_odom);
    
    // Initialize path message header
    slam_path_.header.frame_id = "map";
    
    RCLCPP_INFO(this->get_logger(), "ConeSLAM node initialized");
    RCLCPP_INFO(this->get_logger(), "Inter-landmark factors: %s", 
                mapping_config_.enable_inter_landmark_factors ? "ENABLED" : "DISABLED");
  }

private:
  void load_config() {
    // Preprocessing parameters
    preprocess_config_.max_cone_distance = 
        this->declare_parameter("preprocessing.max_cone_distance", 20.0);
    preprocess_config_.min_cone_confidence = 
        this->declare_parameter("preprocessing.min_cone_confidence", 0.5);
    preprocess_config_.enable_pattern_detection = 
        this->declare_parameter("preprocessing.enable_pattern_detection", true);
    preprocess_config_.line_fitting_threshold = 
        this->declare_parameter("preprocessing.line_fitting_threshold", 0.2);
    preprocess_config_.min_cones_for_line = 
        this->declare_parameter("preprocessing.min_cones_for_line", 3);
    preprocess_config_.association_threshold = 
        this->declare_parameter("preprocessing.association_threshold", 1.0);
    preprocess_config_.max_tracking_frames = 
        this->declare_parameter("preprocessing.max_tracking_frames", 10);
    
    // Mapping parameters
    mapping_config_.enable_inter_landmark_factors = 
        this->declare_parameter("mapping.enable_inter_landmark_factors", true);
    mapping_config_.inter_landmark_distance_noise = 
        this->declare_parameter("mapping.inter_landmark_distance_noise", 0.1);
    mapping_config_.optimize_every_n_frames = 
        this->declare_parameter("mapping.optimize_every_n_frames", 10);
    mapping_config_.min_covisibility_count = 
        this->declare_parameter("mapping.min_covisibility_count", 2);
    mapping_config_.max_landmark_distance = 
        this->declare_parameter("mapping.max_landmark_distance", 10.0);
    mapping_config_.max_association_distance = 
        this->declare_parameter("association.max_association_distance", 2.0);
    
    // Loop closure parameters (temporarily disabled)
    // TODO: Re-enable when loop closure is properly integrated
    mapping_config_.optimize_on_loop_closure = 
        this->declare_parameter("mapping.optimize_on_loop_closure", false);
    
    // Tentative landmark parameters
    TentativeLandmark::min_observations_ = 
        this->declare_parameter("tentative_landmark.min_observations", 3);
    TentativeLandmark::min_time_span_ = 
        this->declare_parameter("tentative_landmark.min_time_span", 0.5);
    TentativeLandmark::max_position_variance_ = 
        this->declare_parameter("tentative_landmark.max_position_variance", 0.5);
    TentativeLandmark::min_color_confidence_ = 
        this->declare_parameter("tentative_landmark.min_color_confidence", 0.6);
    TentativeLandmark::max_observations_ = 
        this->declare_parameter("tentative_landmark.max_observations", 20);
    
    // Keyframe parameters
    keyframe_translation_threshold_ = 
        this->declare_parameter("keyframe.translation_threshold", 1.0);
    keyframe_rotation_threshold_ = 
        this->declare_parameter("keyframe.rotation_threshold", 0.2);
  }
  
  void cone_callback(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Received cone detection with %zu cones", msg->cones.size());
    
    // TEMPORARY: Skip odometry for debugging
    if (true) {
      // Get current robot pose from TF (ground truth for debugging)
      geometry_msgs::msg::TransformStamped transform;
      try {
        transform = tf_buffer_.lookupTransform("odom", "base_link", 
                                             tf2::TimePointZero);
      } catch (tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
        return;
      }
      
      // Convert to Eigen
      Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
      sensor_pose.translation() = Eigen::Vector3d(
          transform.transform.translation.x,
          transform.transform.translation.y,
          transform.transform.translation.z);
      sensor_pose.rotate(Eigen::Quaterniond(
          transform.transform.rotation.w,
          transform.transform.rotation.x,
          transform.transform.rotation.y,
          transform.transform.rotation.z));
      
      // Convert ROS message to internal format
      auto observations = from_ros_msg(*msg);
      
      // Preprocess observations
      auto processed = preprocessor_->process(observations, sensor_pose, 
                                            rclcpp::Time(msg->header.stamp).seconds());
      
      // Check if this should be a keyframe
      if (should_create_keyframe(sensor_pose)) {
        // Create estimation frame for mapping
        auto frame = std::make_shared<EstimationFrame>();
        frame->timestamp = rclcpp::Time(msg->header.stamp).seconds();
        frame->T_world_sensor = sensor_pose;
        frame->cone_observations = processed;
        frame->is_keyframe = true;
      
      if (use_simple_mapping_) {
        frame->id = 0;  // SimpleConeMapping doesn't track IDs
        // Add to simple mapping
        simple_mapping_->add_keyframe(frame);
        RCLCPP_INFO(this->get_logger(), "Using SimpleConeMapping");
      } else {
        frame->id = mapping_->get_next_pose_id();
        RCLCPP_INFO(this->get_logger(), "About to call ConeMapping::add_keyframe for frame %d", frame->id);
        // Add to mapping
        mapping_->add_keyframe(frame);
        RCLCPP_INFO(this->get_logger(), "ConeMapping::add_keyframe returned");
      }
      
      last_keyframe_pose_ = sensor_pose;
      
      // Store odometry pose for drift correction
      double timestamp = rclcpp::Time(msg->header.stamp).seconds();
      drift_manager_->add_odometry_pose(timestamp, sensor_pose);
      
      // Add to path
      geometry_msgs::msg::PoseStamped path_pose;
      path_pose.header.stamp = msg->header.stamp;
      path_pose.header.frame_id = "map";
      path_pose.pose.position.x = sensor_pose.translation().x();
      path_pose.pose.position.y = sensor_pose.translation().y();
      path_pose.pose.position.z = 0.0;
      
      // Convert rotation to quaternion
      Eigen::Quaterniond q(sensor_pose.rotation());
      path_pose.pose.orientation.x = q.x();
      path_pose.pose.orientation.y = q.y();
      path_pose.pose.orientation.z = q.z();
      path_pose.pose.orientation.w = q.w();
      
      slam_path_.poses.push_back(path_pose);
      
        RCLCPP_INFO(this->get_logger(), "Added keyframe %d with %zu cone observations", 
                    frame->id,
                    processed->cones.size());
        
        // Log cone colors for debugging
        for (const auto& cone : processed->cones) {
          RCLCPP_DEBUG(this->get_logger(), "Cone at (%.2f, %.2f) color: %d", 
                      cone.position.x(), cone.position.y(), static_cast<int>(cone.color));
        }
      }
      return; // TEMPORARY: Skip odometry processing
    }
    
    // Original odometry code (temporarily disabled)
    // ...
  }
  
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // Store odometry for motion model
    last_odom_ = *msg;
  }
  
  bool should_create_keyframe(const Eigen::Isometry3d& current_pose) {
    if (!last_keyframe_pose_) {
      return true; // First frame
    }
    
    // Check translation
    double trans_dist = (current_pose.translation() - 
                        last_keyframe_pose_->translation()).norm();
    if (trans_dist > keyframe_translation_threshold_) {
      return true;
    }
    
    // Check rotation
    Eigen::AngleAxisd angle_diff(current_pose.rotation() * 
                                 last_keyframe_pose_->rotation().transpose());
    if (std::abs(angle_diff.angle()) > keyframe_rotation_threshold_) {
      return true;
    }
    
    return false;
  }
  
  void publish_odometry(const std::shared_ptr<AsyncConeOdometry::OdometryResult>& result) {
    // Publish odometry message
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = rclcpp::Time(result->timestamp);
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";
    
    // Set pose
    odom_msg.pose.pose.position.x = result->T_world_sensor.translation().x();
    odom_msg.pose.pose.position.y = result->T_world_sensor.translation().y();
    odom_msg.pose.pose.position.z = result->T_world_sensor.translation().z();
    
    Eigen::Quaterniond q(result->T_world_sensor.rotation());
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    
    // Set velocity from relative motion
    double dt = 0.1; // Approximate time between frames
    odom_msg.twist.twist.linear.x = result->T_prev_curr.translation().x() / dt;
    odom_msg.twist.twist.linear.y = result->T_prev_curr.translation().y() / dt;
    
    // Rotation velocity
    Eigen::AngleAxisd aa(result->T_prev_curr.rotation());
    odom_msg.twist.twist.angular.z = aa.angle() * aa.axis().z() / dt;
    
    odom_pub_->publish(odom_msg);
    
    // Also publish TF for visualization
    geometry_msgs::msg::TransformStamped odom_tf;
    odom_tf.header = odom_msg.header;
    odom_tf.child_frame_id = "base_link_odom";
    odom_tf.transform.translation.x = odom_msg.pose.pose.position.x;
    odom_tf.transform.translation.y = odom_msg.pose.pose.position.y;
    odom_tf.transform.translation.z = odom_msg.pose.pose.position.z;
    odom_tf.transform.rotation = odom_msg.pose.pose.orientation;
    
    tf_broadcaster_.sendTransform(odom_tf);
  }
  
  void visualization_callback() {
    // Get drift correction transform
    auto T_map_odom = drift_manager_->get_map_to_odom();
    
    // Publish map->odom transform with drift correction
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = this->now();
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = "odom";
    
    // Convert Eigen transform to geometry_msgs
    map_to_odom.transform.translation.x = T_map_odom.translation().x();
    map_to_odom.transform.translation.y = T_map_odom.translation().y();
    map_to_odom.transform.translation.z = T_map_odom.translation().z();
    
    Eigen::Quaterniond q_drift(T_map_odom.rotation());
    map_to_odom.transform.rotation.x = q_drift.x();
    map_to_odom.transform.rotation.y = q_drift.y();
    map_to_odom.transform.rotation.z = q_drift.z();
    map_to_odom.transform.rotation.w = q_drift.w();
    
    tf_broadcaster_.sendTransform(map_to_odom);
    
    // Get current estimates
    if (use_simple_mapping_) {
      // SimpleConeMapping visualization
      auto simple_landmarks = simple_mapping_->get_landmarks();
      
      // Create visualization for simple landmarks
      visualization_msgs::msg::MarkerArray markers;
      int marker_id = 0;
      
      for (const auto& [id, landmark] : simple_landmarks) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "landmarks";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = landmark.position.x();
        marker.pose.position.y = landmark.position.y();
        marker.pose.position.z = 0.3;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.6;
        
        // Set color based on cone type
        switch (landmark.color) {
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
        marker.color.a = 1.0;
        
        marker.lifetime = rclcpp::Duration::from_seconds(0);
        markers.markers.push_back(marker);
      }
      
      // Use visualizer instead
      std::unordered_map<int, ConeLandmark::Ptr> landmarks_map;
      for (const auto& [id, simple_lm] : simple_landmarks) {
        landmarks_map[id] = std::make_shared<ConeLandmark>(id, simple_lm.position, simple_lm.color);
      }
      slam_visualizer_->visualizeLandmarks(landmarks_map);
      
      // Visualize factor graph
      try {
        auto factor_graph = simple_mapping_->get_factor_graph();
        auto values = simple_mapping_->get_current_estimate();
        if (factor_graph.size() > 0) {
          slam_visualizer_->visualizeFactorGraph(factor_graph, values);
          
          // Update drift correction for SimpleConeMapping
          if (!values.empty()) {
            // Get latest pose - find highest pose index
            int latest_pose_id = -1;
            for (int i = 0; i < 100; i++) { // reasonable upper bound for simple mapping
              gtsam::Symbol pose_key('x', i);
              if (values.exists(pose_key)) {
                latest_pose_id = i;
              } else {
                break;
              }
            }
            
            if (latest_pose_id >= 0) {
              gtsam::Symbol latest_pose_key('x', latest_pose_id);
              auto pose2d = values.at<gtsam::Pose2>(latest_pose_key);
              
              // Update drift correction
              Eigen::Isometry3d T_map_base = Eigen::Isometry3d::Identity();
              T_map_base.translation() = Eigen::Vector3d(pose2d.x(), pose2d.y(), 0.0);
              T_map_base.linear() = Eigen::AngleAxisd(pose2d.theta(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
              
              double current_time = this->now().seconds();
              drift_manager_->update_slam_pose(current_time, T_map_base);
            }
          }
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "Failed to visualize simple mapping factors: %s", e.what());
      }
      return;  // Skip the rest for simple mapping
    }
    
    auto landmarks = mapping_->get_landmarks();
    
    // Only publish if we have landmarks
    if (!landmarks.empty()) {
      // Use visualizer
      slam_visualizer_->visualizeLandmarks(landmarks);
      
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Publishing %zu landmarks", landmarks.size());
    }
    
    // Publish factor graph visualization
    try {
      auto factor_graph = mapping_->get_factor_graph();
      auto values = mapping_->get_current_estimate();
      
      if (factor_graph.size() > 0) {
        slam_visualizer_->visualizeFactorGraph(factor_graph, values);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "Publishing %zu factors", factor_graph.size());
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Failed to publish factors: %s", e.what());
    }
    
    // Publish current pose
    try {
      auto values = mapping_->get_current_estimate();
      if (!values.empty()) {
        // Get latest pose - need to find the highest pose index
        int latest_pose_id = -1;
        for (int i = 0; i < 1000; i++) { // reasonable upper bound
          gtsam::Symbol pose_key('x', i);
          if (values.exists(pose_key)) {
            latest_pose_id = i;
          } else {
            break;
          }
        }
        
        if (latest_pose_id >= 0) {
          gtsam::Symbol latest_pose_key('x', latest_pose_id);
          auto pose2d = values.at<gtsam::Pose2>(latest_pose_key);
          
          geometry_msgs::msg::PoseStamped pose_msg;
          pose_msg.header.stamp = this->now();
          pose_msg.header.frame_id = "map";
          pose_msg.pose.position.x = pose2d.x();
          pose_msg.pose.position.y = pose2d.y();
          pose_msg.pose.position.z = 0.0;
          
          tf2::Quaternion q;
          q.setRPY(0, 0, pose2d.theta());
          pose_msg.pose.orientation = tf2::toMsg(q);
          
          pose_pub_->publish(pose_msg);
          
          // Publish TF
          geometry_msgs::msg::TransformStamped tf_msg;
          tf_msg.header = pose_msg.header;
          tf_msg.child_frame_id = "base_link_slam";
          tf_msg.transform.translation.x = pose2d.x();
          tf_msg.transform.translation.y = pose2d.y();
          tf_msg.transform.translation.z = 0.0;
          tf_msg.transform.rotation = pose_msg.pose.orientation;
          
          tf_broadcaster_.sendTransform(tf_msg);
          
          // Update drift correction with optimized SLAM pose
          Eigen::Isometry3d T_map_base = Eigen::Isometry3d::Identity();
          T_map_base.translation() = Eigen::Vector3d(pose2d.x(), pose2d.y(), 0.0);
          T_map_base.linear() = Eigen::AngleAxisd(pose2d.theta(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
          
          // Use current time for now (ideally should get timestamp from keyframe)
          double current_time = this->now().seconds();
          drift_manager_->update_slam_pose(current_time, T_map_base);
          
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                              "Current pose: x=%.2f, y=%.2f, theta=%.2f", 
                              pose2d.x(), pose2d.y(), pose2d.theta());
        }
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Visualization error: %s", e.what());
    }
    
    // Publish accumulated path
    if (!slam_path_.poses.empty()) {
      slam_path_.header.stamp = this->now();
      slam_visualizer_->updatePath(slam_path_);
    }
    
    // Publish keyframes
    try {
      auto keyframe_poses = mapping_->get_poses();
      slam_visualizer_->visualizeKeyframes(keyframe_poses);
      
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Publishing %zu keyframes", keyframe_poses.size());
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Failed to publish keyframes: %s", e.what());
    }
  }
  
  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  
  // Subscribers
  rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr cone_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  
  // Visualizer
  viewer::SLAMVisualizer::Ptr slam_visualizer_;
  
  // Timers
  rclcpp::TimerBase::SharedPtr visualization_timer_;
  
  // SLAM components
  ConePreprocessor::Ptr preprocessor_;
  AsyncConeOdometry::Ptr async_odometry_;
  ConeMapping::Ptr mapping_;
  SimpleConeMapping::Ptr simple_mapping_;
  bool use_simple_mapping_ = false;
  std::shared_ptr<DriftCorrectionManager> drift_manager_;
  
  // Configuration
  ConePreprocessor::Config preprocess_config_;
  ConeMapping::Config mapping_config_;
  
  // State
  std::optional<Eigen::Isometry3d> last_keyframe_pose_;
  nav_msgs::msg::Odometry last_odom_;
  nav_msgs::msg::Path slam_path_;  // Accumulated path
  
  // Parameters
  double keyframe_translation_threshold_;
  double keyframe_rotation_threshold_;
};

} // namespace cone_stellation

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<cone_stellation::ConeSLAMNode>();
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}