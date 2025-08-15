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
#include "cone_stellation/common/tentative_landmark.hpp"
#include "cone_stellation/util/ros_utils.hpp"
#include "cone_stellation/util/drift_correction_manager.hpp"
#include "cone_stellation/viewer/slam_visualizer.hpp"
#include "custom_interface/msg/tracked_cone_array.hpp"

namespace cone_stellation {

class ConeSLAMNode : public rclcpp::Node {
public:
  ConeSLAMNode() 
    : Node("cone_slam", rclcpp::NodeOptions().use_intra_process_comms(false)), 
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
    
    // Initialize ConeMapping with inter-landmark factors support
    RCLCPP_INFO(this->get_logger(), "Using ConeMapping with inter-landmark factors support");
    mapping_ = std::make_shared<ConeMapping>(mapping_config_);
    
    // Subscribers with QoS settings
    rclcpp::QoS cone_qos(10);
    cone_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    cone_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    
    cone_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
        "/cones/fused/ukf", cone_qos,
        std::bind(&ConeSLAMNode::cone_callback, this, std::placeholders::_1));
    
    rclcpp::QoS odom_qos(100);
    odom_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    odom_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", odom_qos,
        std::bind(&ConeSLAMNode::odom_callback, this, std::placeholders::_1));
    
    // Initialize visualizer
    slam_visualizer_ = std::make_shared<viewer::SLAMVisualizer>(this);
    slam_visualizer_->initialize();
    
    // Initialize drift correction manager
    drift_manager_ = std::make_shared<DriftCorrectionManager>();
    
    // Publishers with best effort QoS for real-time performance
    rclcpp::QoS pub_qos(10);
    pub_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    pub_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", pub_qos);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", pub_qos);
    
    // Timers - use regular timer instead of wall timer for sim time compatibility
    visualization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&ConeSLAMNode::visualization_callback, this));
    
    // Initialize map->odom transform as identity
    // This is needed even with drift correction disabled to complete the TF tree
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = this->now();
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = "odom";
    map_to_odom.transform.translation.x = 0.0;
    map_to_odom.transform.translation.y = 0.0;
    map_to_odom.transform.translation.z = 0.0;
    map_to_odom.transform.rotation.w = 1.0;
    tf_broadcaster_.sendTransform(map_to_odom);
    
    // Start a timer to continuously publish identity map->odom
    map_odom_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        [this]() {
          geometry_msgs::msg::TransformStamped tf;
          tf.header.stamp = this->now();
          tf.header.frame_id = "map";
          tf.child_frame_id = "odom";
          tf.transform.translation.x = 0.0;
          tf.transform.translation.y = 0.0;
          tf.transform.translation.z = 0.0;
          tf.transform.rotation.w = 1.0;
          tf_broadcaster_.sendTransform(tf);
        });
    
    // REMOVED: base_link -> base_link_slam transform is unnecessary
    // Only map -> base_link_slam is needed for debugging
    
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
    
    // Use odometry data instead of TF lookup
    if (last_odom_.header.stamp.sec == 0) {
      RCLCPP_WARN(this->get_logger(), "No odometry data available yet");
      return;
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Using odometry pose: x=%.2f, y=%.2f, z=%.2f",
                 last_odom_.pose.pose.position.x,
                 last_odom_.pose.pose.position.y,
                 last_odom_.pose.pose.position.z);
    
    // Get current robot pose from odometry
    Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
    sensor_pose.translation() = Eigen::Vector3d(
        last_odom_.pose.pose.position.x,
        last_odom_.pose.pose.position.y,
        last_odom_.pose.pose.position.z);
    sensor_pose.rotate(Eigen::Quaterniond(
        last_odom_.pose.pose.orientation.w,
        last_odom_.pose.pose.orientation.x,
        last_odom_.pose.pose.orientation.y,
        last_odom_.pose.pose.orientation.z));
      
      // Convert ROS message to internal format
      auto observations = from_ros_msg(*msg);
      RCLCPP_INFO(this->get_logger(), "Converted %zu cone observations from ROS msg", 
                  observations.size());
      
      // Transform cone observations from os_sensor frame to map frame
      try {
        // Get os_sensor to base_link transform
        geometry_msgs::msg::TransformStamped os_to_base_tf;
        try {
          os_to_base_tf = tf_buffer_.lookupTransform("base_link", msg->header.frame_id,
                                                      tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
          RCLCPP_WARN(this->get_logger(), 
                      "Could not get transform from %s to base_link: %s. Using identity.",
                      msg->header.frame_id.c_str(), ex.what());
          // Use identity if transform not available
          os_to_base_tf.transform.rotation.w = 1.0;
        }
        
        // Convert TF to Eigen
        Eigen::Isometry3d T_base_sensor = Eigen::Isometry3d::Identity();
        T_base_sensor.translation() = Eigen::Vector3d(
            os_to_base_tf.transform.translation.x,
            os_to_base_tf.transform.translation.y,
            os_to_base_tf.transform.translation.z);
        T_base_sensor.rotate(Eigen::Quaterniond(
            os_to_base_tf.transform.rotation.w,
            os_to_base_tf.transform.rotation.x,
            os_to_base_tf.transform.rotation.y,
            os_to_base_tf.transform.rotation.z));
        
        // Transform cone observations to map frame
        for (auto& obs : observations) {
          // Convert 2D cone position to 3D in sensor frame
          Eigen::Vector3d cone_sensor(obs.position.x(), obs.position.y(), 0.0);
          
          // Transform to base_link frame
          Eigen::Vector3d cone_base = T_base_sensor * cone_sensor;
          
          // Keep observation in vehicle frame (base_link) for factor graph
          // Factor expects relative position from vehicle, not absolute map position
          obs.position = Eigen::Vector2d(cone_base.x(), cone_base.y());
        }
        
        RCLCPP_INFO(this->get_logger(), "Transformed %zu cones to base_link frame", 
                    observations.size());
        
      } catch (const std::exception& ex) {
        RCLCPP_ERROR(this->get_logger(), "Error transforming cones: %s", ex.what());
        return;
      }
      
      // Preprocess observations
      auto processed = preprocessor_->process(observations, sensor_pose, 
                                            rclcpp::Time(msg->header.stamp).seconds());
      RCLCPP_INFO(this->get_logger(), "After preprocessing: %zu cones", 
                  processed->cones.size());
      
      // Check if this should be a keyframe
      bool is_keyframe = should_create_keyframe(sensor_pose);
      RCLCPP_INFO(this->get_logger(), "Should create keyframe: %s", is_keyframe ? "YES" : "NO");
      
      if (is_keyframe) {
        // Create estimation frame for mapping
        auto frame = std::make_shared<EstimationFrame>();
        frame->timestamp = rclcpp::Time(msg->header.stamp).seconds();
        frame->T_world_sensor = sensor_pose;
        frame->cone_observations = processed;
        frame->is_keyframe = true;
      
      frame->id = mapping_->get_next_pose_id();
      RCLCPP_INFO(this->get_logger(), "About to call ConeMapping::add_keyframe for frame %d", frame->id);
      // Add to mapping
      mapping_->add_keyframe(frame);
      RCLCPP_INFO(this->get_logger(), "ConeMapping::add_keyframe returned");
      
      last_keyframe_pose_ = sensor_pose;
      
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
  }
  
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // Store odometry for motion model
    last_odom_ = *msg;
    
    // Convert odometry to Eigen transform
    Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
    T_odom_base.translation() = Eigen::Vector3d(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z);
    T_odom_base.rotate(Eigen::Quaterniond(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z));
    
    // DISABLED: Drift correction temporarily disabled to fix circular dependency
    // double timestamp = rclcpp::Time(msg->header.stamp).seconds();
    // drift_manager_->add_odometry_pose(timestamp, T_odom_base);
    // 
    // RCLCPP_DEBUG(this->get_logger(), "Added odometry pose to drift manager at %.3f", timestamp);
  }
  
  bool should_create_keyframe(const Eigen::Isometry3d& current_pose) {
    if (!last_keyframe_pose_) {
      RCLCPP_INFO(this->get_logger(), "First keyframe - no previous pose");
      return true; // First frame
    }
    
    // Check translation
    double trans_dist = (current_pose.translation() - 
                        last_keyframe_pose_->translation()).norm();
    
    // Check rotation
    Eigen::AngleAxisd angle_diff(current_pose.rotation() * 
                                 last_keyframe_pose_->rotation().transpose());
    double rot_dist = std::abs(angle_diff.angle());
    
    RCLCPP_DEBUG(this->get_logger(), 
                 "Keyframe check: trans_dist=%.3f (thresh=%.3f), rot_dist=%.3f (thresh=%.3f)",
                 trans_dist, keyframe_translation_threshold_,
                 rot_dist, keyframe_rotation_threshold_);
    
    if (trans_dist > keyframe_translation_threshold_) {
      RCLCPP_INFO(this->get_logger(), "New keyframe: translation threshold exceeded (%.3f > %.3f)",
                  trans_dist, keyframe_translation_threshold_);
      return true;
    }
    
    if (rot_dist > keyframe_rotation_threshold_) {
      RCLCPP_INFO(this->get_logger(), "New keyframe: rotation threshold exceeded (%.3f > %.3f)",
                  rot_dist, keyframe_rotation_threshold_);
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
    
    // DISABLED: TF publishing to prevent conflict with EKF
    // The EKF publishes odom->base_link, SLAM should only publish map->odom
    // geometry_msgs::msg::TransformStamped odom_tf;
    // odom_tf.header = odom_msg.header;
    // odom_tf.child_frame_id = "base_link_odom";
    // odom_tf.transform.translation.x = odom_msg.pose.pose.position.x;
    // odom_tf.transform.translation.y = odom_msg.pose.pose.position.y;
    // odom_tf.transform.translation.z = odom_msg.pose.pose.position.z;
    // odom_tf.transform.rotation = odom_msg.pose.pose.orientation;
    // 
    // tf_broadcaster_.sendTransform(odom_tf);
  }
  
  void visualization_callback() {
    RCLCPP_DEBUG(this->get_logger(), "Visualization callback called");
    
    // TEMPORARY FIX: Use identity transform for map->odom to prevent circular dependency
    // auto T_map_odom = drift_manager_->get_map_to_odom();
    // NOTE: map->odom identity transform is now published by the separate timer
    
    // Use odometry timestamp for all visualizations
    rclcpp::Time viz_timestamp;
    if (last_odom_.header.stamp.sec > 0) {
      viz_timestamp = rclcpp::Time(last_odom_.header.stamp);
    } else {
      viz_timestamp = this->now();
    }
    
    // REMOVED: base_link -> base_link_slam transform
    // The map -> base_link_slam transform from SLAM optimization is sufficient
    
    // Get current estimates from ConeMapping
    auto landmarks = mapping_->get_landmarks();
    
    // Debug: Always log landmark count
    RCLCPP_INFO(this->get_logger(), "Retrieved %zu landmarks from mapping", landmarks.size());
    
    // Only publish if we have landmarks
    if (!landmarks.empty()) {
      // Use visualizer
      slam_visualizer_->visualizeLandmarks(landmarks, viz_timestamp);
      
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Publishing %zu landmarks", landmarks.size());
      
      // Debug: Log first few landmarks
      int count = 0;
      for (const auto& [id, landmark] : landmarks) {
        if (count++ < 3) {
          RCLCPP_INFO(this->get_logger(), "Landmark %d at (%.2f, %.2f) color: %d",
                      id, landmark->position().x(), landmark->position().y(), 
                      static_cast<int>(landmark->color()));
        }
      }
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
          
          // Publish TF from map to base_link_slam
          geometry_msgs::msg::TransformStamped tf_msg;
          tf_msg.header = pose_msg.header;
          tf_msg.header.frame_id = "map";  // Publish from map frame
          tf_msg.child_frame_id = "base_link_slam";
          tf_msg.transform.translation.x = pose2d.x();
          tf_msg.transform.translation.y = pose2d.y();
          tf_msg.transform.translation.z = 0.0;
          tf_msg.transform.rotation = pose_msg.pose.orientation;
          
          tf_broadcaster_.sendTransform(tf_msg);
          
          // DISABLED: Drift correction temporarily disabled to fix circular dependency
          // Eigen::Isometry3d T_map_base = Eigen::Isometry3d::Identity();
          // T_map_base.translation() = Eigen::Vector3d(pose2d.x(), pose2d.y(), 0.0);
          // T_map_base.linear() = Eigen::AngleAxisd(pose2d.theta(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
          // 
          // double current_time = this->now().seconds();
          // drift_manager_->update_slam_pose(current_time, T_map_base);
          
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
      slam_path_.header.stamp = viz_timestamp;
      slam_visualizer_->updatePath(slam_path_);
    }
    
    // Publish keyframes
    try {
      auto keyframe_poses = mapping_->get_poses();
      slam_visualizer_->visualizeKeyframes(keyframe_poses, viz_timestamp);
      
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
  rclcpp::TimerBase::SharedPtr map_odom_timer_;
  
  // SLAM components
  ConePreprocessor::Ptr preprocessor_;
  AsyncConeOdometry::Ptr async_odometry_;
  ConeMapping::Ptr mapping_;
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
  
  // Use MultiThreadedExecutor to prevent blocking
  // This allows visualization_callback to run even during heavy keyframe processing
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}