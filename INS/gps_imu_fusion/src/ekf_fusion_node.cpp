#include "gps_imu_fusion/ekf_fusion_node.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>
#include <chrono>

using namespace std::chrono_literals;

namespace gps_imu_fusion {

EkfFusionNode::EkfFusionNode(const rclcpp::NodeOptions& options)
: Node("ekf_fusion_node", options), last_update_time_(this->now()) {
  loadParameters();
  
  ekf_ = std::make_unique<kai::KaiEkfCore>();
  
  configureEkfParameters();
  
  gnss_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
    "/ublox_gps_node/fix", 10, std::bind(&EkfFusionNode::gnssCallback, this, std::placeholders::_1));
    
  gnss_vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "/ublox_gps_node/fix_velocity", 10, std::bind(&EkfFusionNode::gnssVelCallback, this, std::placeholders::_1));
    
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "imu/data", 10, std::bind(&EkfFusionNode::imuCallback, this, std::placeholders::_1));

  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odometry/filtered", 10);
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose/filtered", 10);
  
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / update_rate_),
    std::bind(&EkfFusionNode::updateAndPublish, this));
    
  RCLCPP_INFO(this->get_logger(), "EKF Fusion Node initialized");
  RCLCPP_INFO(this->get_logger(), "Listening to GNSS on: %s", gnss_sub_->get_topic_name());
  RCLCPP_INFO(this->get_logger(), "Listening to GNSS velocity on: %s", gnss_vel_sub_->get_topic_name());
  RCLCPP_INFO(this->get_logger(), "Listening to IMU on: %s", imu_sub_->get_topic_name());
}

void EkfFusionNode::loadParameters() {

  if (!this->has_parameter("world_frame_id")) {
    RCLCPP_INFO(this->get_logger(), "Parameters not loaded from file, using defaults");
    
    this->declare_parameter("world_frame_id", "map");
    this->declare_parameter("base_frame_id", "base_link");
    this->declare_parameter("gnss_frame_id", "gps");
    this->declare_parameter("imu_frame_id", "imu_link");
    this->declare_parameter("update_rate", 50.0); 
    this->declare_parameter("mag_declination", 0.0); 
    this->declare_parameter("use_magnetic_declination", false);
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("use_gnss_heading", true);
    this->declare_parameter("min_speed_for_gnss_heading", 0.5);  
    this->declare_parameter("accel_noise", 0.05);
    this->declare_parameter("gyro_noise", 0.00175);
    this->declare_parameter("accel_bias_noise", 0.01);
    this->declare_parameter("gyro_bias_noise", 0.00025);
    this->declare_parameter("accel_bias_tau", 100.0);
    this->declare_parameter("gyro_bias_tau", 50.0);
    this->declare_parameter("gps_pos_noise_ne", 3.0);
    this->declare_parameter("gps_pos_noise_d", 6.0);
    this->declare_parameter("gps_vel_noise_ne", 0.5);
    this->declare_parameter("gps_vel_noise_d", 1.0);
    this->declare_parameter("init_pos_unc", 10.0);
    this->declare_parameter("init_vel_unc", 1.0);
    this->declare_parameter("init_att_unc", 0.34906);
    this->declare_parameter("init_hdg_unc", 3.14159);
    this->declare_parameter("init_accel_bias_unc", 0.981);
    this->declare_parameter("init_gyro_bias_unc", 0.01745);
  } else {
    RCLCPP_INFO(this->get_logger(), "Parameters loaded from file");
  }
  
  world_frame_id_ = this->get_parameter("world_frame_id").as_string();
  base_frame_id_ = this->get_parameter("base_frame_id").as_string();
  gnss_frame_id_ = this->get_parameter("gnss_frame_id").as_string();
  imu_frame_id_ = this->get_parameter("imu_frame_id").as_string();
  update_rate_ = this->get_parameter("update_rate").as_double();
  mag_declination_ = this->get_parameter("mag_declination").as_double() * M_PI / 180.0;  
  use_magnetic_declination_ = this->get_parameter("use_magnetic_declination").as_bool();
  publish_tf_ = this->get_parameter("publish_tf").as_bool();
  use_gnss_heading_ = this->get_parameter("use_gnss_heading").as_bool();
  min_speed_for_gnss_heading_ = this->get_parameter("min_speed_for_gnss_heading").as_double();
  
  RCLCPP_INFO(this->get_logger(), "Update rate: %.1f Hz", update_rate_);
  RCLCPP_INFO(this->get_logger(), "Magnetic declination: %.2f deg (%s)", 
              this->get_parameter("mag_declination").as_double(),
              use_magnetic_declination_ ? "enabled" : "disabled");
  RCLCPP_INFO(this->get_logger(), "GNSS heading: %s (min speed: %.1f m/s)", 
              use_gnss_heading_ ? "enabled" : "disabled",
              min_speed_for_gnss_heading_);
}

void EkfFusionNode::configureEkfParameters() {
  kai::EkfParams params;
  params.accel_noise = this->get_parameter("accel_noise").as_double();
  params.gyro_noise = this->get_parameter("gyro_noise").as_double();
  params.accel_bias_noise = this->get_parameter("accel_bias_noise").as_double();
  params.gyro_bias_noise = this->get_parameter("gyro_bias_noise").as_double();
  params.accel_bias_tau = this->get_parameter("accel_bias_tau").as_double();
  params.gyro_bias_tau = this->get_parameter("gyro_bias_tau").as_double();
  params.gps_pos_noise_ne = this->get_parameter("gps_pos_noise_ne").as_double();
  params.gps_pos_noise_d = this->get_parameter("gps_pos_noise_d").as_double();
  params.gps_vel_noise_ne = this->get_parameter("gps_vel_noise_ne").as_double();
  params.gps_vel_noise_d = this->get_parameter("gps_vel_noise_d").as_double();
  params.init_pos_unc = this->get_parameter("init_pos_unc").as_double();
  params.init_vel_unc = this->get_parameter("init_vel_unc").as_double();
  params.init_att_unc = this->get_parameter("init_att_unc").as_double();
  params.init_hdg_unc = this->get_parameter("init_hdg_unc").as_double();
  params.init_accel_bias_unc = this->get_parameter("init_accel_bias_unc").as_double();
  params.init_gyro_bias_unc = this->get_parameter("init_gyro_bias_unc").as_double();
  
  ekf_->setParameters(params);
  
  RCLCPP_INFO(this->get_logger(), "EKF parameters configured:");
  RCLCPP_INFO(this->get_logger(), "  GPS position noise (NE/D): %.1f / %.1f m", 
              params.gps_pos_noise_ne, params.gps_pos_noise_d);
  RCLCPP_INFO(this->get_logger(), "  GPS velocity noise (NE/D): %.2f / %.2f m/s", 
              params.gps_vel_noise_ne, params.gps_vel_noise_d);
  RCLCPP_INFO(this->get_logger(), "  IMU accel/gyro noise: %.3f m/s² / %.5f rad/s", 
              params.accel_noise, params.gyro_noise);
}

EkfFusionNode::UTMCoordinate EkfFusionNode::llToUtm(double lat, double lon) {
  UTMCoordinate result;
  int zone;
  bool northp;
  
  GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, result.easting, result.northing);
  result.zone = zone;
  result.band = northp ? 'N' : 'S';
  
  return result;
}

double EkfFusionNode::calculateCourse(double vx, double vy) {
  return std::atan2(vy, vx);
}

void EkfFusionNode::gnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  
  RCLCPP_DEBUG(this->get_logger(), "Received GNSS data: lat=%.6f, lon=%.6f, alt=%.2f",
               msg->latitude, msg->longitude, msg->altitude);
  
  latest_gnss_ = *msg;
  
  latest_utm_ = llToUtm(msg->latitude, msg->longitude);
  
  RCLCPP_DEBUG(this->get_logger(), "Converted to UTM: zone=%d%c, easting=%.2f, northing=%.2f",
               latest_utm_.zone, latest_utm_.band, latest_utm_.easting, latest_utm_.northing);
  
  if (!origin_set_) {
    origin_utm_x_ = latest_utm_.easting;
    origin_utm_y_ = latest_utm_.northing;
    origin_set_ = true;
    RCLCPP_INFO(this->get_logger(), "Origin set at UTM: %.2f, %.2f (zone %d%c)", 
                origin_utm_x_, origin_utm_y_, latest_utm_.zone, latest_utm_.band);
  }
  
  kai::GpsCoordinate coor;
  coor.lat = msg->latitude * M_PI / 180.0;  
  coor.lon = msg->longitude * M_PI / 180.0;
  coor.alt = msg->altitude;  
  
  RCLCPP_DEBUG(this->get_logger(), "Updating EKF with GPS coordinates: %.6f, %.6f rad, %.2f m", 
               coor.lat, coor.lon, coor.alt);
  ekf_->gpsCoordinateUpdateEkf(coor);
  
  if (!received_gnss_) {
    RCLCPP_INFO(this->get_logger(), "First GNSS data received");
  }
  received_gnss_ = true;
}

void EkfFusionNode::gnssVelCallback(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  
  RCLCPP_DEBUG(this->get_logger(), "Received GNSS velocity data: vx=%.3f, vy=%.3f, vz=%.3f m/s",
               msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
  
  latest_gnss_vel_ = *msg;
  
  kai::GpsVelocity vel;
  
  vel.vN = msg->twist.twist.linear.x;  
  vel.vE = msg->twist.twist.linear.y;  
  vel.vD = -msg->twist.twist.linear.z; 
  
  RCLCPP_DEBUG(this->get_logger(), "Updating EKF with GPS velocity: vN=%.3f, vE=%.3f, vD=%.3f m/s", 
               vel.vN, vel.vE, vel.vD);
  ekf_->gpsVelocityUpdateEkf(vel);
  
  if (!received_gnss_vel_) {
    RCLCPP_INFO(this->get_logger(), "First GNSS velocity data received");
  }
  received_gnss_vel_ = true;
}

void EkfFusionNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  
  RCLCPP_DEBUG(this->get_logger(), "Received IMU data: gyro=[%.3f, %.3f, %.3f] rad/s, accel=[%.3f, %.3f, %.3f] m/s²",
               msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
               msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  
  latest_imu_ = *msg;
  
  kai::ImuData imu_data;
  
  imu_data.gyroX = msg->angular_velocity.x;
  imu_data.gyroY = msg->angular_velocity.y;
  imu_data.gyroZ = msg->angular_velocity.z;
  
  imu_data.accX = msg->linear_acceleration.x;
  imu_data.accY = msg->linear_acceleration.y;
  imu_data.accZ = msg->linear_acceleration.z;
  
  imu_data.hX = 1.0;
  imu_data.hY = 0.0;
  imu_data.hZ = 0.0;
  
  if (use_magnetic_declination_) {
    float cos_dec = cos(mag_declination_);
    float sin_dec = sin(mag_declination_);
    float hx_temp = imu_data.hX;
    imu_data.hX = hx_temp * cos_dec - imu_data.hY * sin_dec;
    imu_data.hY = hx_temp * sin_dec + imu_data.hY * cos_dec;
    
    RCLCPP_DEBUG(this->get_logger(), "Applied magnetic declination: %.2f deg, hX=%.2f, hY=%.2f", 
                 mag_declination_ * 180.0 / M_PI, imu_data.hX, imu_data.hY);
  }
  
  uint64_t time_us = msg->header.stamp.sec * 1000000LL + msg->header.stamp.nanosec / 1000LL;
  
  if (!received_imu_) {
    RCLCPP_INFO(this->get_logger(), "First IMU data received");
  }
  received_imu_ = true;
  
  RCLCPP_DEBUG(this->get_logger(), "Updating EKF with IMU data, timestamp: %ld us", time_us);
  ekf_->imuUpdateEkf(time_us, imu_data);
}

void EkfFusionNode::updateAndPublish() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  
  if (!received_gnss_ || !received_gnss_vel_ || !received_imu_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000, 
                        "Waiting for sensor data: GNSS=%s, GNSS Vel=%s, IMU=%s", 
                        received_gnss_ ? "received" : "waiting",
                        received_gnss_vel_ ? "received" : "waiting",
                        received_imu_ ? "received" : "waiting");
    return;
  }
  
  bool is_initialized = ekf_->initialized();
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                      "EKF initialization status: %s", is_initialized ? "INITIALIZED" : "NOT INITIALIZED");
  
  if (!is_initialized) {
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "EKF not initialized yet. All sensors are receiving data but initialization may not be complete.");
    return;
  }
  
  publishOdometry();
  if (publish_tf_) {
    publishTransform();
  }
  
  last_update_time_ = this->now();
}

void EkfFusionNode::publishOdometry() {
  if (!ekf_->initialized()) {
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "EKF not initialized yet, skipping odometry publication");
    return;
  }
  
  rclcpp::Time now = this->now();
  
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = now;
  odom.header.frame_id = world_frame_id_;
  odom.child_frame_id = base_frame_id_;
  
  if (origin_set_) {
    odom.pose.pose.position.x = latest_utm_.easting - origin_utm_x_;
    odom.pose.pose.position.y = latest_utm_.northing - origin_utm_y_;
    odom.pose.pose.position.z = latest_gnss_.altitude; 
    
    RCLCPP_DEBUG(this->get_logger(), "Position relative to origin: x=%.2f, y=%.2f, z=%.2f m", 
                 odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Origin not set yet, using zero position");
    odom.pose.pose.position.x = 0.0;
    odom.pose.pose.position.y = 0.0;
    odom.pose.pose.position.z = 0.0;
  }
  
  float roll = ekf_->getRoll_rad();
  float pitch = ekf_->getPitch_rad();
  float heading;
  
  double gps_speed = std::sqrt(
    latest_gnss_vel_.twist.twist.linear.x * latest_gnss_vel_.twist.twist.linear.x +
    latest_gnss_vel_.twist.twist.linear.y * latest_gnss_vel_.twist.twist.linear.y);
  
  if (use_gnss_heading_ && gps_speed > min_speed_for_gnss_heading_) {
    heading = calculateCourse(latest_gnss_vel_.twist.twist.linear.x, latest_gnss_vel_.twist.twist.linear.y);
    RCLCPP_DEBUG(this->get_logger(), "Using GNSS heading: %.2f rad from velocity data", heading);
  } else {
    heading = ekf_->getHeading_rad();  
    RCLCPP_DEBUG(this->get_logger(), "Using EKF estimated heading: %.2f rad", heading);
  }
  
  RCLCPP_DEBUG(this->get_logger(), "Attitude: roll=%.2f, pitch=%.2f, heading=%.2f rad", roll, pitch, heading);
  
  tf2::Quaternion q;
  q.setRPY(roll, pitch, heading);
  
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  
  double vn = ekf_->getVelNorth_ms();
  double ve = ekf_->getVelEast_ms();
  double vd = ekf_->getVelDown_ms();
  
  odom.twist.twist.linear.x = vn;  
  odom.twist.twist.linear.y = ve;  
  odom.twist.twist.linear.z = -vd; 
  
  RCLCPP_DEBUG(this->get_logger(), "Velocity: vN=%.2f, vE=%.2f, vD=%.2f m/s", vn, ve, vd);
  
  odom.twist.twist.angular.x = latest_imu_.angular_velocity.x;
  odom.twist.twist.angular.y = latest_imu_.angular_velocity.y;
  odom.twist.twist.angular.z = latest_imu_.angular_velocity.z;
  
  if (latest_gnss_.position_covariance_type != 0) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        odom.pose.covariance[i * 6 + j] = latest_gnss_.position_covariance[i * 3 + j];
      }
    }
  }
  
  if (latest_gnss_vel_.twist.covariance[0] != 0) {
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        odom.twist.covariance[i * 6 + j] = latest_gnss_vel_.twist.covariance[i * 6 + j];
      }
    }
  }

  odom_pub_->publish(odom);
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000, 
                      "Published odometry: pos=[%.2f, %.2f, %.2f], heading=%.2f deg, speed=%.2f m/s", 
                      odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z,
                      heading * 180.0 / M_PI, gps_speed);

  geometry_msgs::msg::PoseStamped pose;
  pose.header = odom.header;
  pose.pose = odom.pose.pose;
  pose_pub_->publish(pose);
}

void EkfFusionNode::publishTransform() {
  if (!ekf_->initialized() || !origin_set_) {
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 3000, 
                         "Not publishing transform: initialized=%s, origin_set=%s", 
                         ekf_->initialized() ? "true" : "false", 
                         origin_set_ ? "true" : "false");
    return;
  }

  rclcpp::Time now = this->now();

  float roll = ekf_->getRoll_rad();
  float pitch = ekf_->getPitch_rad();
  float heading;

  double gps_speed = std::sqrt(
    latest_gnss_vel_.twist.twist.linear.x * latest_gnss_vel_.twist.twist.linear.x +
    latest_gnss_vel_.twist.twist.linear.y * latest_gnss_vel_.twist.twist.linear.y);
  
  if (use_gnss_heading_ && gps_speed > min_speed_for_gnss_heading_) {
    heading = calculateCourse(latest_gnss_vel_.twist.twist.linear.x, latest_gnss_vel_.twist.twist.linear.y);
  } else {
    heading = ekf_->getHeading_rad();  
  }
  
  tf2::Quaternion q;
  q.setRPY(roll, pitch, heading);
  
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = now;
  transform.header.frame_id = world_frame_id_;
  transform.child_frame_id = base_frame_id_;
  
  transform.transform.translation.x = latest_utm_.easting - origin_utm_x_;
  transform.transform.translation.y = latest_utm_.northing - origin_utm_y_;
  transform.transform.translation.z = latest_gnss_.altitude; 
  
  transform.transform.rotation.w = q.w();
  transform.transform.rotation.x = q.x();
  transform.transform.rotation.y = q.y();
  transform.transform.rotation.z = q.z();

  tf_broadcaster_->sendTransform(transform);
  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 3000, 
                       "Published transform: %s -> %s", 
                       world_frame_id_.c_str(), base_frame_id_.c_str());
}

}