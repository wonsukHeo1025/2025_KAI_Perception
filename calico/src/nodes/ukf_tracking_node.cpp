#include <rclcpp/rclcpp.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <atomic>
#include <kalman_filters/tracking/multi_tracker.hpp>
#include "calico/utils/imu_compensator.hpp"
#include "calico/utils/message_converter.hpp"

namespace calico {

class UKFTrackingNode : public rclcpp::Node {
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        custom_interface::msg::TrackedConeArray,
        sensor_msgs::msg::Imu>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

public:
    UKFTrackingNode() : Node("calico_ukf_tracking") {
        RCLCPP_INFO(this->get_logger(), "Initializing CALICO UKF Tracking Node");
        
        // Declare dynamic parameters (matching Python)
        this->declare_parameter<int>("max_missed_detections", 4);
        this->declare_parameter<double>("distance_threshold", 0.7);
        this->declare_parameter<double>("ukf.P_initial_pos", 0.001);
        this->declare_parameter<double>("ukf.P_initial_vel", 100.0);
        this->declare_parameter<double>("ukf.R_measurement", 0.1);
        this->declare_parameter<double>("ukf.Q_process_diag_pos", 0.1);
        this->declare_parameter<double>("ukf.Q_process_diag_vel", 0.1);
        this->declare_parameter<bool>("use_imu", true);
        this->declare_parameter<double>("fixed_dt", 0.056);
        // Time sync parameters
        this->declare_parameter<std::string>("time_sync_mode", "arrival_ros"); // header | arrival_ros | arrival_wall
        this->declare_parameter<double>("arrival_slop", 0.2);
        this->declare_parameter<bool>("override_tracked_stamp_now", true);
        
        // IMU filter parameters (matching Python)
        this->declare_parameter<std::string>("imu_filter.type", "butterworth");
        this->declare_parameter<double>("imu_filter.ema_alpha", 0.1);
        this->declare_parameter<double>("imu_filter.butterworth_cutoff", 10.0);
        this->declare_parameter<int>("imu_filter.butterworth_order", 2);
        
        // IMU to sensor transform (os_imu -> os_sensor)
        std::vector<double> default_transform = {
            1.0, 0.0, 0.0, 0.006253,
            0.0, 1.0, 0.0, -0.011775,
            0.0, 0.0, 1.0, 0.007645,
            0.0, 0.0, 0.0, 1.0
        };
        this->declare_parameter<std::vector<double>>("imu_to_sensor_transform", default_transform);
        
        // Initialize tracking config from parameters
        kalman_filters::tracking::TrackingConfig config;
        config.q_pos = this->get_parameter("ukf.Q_process_diag_pos").as_double();
        config.q_vel = this->get_parameter("ukf.Q_process_diag_vel").as_double();
        config.r_pos = this->get_parameter("ukf.R_measurement").as_double();
        config.p_initial_pos = this->get_parameter("ukf.P_initial_pos").as_double();
        config.p_initial_vel = this->get_parameter("ukf.P_initial_vel").as_double();
        config.alpha = 0.1;  // Fixed as in Python
        config.beta = 2.0;   // Fixed as in Python
        config.kappa = -1.0; // Will be computed as 3 - dim_x
        config.max_age = this->get_parameter("max_missed_detections").as_int();
        config.min_hits = 3;  // Fixed as in Python
        config.max_association_dist = this->get_parameter("distance_threshold").as_double();
        
        use_imu_ = this->get_parameter("use_imu").as_bool();
        fixed_dt_ = this->get_parameter("fixed_dt").as_double();
        {
            const auto mode = this->get_parameter("time_sync_mode").as_string();
            if (mode == "header") time_sync_mode_ = TimeSyncMode::HEADER;
            else if (mode == "arrival_wall") time_sync_mode_ = TimeSyncMode::ARRIVAL_WALL;
            else time_sync_mode_ = TimeSyncMode::ARRIVAL_ROS;
            arrival_slop_ = this->get_parameter("arrival_slop").as_double();
            override_tracked_stamp_now_ = this->get_parameter("override_tracked_stamp_now").as_bool();
        }
        
        // Initialize tracker with IMU transform
        auto transform_vec = this->get_parameter("imu_to_sensor_transform").as_double_array();
        imu_to_sensor_transform_ = Eigen::Matrix4d::Identity();
        if (transform_vec.size() == 16) {
            for (int i = 0; i < 16; ++i) {
                imu_to_sensor_transform_(i / 4, i % 4) = transform_vec[i];
            }
        }
        config.imu_to_sensor_transform = imu_to_sensor_transform_;
        config.enable_imu_compensation = use_imu_;
        
        tracker_ = std::make_shared<kalman_filters::tracking::MultiTracker>(config);
        
        // Initialize IMU compensator if IMU is used
        if (use_imu_) {
            utils::IMUCompensatorConfig imu_config;
            auto filter_type = this->get_parameter("imu_filter.type").as_string();
            if (filter_type == "ema") {
                imu_config.filter_type = utils::IMUFilterType::EMA;
            } else if (filter_type == "butterworth") {
                imu_config.filter_type = utils::IMUFilterType::BUTTERWORTH;
            } else {
                imu_config.filter_type = utils::IMUFilterType::NONE;
            }
            imu_config.ema_alpha = this->get_parameter("imu_filter.ema_alpha").as_double();
            imu_config.butterworth_cutoff = this->get_parameter("imu_filter.butterworth_cutoff").as_double();
            imu_config.butterworth_order = this->get_parameter("imu_filter.butterworth_order").as_int();
            imu_config.remove_gravity = true;
            
            imu_compensator_ = std::make_unique<utils::IMUCompensator>(imu_config);
        }
        
        // Setup QoS (align with RELIABLE publishers)
        rclcpp::QoS qos_rel(10);
        qos_rel.reliability(rclcpp::ReliabilityPolicy::Reliable);
        // IMU often publishes BestEffort (e.g., rosbag2 or sensor drivers)
        rclcpp::QoS qos_bef(10);
        qos_bef.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        // Create subscribers according to sync mode
        if (use_imu_ && time_sync_mode_ == TimeSyncMode::HEADER) {
            // message_filters-based header sync
            cones_sub_.subscribe(this, "/cone/fused", qos_rel.get_rmw_qos_profile());
            imu_sub_.subscribe(this, "/ouster/imu", qos_bef.get_rmw_qos_profile());

            sync_ = std::make_shared<Synchronizer>(
                SyncPolicy(20),
                cones_sub_, imu_sub_);
            sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(arrival_slop_));
            sync_->registerCallback(
                std::bind(&UKFTrackingNode::synchronizedCallback, this,
                          std::placeholders::_1, std::placeholders::_2));
        } else {
            // Arrival-time based processing
            if (use_imu_) {
                imu_arrival_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                    "/ouster/imu", qos_bef,
                    [this](sensor_msgs::msg::Imu::SharedPtr msg){
                        last_imu_msg_ = msg;
                        last_imu_recv_time_ = nowForSync();
                        if (imu_compensator_) {
                            auto imu_data = utils::MessageConverter::fromImuMsg(*msg);
                            imu_compensator_->processIMU(imu_data);
                            utils::IMUData filtered = imu_data;
                            auto filtered_acc = imu_compensator_->getCompensatedAcceleration();
                            filtered.linear_accel_x = filtered_acc.x();
                            filtered.linear_accel_y = filtered_acc.y();
                            filtered.linear_accel_z = filtered_acc.z();
                            auto ang = imu_compensator_->getAngularVelocity();
                            filtered.angular_vel_x = ang.x();
                            filtered.angular_vel_y = ang.y();
                            filtered.angular_vel_z = ang.z();
                            last_kf_imu_ = utils::MessageConverter::toKalmanIMUData(filtered);
                            has_kf_imu_ = true;
                        }
                    });
            }
            cones_arrival_sub_ = this->create_subscription<custom_interface::msg::TrackedConeArray>(
                "/cone/fused", qos_rel,
                [this](custom_interface::msg::TrackedConeArray::SharedPtr msg){
                    // Convert to detections
                    auto cones = utils::MessageConverter::fromTrackedConeArray(*msg);
                    auto detections = utils::MessageConverter::conesToDetections(cones);
                    const auto t = nowForSync();
                    double timestamp = t.seconds();

                    if (use_imu_ && has_kf_imu_ && withinSlop(t, last_imu_recv_time_)) {
                        auto imu_copy = last_kf_imu_;
                        tracker_->update(detections, timestamp, &imu_copy);
                    } else if (use_imu_) {
                        // Update without IMU if not within slop
                        tracker_->update(detections, timestamp, nullptr);
                    } else {
                        tracker_->update(detections, timestamp, nullptr);
                    }

                    publishTrackedConesWithStamp(msg->header);
                });
        }
        
        // Create publisher
        tracked_cones_pub_ = this->create_publisher<custom_interface::msg::TrackedConeArray>(
            "/cone/fused/ukf", qos_rel);
        
        // Add parameter callback for dynamic reconfiguration
        parameter_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&UKFTrackingNode::parametersCallback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "UKF Tracking Node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "IMU usage: %s", use_imu_ ? "enabled" : "disabled");
    }
    
private:
    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter>& parameters) {
        
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        
        for (const auto& param : parameters) {
            // Update config based on parameter changes
            if (param.get_name() == "max_missed_detections") {
                int value = param.as_int();
                if (value > 0 && value < 100) {
                    kalman_filters::tracking::TrackingConfig new_config = tracker_->getConfig();
                    new_config.max_age = value;
                    tracker_->setConfig(new_config);
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated max_missed_detections to %d", value);
                } else {
                    result.successful = false;
                    result.reason = "max_missed_detections must be between 1 and 99";
                }
            }
            else if (param.get_name() == "distance_threshold") {
                double value = param.as_double();
                if (value > 0.0 && value < 10.0) {
                    kalman_filters::tracking::TrackingConfig new_config = tracker_->getConfig();
                    new_config.max_association_dist = value;
                    tracker_->setConfig(new_config);
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated distance_threshold to %.2f", value);
                } else {
                    result.successful = false;
                    result.reason = "distance_threshold must be between 0 and 10";
                }
            }
            else if (param.get_name() == "ukf.R_measurement") {
                double value = param.as_double();
                if (value > 0.0 && value < 10.0) {
                    kalman_filters::tracking::TrackingConfig new_config = tracker_->getConfig();
                    new_config.r_pos = value;
                    tracker_->setConfig(new_config);
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated R_measurement to %.3f", value);
                } else {
                    result.successful = false;
                    result.reason = "R_measurement must be between 0 and 10";
                }
            }
            else if (param.get_name() == "ukf.Q_process_diag_pos") {
                double value = param.as_double();
                if (value > 0.0 && value < 10.0) {
                    kalman_filters::tracking::TrackingConfig new_config = tracker_->getConfig();
                    new_config.q_pos = value;
                    tracker_->setConfig(new_config);
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated Q_process_diag_pos to %.3f", value);
                } else {
                    result.successful = false;
                    result.reason = "Q_process_diag_pos must be between 0 and 10";
                }
            }
            else if (param.get_name() == "ukf.Q_process_diag_vel") {
                double value = param.as_double();
                if (value > 0.0 && value < 10.0) {
                    kalman_filters::tracking::TrackingConfig new_config = tracker_->getConfig();
                    new_config.q_vel = value;
                    tracker_->setConfig(new_config);
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated Q_process_diag_vel to %.3f", value);
                } else {
                    result.successful = false;
                    result.reason = "Q_process_diag_vel must be between 0 and 10";
                }
            }
            else if (param.get_name() == "fixed_dt") {
                double value = param.as_double();
                if (value > 0.0 && value < 1.0) {
                    fixed_dt_ = value;
                    RCLCPP_INFO(this->get_logger(), 
                               "Updated fixed_dt to %.3f", value);
                } else {
                    result.successful = false;
                    result.reason = "fixed_dt must be between 0 and 1";
                }
            }
        }
        
        return result;
    }
    
    // Removed updateConfigField and updateExistingTracks functions as they're no longer needed
    
    void synchronizedCallback(
        const custom_interface::msg::TrackedConeArray::ConstSharedPtr cone_msg,
        const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
        
        // Convert cone message to internal representation
        auto cones = utils::MessageConverter::fromTrackedConeArray(*cone_msg);
        
        // Convert to kalman_filters Detection format
        auto detections = utils::MessageConverter::conesToDetections(cones);
        
        // Get timestamp
        double timestamp = cone_msg->header.stamp.sec + cone_msg->header.stamp.nanosec * 1e-9;
        
        // Process IMU data through compensator
        if (imu_compensator_) {
            auto imu_data = utils::MessageConverter::fromImuMsg(*imu_msg);
            imu_compensator_->processIMU(imu_data);
            
            // Get filtered IMU data
            utils::IMUData filtered_imu = imu_data;
            auto filtered_accel = imu_compensator_->getCompensatedAcceleration();
            filtered_imu.linear_accel_x = filtered_accel.x();
            filtered_imu.linear_accel_y = filtered_accel.y();
            filtered_imu.linear_accel_z = filtered_accel.z();
            
            auto angular_vel = imu_compensator_->getAngularVelocity();
            filtered_imu.angular_vel_x = angular_vel.x();
            filtered_imu.angular_vel_y = angular_vel.y();
            filtered_imu.angular_vel_z = angular_vel.z();
            
            // Convert to kalman_filters IMU format and update tracker
            auto kf_imu = utils::MessageConverter::toKalmanIMUData(filtered_imu);
            tracker_->update(detections, timestamp, &kf_imu);
        } else {
            // Fallback if compensator not initialized
            auto imu_data = utils::MessageConverter::fromImuMsg(*imu_msg);
            auto kf_imu = utils::MessageConverter::toKalmanIMUData(imu_data);
            tracker_->update(detections, timestamp, &kf_imu);
        }
        
        publishTrackedCones(cone_msg->header);
    }
    
    void conesOnlyCallback(const custom_interface::msg::TrackedConeArray::SharedPtr msg) {
        // Convert message to internal representation
        auto cones = utils::MessageConverter::fromTrackedConeArray(*msg);
        
        // Convert to kalman_filters Detection format
        auto detections = utils::MessageConverter::conesToDetections(cones);
        
        // Get timestamp
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        
        // Update tracker without IMU
        tracker_->update(detections, timestamp, nullptr);
        
        publishTrackedCones(msg->header);
    }
    
    void publishTrackedCones(const std_msgs::msg::Header& header) {
        // Get tracked objects from kalman_filters
        auto tracked_objects = tracker_->getTrackedObjects();
        
        // Convert to calico Cone format with velocity
        auto tracked_cones = utils::MessageConverter::trackedObjectsToCones(tracked_objects);
        
        // Convert to ROS message
        auto output_msg = utils::MessageConverter::toTrackedConeArray(tracked_cones);
        output_msg.header = header;
        
        // Publish
        tracked_cones_pub_->publish(output_msg);
        
        // Log statistics
        static std::atomic<int> update_count{0};
        if (update_count.fetch_add(1) % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                       "Tracking %zu cones, %zu active tracks",
                       tracked_cones.size(), tracker_->getNumTracks());
        }
    }

    // Variant that can override timestamp
    void publishTrackedConesWithStamp(const std_msgs::msg::Header& header) {
        auto tracked_objects = tracker_->getTrackedObjects();
        auto tracked_cones = utils::MessageConverter::trackedObjectsToCones(tracked_objects);
        auto output_msg = utils::MessageConverter::toTrackedConeArray(tracked_cones);
        output_msg.header = header;
        if (override_tracked_stamp_now_) {
            const auto t = nowForSync();
            output_msg.header.stamp = t;
        }
        tracked_cones_pub_->publish(output_msg);
    }
    
private:
    enum class TimeSyncMode { HEADER, ARRIVAL_ROS, ARRIVAL_WALL };
    // Tracker
    std::shared_ptr<kalman_filters::tracking::MultiTracker> tracker_;
    std::unique_ptr<utils::IMUCompensator> imu_compensator_;
    
    // Synchronized subscribers
    message_filters::Subscriber<custom_interface::msg::TrackedConeArray> cones_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
    std::shared_ptr<Synchronizer> sync_;
    
    // Simple subscriber (when IMU not used)
    rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr cones_only_sub_;
    // Arrival-based subscribers
    rclcpp::Subscription<custom_interface::msg::TrackedConeArray>::SharedPtr cones_arrival_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_arrival_sub_;
    
    // Publisher
    rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr tracked_cones_pub_;
    
    // Parameters
    bool use_imu_;
    double fixed_dt_;
    Eigen::Matrix4d imu_to_sensor_transform_;
    TimeSyncMode time_sync_mode_;
    double arrival_slop_;
    bool override_tracked_stamp_now_;
    
    // Arrival-time IMU cache
    sensor_msgs::msg::Imu::SharedPtr last_imu_msg_;
    rclcpp::Time last_imu_recv_time_{0, 0, RCL_ROS_TIME};
    kalman_filters::tracking::IMUData last_kf_imu_{};
    bool has_kf_imu_{false};
    
    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr 
        parameter_callback_handle_;

    // Helpers
    rclcpp::Time nowForSync() const {
        if (time_sync_mode_ == TimeSyncMode::ARRIVAL_WALL) {
            return rclcpp::Clock(RCL_SYSTEM_TIME).now();
        }
        return this->now();
    }
    bool withinSlop(const rclcpp::Time& a, const rclcpp::Time& b) const {
        const rclcpp::Duration d = (a > b) ? (a - b) : (b - a);
        return d <= rclcpp::Duration::from_seconds(arrival_slop_);
    }
};

} // namespace calico

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<calico::UKFTrackingNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
