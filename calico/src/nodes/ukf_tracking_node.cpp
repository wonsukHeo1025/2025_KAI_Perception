#include <rclcpp/rclcpp.hpp>
#include <custom_interface/msg/modified_float32_multi_array.hpp>
#include <custom_interface/msg/tracked_cone_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "calico/tracking/ukf_tracker.hpp"
#include "calico/tracking/imu_compensator.hpp"
#include "calico/utils/message_converter.hpp"

namespace calico {

class UKFTrackingNode : public rclcpp::Node {
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        custom_interface::msg::ModifiedFloat32MultiArray,
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
        
        // Initialize UKF config from parameters
        tracking::UKFConfig config;
        config.q_pos = this->get_parameter("ukf.Q_process_diag_pos").as_double();
        config.q_vel = this->get_parameter("ukf.Q_process_diag_vel").as_double();
        config.r_pos = this->get_parameter("ukf.R_measurement").as_double();
        config.p_initial_pos = this->get_parameter("ukf.P_initial_pos").as_double();
        config.p_initial_vel = this->get_parameter("ukf.P_initial_vel").as_double();
        config.alpha = 0.1;  // Fixed as in Python
        config.beta = 2.0;   // Fixed as in Python
        config.kappa = -1.0; // Will be computed as 3 - dim_x
        config.max_age_before_deletion = this->get_parameter("max_missed_detections").as_int();
        config.min_hits_before_confirmation = 3;  // Fixed as in Python
        config.max_association_distance = this->get_parameter("distance_threshold").as_double();
        
        use_imu_ = this->get_parameter("use_imu").as_bool();
        fixed_dt_ = this->get_parameter("fixed_dt").as_double();
        
        // Initialize tracker with IMU transform
        auto transform_vec = this->get_parameter("imu_to_sensor_transform").as_double_array();
        imu_to_sensor_transform_ = Eigen::Matrix4d::Identity();
        if (transform_vec.size() == 16) {
            for (int i = 0; i < 16; ++i) {
                imu_to_sensor_transform_(i / 4, i % 4) = transform_vec[i];
            }
        }
        config.imu_to_sensor_transform = imu_to_sensor_transform_;
        
        tracker_ = std::make_unique<tracking::UKFTracker>(config);
        
        // Initialize IMU compensator if IMU is used
        if (use_imu_) {
            tracking::IMUCompensatorConfig imu_config;
            auto filter_type = this->get_parameter("imu_filter.type").as_string();
            if (filter_type == "ema") {
                imu_config.filter_type = tracking::IMUFilterType::EMA;
            } else if (filter_type == "butterworth") {
                imu_config.filter_type = tracking::IMUFilterType::BUTTERWORTH;
            } else {
                imu_config.filter_type = tracking::IMUFilterType::NONE;
            }
            imu_config.ema_alpha = this->get_parameter("imu_filter.ema_alpha").as_double();
            imu_config.butterworth_cutoff = this->get_parameter("imu_filter.butterworth_cutoff").as_double();
            imu_config.butterworth_order = this->get_parameter("imu_filter.butterworth_order").as_int();
            imu_config.remove_gravity = true;
            
            imu_compensator_ = std::make_unique<tracking::IMUCompensator>(imu_config);
        }
        
        // Setup QoS
        rclcpp::QoS qos(10);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        // Create subscribers for time synchronization
        if (use_imu_) {
            cones_sub_.subscribe(this, "/fused_sorted_cones", qos.get_rmw_qos_profile());
            imu_sub_.subscribe(this, "/ouster/imu", qos.get_rmw_qos_profile());
            
            // Create synchronizer
            sync_ = std::make_shared<Synchronizer>(
                SyncPolicy(20),  // queue size
                cones_sub_,
                imu_sub_);
            sync_->setMaxIntervalDuration(rclcpp::Duration(0, 150000000));  // 0.15s slop
            sync_->registerCallback(
                std::bind(&UKFTrackingNode::synchronizedCallback, this,
                         std::placeholders::_1, std::placeholders::_2));
        } else {
            // Without IMU, use simple subscription
            cones_only_sub_ = this->create_subscription<custom_interface::msg::ModifiedFloat32MultiArray>(
                "/fused_sorted_cones", qos,
                std::bind(&UKFTrackingNode::conesOnlyCallback, this, std::placeholders::_1));
        }
        
        // Create publisher
        tracked_cones_pub_ = this->create_publisher<custom_interface::msg::TrackedConeArray>(
            "/fused_sorted_cones_ukf", qos);
        
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
                    tracker_->setConfig(updateConfigField(
                        tracker_->getConfig(), "max_age_before_deletion", value));
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
                    tracker_->setConfig(updateConfigField(
                        tracker_->getConfig(), "max_association_distance", value));
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
                    tracker_->setConfig(updateConfigField(
                        tracker_->getConfig(), "r_pos", value));
                    updateExistingTracks("R", value);
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
                    tracker_->setConfig(updateConfigField(
                        tracker_->getConfig(), "q_pos", value));
                    updateExistingTracks("Q_pos", value);
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
                    tracker_->setConfig(updateConfigField(
                        tracker_->getConfig(), "q_vel", value));
                    updateExistingTracks("Q_vel", value);
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
    
    tracking::UKFConfig updateConfigField(const tracking::UKFConfig& config,
                                         const std::string& field,
                                         double value) {
        tracking::UKFConfig new_config = config;
        if (field == "max_age_before_deletion") {
            new_config.max_age_before_deletion = static_cast<int>(value);
        } else if (field == "max_association_distance") {
            new_config.max_association_distance = value;
        } else if (field == "r_pos") {
            new_config.r_pos = value;
        } else if (field == "q_pos") {
            new_config.q_pos = value;
        } else if (field == "q_vel") {
            new_config.q_vel = value;
        }
        return new_config;
    }
    
    void updateExistingTracks(const std::string& param_type, double value) {
        // Update existing tracks' UKF parameters
        auto& tracks = tracker_->getTracks();
        for (auto& [id, track] : tracks) {
            if (param_type == "R") {
                track->setMeasurementNoise(value);
            } else if (param_type == "Q_pos" || param_type == "Q_vel") {
                // Need to get current Q values and update
                auto current_config = tracker_->getConfig();
                track->setProcessNoise(current_config.q_pos, current_config.q_vel);
            }
        }
    }
    
    void synchronizedCallback(
        const custom_interface::msg::ModifiedFloat32MultiArray::ConstSharedPtr cone_msg,
        const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
        
        // Convert cone message to internal representation
        auto cones = utils::MessageConverter::fromModifiedFloat32MultiArray(*cone_msg);
        
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
            
            // Update tracker with filtered IMU
            tracker_->update(cones, timestamp, &filtered_imu);
        } else {
            // Fallback if compensator not initialized
            auto imu_data = utils::MessageConverter::fromImuMsg(*imu_msg);
            tracker_->update(cones, timestamp, &imu_data);
        }
        
        publishTrackedCones(cone_msg->header);
    }
    
    void conesOnlyCallback(const custom_interface::msg::ModifiedFloat32MultiArray::SharedPtr msg) {
        // Convert message to internal representation
        auto cones = utils::MessageConverter::fromModifiedFloat32MultiArray(*msg);
        
        // Get timestamp
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        
        // Update tracker without IMU
        tracker_->update(cones, timestamp, nullptr);
        
        publishTrackedCones(msg->header);
    }
    
    void publishTrackedCones(const std_msgs::msg::Header& header) {
        // Get tracked cones
        auto tracked_cones = tracker_->getTrackedCones();
        
        // Convert to ROS message
        auto output_msg = utils::MessageConverter::toTrackedConeArray(tracked_cones);
        output_msg.header = header;
        
        // Publish
        tracked_cones_pub_->publish(output_msg);
        
        // Log statistics
        static int update_count = 0;
        if (++update_count % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                       "Tracking %zu cones, %zu active tracks",
                       tracked_cones.size(), tracker_->getTracks().size());
        }
    }
    
private:
    // Tracker
    std::unique_ptr<tracking::UKFTracker> tracker_;
    std::unique_ptr<tracking::IMUCompensator> imu_compensator_;
    
    // Synchronized subscribers
    message_filters::Subscriber<custom_interface::msg::ModifiedFloat32MultiArray> cones_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
    std::shared_ptr<Synchronizer> sync_;
    
    // Simple subscriber (when IMU not used)
    rclcpp::Subscription<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr cones_only_sub_;
    
    // Publisher
    rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr tracked_cones_pub_;
    
    // Parameters
    bool use_imu_;
    double fixed_dt_;
    Eigen::Matrix4d imu_to_sensor_transform_;
    
    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr 
        parameter_callback_handle_;
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