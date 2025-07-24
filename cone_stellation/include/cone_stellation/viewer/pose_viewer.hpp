#ifndef CONE_STELLATION_POSE_VIEWER_HPP
#define CONE_STELLATION_POSE_VIEWER_HPP

#include "cone_stellation/viewer/viewer_base.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Vehicle pose and trajectory visualization
 */
class PoseViewer : public ViewerBase {
public:
    struct PoseData {
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        rclcpp::Time timestamp;
    };

    PoseViewer(rclcpp::Node* node);
    ~PoseViewer() override;

    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;
    void update() override;
    void clear() override;

    /**
     * @brief Update current pose
     * @param pose Current pose data
     */
    void updatePose(const PoseData& pose);

    /**
     * @brief Update pose from ROS message
     * @param msg Odometry message
     */
    void updatePose(const nav_msgs::msg::Odometry::SharedPtr msg);

    /**
     * @brief Set trajectory history length
     * @param length Maximum number of poses to keep in history
     */
    void setTrajectoryLength(size_t length) { max_trajectory_length_ = length; }

    /**
     * @brief Enable/disable trajectory visualization
     * @param enable Enable flag
     */
    void enableTrajectory(bool enable) { show_trajectory_ = enable; }

    /**
     * @brief Enable/disable velocity vector visualization
     * @param enable Enable flag
     */
    void enableVelocityVector(bool enable) { show_velocity_ = enable; }

    /**
     * @brief Enable/disable coordinate frame visualization
     * @param enable Enable flag
     */
    void enableCoordinateFrame(bool enable) { show_coordinate_frame_ = enable; }

    /**
     * @brief Set vehicle model scale
     * @param scale Scale factor
     */
    void setVehicleScale(double scale) { vehicle_scale_ = scale; }

    /**
     * @brief Set frame ID for visualization
     * @param frame_id Frame ID
     */
    void setFrameId(const std::string& frame_id) { frame_id_ = frame_id; }

private:
    /**
     * @brief Create vehicle marker
     * @param pose Current pose
     * @return Marker message
     */
    visualization_msgs::msg::Marker createVehicleMarker(const PoseData& pose);

    /**
     * @brief Create trajectory marker
     * @return Marker message
     */
    visualization_msgs::msg::Marker createTrajectoryMarker();

    /**
     * @brief Create velocity vector marker
     * @param pose Current pose with velocity
     * @return Marker message
     */
    visualization_msgs::msg::Marker createVelocityMarker(const PoseData& pose);

    /**
     * @brief Create coordinate frame markers
     * @param pose Current pose
     * @return Marker array for X, Y, Z axes
     */
    visualization_msgs::msg::MarkerArray createCoordinateFrameMarkers(const PoseData& pose);

    /**
     * @brief Publish current pose as PoseStamped
     * @param pose Current pose
     */
    void publishPoseStamped(const PoseData& pose);

    rclcpp::Node* node_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    
    PoseData current_pose_;
    std::vector<PoseData> trajectory_history_;
    
    std::string frame_id_{"map"};
    size_t max_trajectory_length_{500};
    double vehicle_scale_{1.0};
    
    bool initialized_{false};
    bool show_trajectory_{true};
    bool show_velocity_{true};
    bool show_coordinate_frame_{true};
    
    // Vehicle dimensions (scaled by vehicle_scale_)
    const double vehicle_length_{4.5};
    const double vehicle_width_{2.0};
    const double vehicle_height_{1.5};
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_POSE_VIEWER_HPP