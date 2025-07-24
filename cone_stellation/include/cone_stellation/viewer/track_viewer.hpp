#ifndef CONE_STELLATION_TRACK_VIEWER_HPP
#define CONE_STELLATION_TRACK_VIEWER_HPP

#include "cone_stellation/viewer/viewer_base.hpp"
#include <Eigen/Core>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Track boundary and centerline visualization
 */
class TrackViewer : public ViewerBase {
public:
    struct TrackBoundary {
        std::vector<Eigen::Vector3d> left_boundary;
        std::vector<Eigen::Vector3d> right_boundary;
        std::vector<Eigen::Vector3d> centerline;
    };

    TrackViewer(rclcpp::Node* node);
    ~TrackViewer() override;

    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;
    void update() override;
    void clear() override;

    /**
     * @brief Set track boundaries
     * @param track Track boundary data
     */
    void setTrackBoundaries(const TrackBoundary& track);

    /**
     * @brief Set left boundary
     * @param points Left boundary points
     */
    void setLeftBoundary(const std::vector<Eigen::Vector3d>& points);

    /**
     * @brief Set right boundary
     * @param points Right boundary points
     */
    void setRightBoundary(const std::vector<Eigen::Vector3d>& points);

    /**
     * @brief Set centerline
     * @param points Centerline points
     */
    void setCenterline(const std::vector<Eigen::Vector3d>& points);

    /**
     * @brief Set frame ID for visualization
     * @param frame_id Frame ID
     */
    void setFrameId(const std::string& frame_id) { frame_id_ = frame_id; }

    /**
     * @brief Set line width for boundaries
     * @param width Line width
     */
    void setBoundaryLineWidth(double width) { boundary_line_width_ = width; }

    /**
     * @brief Set line width for centerline
     * @param width Line width
     */
    void setCenterlineWidth(double width) { centerline_width_ = width; }

private:
    /**
     * @brief Create line strip marker
     * @param points Points for line strip
     * @param color RGB color
     * @param width Line width
     * @param id Marker ID
     * @return Marker message
     */
    visualization_msgs::msg::Marker createLineStrip(
        const std::vector<Eigen::Vector3d>& points,
        const std::array<float, 3>& color,
        double width,
        int id);

    /**
     * @brief Create path message from points
     * @param points Path points
     * @return Path message
     */
    nav_msgs::msg::Path createPath(const std::vector<Eigen::Vector3d>& points);

    rclcpp::Node* node_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr centerline_pub_;
    
    TrackBoundary track_;
    std::string frame_id_{"map"};
    double boundary_line_width_{0.1};
    double centerline_width_{0.15};
    bool initialized_{false};

    // Colors for visualization
    const std::array<float, 3> left_boundary_color_{0.0f, 0.0f, 1.0f};    // Blue
    const std::array<float, 3> right_boundary_color_{1.0f, 1.0f, 0.0f};   // Yellow
    const std::array<float, 3> centerline_color_{0.0f, 1.0f, 0.0f};       // Green
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_TRACK_VIEWER_HPP