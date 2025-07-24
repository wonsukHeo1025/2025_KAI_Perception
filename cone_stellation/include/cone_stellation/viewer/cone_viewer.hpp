#ifndef CONE_STELLATION_CONE_VIEWER_HPP
#define CONE_STELLATION_CONE_VIEWER_HPP

#include "cone_stellation/viewer/viewer_base.hpp"
#include <Eigen/Core>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Cone detection visualization
 */
class ConeViewer : public ViewerBase {
public:
    struct ConeVisualization {
        Eigen::Vector3d position;
        std::string type;  // "blue", "yellow", "orange", "unknown"
        double confidence;
        int id;
    };

    ConeViewer(rclcpp::Node* node);
    ~ConeViewer() override;

    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;
    void update() override;
    void clear() override;

    /**
     * @brief Add cone to visualization
     * @param cone Cone visualization data
     */
    void addCone(const ConeVisualization& cone);

    /**
     * @brief Add multiple cones to visualization
     * @param cones Vector of cone visualization data
     */
    void addCones(const std::vector<ConeVisualization>& cones);

    /**
     * @brief Set cone visualization scale
     * @param scale Scale factor
     */
    void setConeScale(double scale) { cone_scale_ = scale; }

    /**
     * @brief Set cone visualization height
     * @param height Cone height
     */
    void setConeHeight(double height) { cone_height_ = height; }

    /**
     * @brief Set frame ID for visualization
     * @param frame_id Frame ID
     */
    void setFrameId(const std::string& frame_id) { frame_id_ = frame_id; }

private:
    /**
     * @brief Create marker for cone
     * @param cone Cone data
     * @return Marker message
     */
    visualization_msgs::msg::Marker createConeMarker(const ConeVisualization& cone);

    /**
     * @brief Get color for cone type
     * @param type Cone type
     * @return RGB color values
     */
    std::array<float, 3> getConeColor(const std::string& type);

    rclcpp::Node* node_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    std::vector<ConeVisualization> cones_;
    std::string frame_id_{"map"};
    double cone_scale_{0.5};
    double cone_height_{0.3};
    bool initialized_{false};
    int marker_id_counter_{0};
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_CONE_VIEWER_HPP