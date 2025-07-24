#ifndef CONE_STELLATION_VISUALIZATION_UTILS_HPP
#define CONE_STELLATION_VISUALIZATION_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <vector>
#include <array>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Utility functions for visualization
 */
class VisualizationUtils {
public:
    /**
     * @brief Convert Eigen vector to geometry_msgs Point
     * @param vec Eigen vector
     * @return Geometry point message
     */
    static geometry_msgs::msg::Point eigenToPoint(const Eigen::Vector3d& vec);

    /**
     * @brief Convert Eigen quaternion to geometry_msgs Quaternion
     * @param quat Eigen quaternion
     * @return Geometry quaternion message
     */
    static geometry_msgs::msg::Quaternion eigenToQuaternion(const Eigen::Quaterniond& quat);

    /**
     * @brief Create color message
     * @param r Red component (0-1)
     * @param g Green component (0-1)
     * @param b Blue component (0-1)
     * @param a Alpha component (0-1)
     * @return Color message
     */
    static std_msgs::msg::ColorRGBA createColor(float r, float g, float b, float a = 1.0f);

    /**
     * @brief Create color from array
     * @param rgb RGB array
     * @param a Alpha component
     * @return Color message
     */
    static std_msgs::msg::ColorRGBA createColor(const std::array<float, 3>& rgb, float a = 1.0f);

    /**
     * @brief Generate rainbow color based on value
     * @param value Value between 0 and 1
     * @return RGB color array
     */
    static std::array<float, 3> getRainbowColor(double value);

    /**
     * @brief Create cylinder marker
     * @param position Position
     * @param height Cylinder height
     * @param radius Cylinder radius
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createCylinder(
        const Eigen::Vector3d& position,
        double height,
        double radius,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create sphere marker
     * @param position Position
     * @param radius Sphere radius
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createSphere(
        const Eigen::Vector3d& position,
        double radius,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create arrow marker
     * @param start Start position
     * @param end End position
     * @param shaft_diameter Arrow shaft diameter
     * @param head_diameter Arrow head diameter
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createArrow(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        double shaft_diameter,
        double head_diameter,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create line list marker
     * @param points Line points (pairs of start/end points)
     * @param width Line width
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createLineList(
        const std::vector<Eigen::Vector3d>& points,
        double width,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create line strip marker
     * @param points Connected line points
     * @param width Line width
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createLineStrip(
        const std::vector<Eigen::Vector3d>& points,
        double width,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create text marker
     * @param position Position
     * @param text Text content
     * @param height Text height
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createText(
        const Eigen::Vector3d& position,
        const std::string& text,
        double height,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create covariance ellipsoid marker
     * @param mean Mean position
     * @param covariance 3x3 covariance matrix
     * @param scale Scale factor for visualization
     * @param color Color
     * @return Marker message
     */
    static visualization_msgs::msg::Marker createCovarianceEllipsoid(
        const Eigen::Vector3d& mean,
        const Eigen::Matrix3d& covariance,
        double scale,
        const std_msgs::msg::ColorRGBA& color);

    /**
     * @brief Create delete all marker
     * @return Marker message to delete all markers
     */
    static visualization_msgs::msg::Marker createDeleteAllMarker();

    /**
     * @brief Interpolate between two colors
     * @param color1 First color
     * @param color2 Second color
     * @param t Interpolation parameter (0-1)
     * @return Interpolated color
     */
    static std_msgs::msg::ColorRGBA interpolateColor(
        const std_msgs::msg::ColorRGBA& color1,
        const std_msgs::msg::ColorRGBA& color2,
        double t);

    /**
     * @brief Convert HSV to RGB color
     * @param h Hue (0-360)
     * @param s Saturation (0-1)
     * @param v Value (0-1)
     * @return RGB color array
     */
    static std::array<float, 3> hsvToRgb(double h, double s, double v);
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_VISUALIZATION_UTILS_HPP