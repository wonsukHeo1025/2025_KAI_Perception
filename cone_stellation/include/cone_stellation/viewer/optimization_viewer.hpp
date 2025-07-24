#ifndef CONE_STELLATION_OPTIMIZATION_VIEWER_HPP
#define CONE_STELLATION_OPTIMIZATION_VIEWER_HPP

#include "cone_stellation/viewer/viewer_base.hpp"
#include <Eigen/Core>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Optimization process visualization
 */
class OptimizationViewer : public ViewerBase {
public:
    struct OptimizationStep {
        std::vector<Eigen::Vector3d> positions;
        std::vector<Eigen::Matrix3d> covariances;
        double cost;
        int iteration;
    };

    struct ConstraintVisualization {
        Eigen::Vector3d start;
        Eigen::Vector3d end;
        std::string type;  // "odometry", "loop_closure", "cone_observation"
        double weight;
    };

    OptimizationViewer(rclcpp::Node* node);
    ~OptimizationViewer() override;

    bool initialize() override;
    void shutdown() override;
    bool isInitialized() const override;
    void update() override;
    void clear() override;

    /**
     * @brief Add optimization step
     * @param step Optimization step data
     */
    void addOptimizationStep(const OptimizationStep& step);

    /**
     * @brief Set current optimization state
     * @param positions Current positions
     * @param covariances Current covariances
     */
    void setCurrentState(const std::vector<Eigen::Vector3d>& positions,
                        const std::vector<Eigen::Matrix3d>& covariances);

    /**
     * @brief Add constraint visualization
     * @param constraint Constraint data
     */
    void addConstraint(const ConstraintVisualization& constraint);

    /**
     * @brief Clear constraints
     */
    void clearConstraints();

    /**
     * @brief Enable/disable trajectory visualization
     * @param enable Enable flag
     */
    void enableTrajectoryVisualization(bool enable) { show_trajectory_ = enable; }

    /**
     * @brief Enable/disable covariance visualization
     * @param enable Enable flag
     */
    void enableCovarianceVisualization(bool enable) { show_covariances_ = enable; }

    /**
     * @brief Enable/disable constraint visualization
     * @param enable Enable flag
     */
    void enableConstraintVisualization(bool enable) { show_constraints_ = enable; }

    /**
     * @brief Set frame ID for visualization
     * @param frame_id Frame ID
     */
    void setFrameId(const std::string& frame_id) { frame_id_ = frame_id; }

private:
    /**
     * @brief Create trajectory marker
     * @param positions Trajectory positions
     * @param id Marker ID
     * @return Marker message
     */
    visualization_msgs::msg::Marker createTrajectoryMarker(
        const std::vector<Eigen::Vector3d>& positions, int id);

    /**
     * @brief Create covariance marker
     * @param position Position
     * @param covariance Covariance matrix
     * @param id Marker ID
     * @return Marker message
     */
    visualization_msgs::msg::Marker createCovarianceMarker(
        const Eigen::Vector3d& position,
        const Eigen::Matrix3d& covariance,
        int id);

    /**
     * @brief Create constraint marker
     * @param constraint Constraint data
     * @param id Marker ID
     * @return Marker message
     */
    visualization_msgs::msg::Marker createConstraintMarker(
        const ConstraintVisualization& constraint, int id);

    /**
     * @brief Get color for constraint type
     * @param type Constraint type
     * @return RGB color
     */
    std::array<float, 3> getConstraintColor(const std::string& type);

    rclcpp::Node* node_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    std::vector<OptimizationStep> optimization_history_;
    std::vector<Eigen::Vector3d> current_positions_;
    std::vector<Eigen::Matrix3d> current_covariances_;
    std::vector<ConstraintVisualization> constraints_;
    
    std::string frame_id_{"map"};
    bool initialized_{false};
    bool show_trajectory_{true};
    bool show_covariances_{true};
    bool show_constraints_{true};
    
    int marker_id_counter_{0};
    double covariance_scale_{3.0};  // Scale factor for covariance ellipsoids
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_OPTIMIZATION_VIEWER_HPP