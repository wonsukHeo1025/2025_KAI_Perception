#ifndef CALICO_NODES_MULTI_IOU_FUSION_NODE_HPP
#define CALICO_NODES_MULTI_IOU_FUSION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <memory>

namespace calico {
namespace nodes {

/**
 * @brief Multi-camera IoU-based sensor fusion node
 * 
 * This node performs sensor fusion between LiDAR 3D bounding boxes and 
 * YOLO detections from multiple cameras using IoU-based Hungarian matching.
 * 
 * Features:
 * - Supports multiple cameras (typically 2)
 * - Projects 3D LiDAR boxes to 2D camera planes
 * - Computes IoU between projected boxes and YOLO detections
 * - Uses Hungarian algorithm for optimal matching
 * - Merges classifications from multiple cameras
 * - Provides debug visualization for each camera
 */
class MultiIoUFusionNode : public rclcpp::Node
{
public:
    explicit MultiIoUFusionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~MultiIoUFusionNode() = default;
};

} // namespace nodes
} // namespace calico

#endif // CALICO_NODES_MULTI_IOU_FUSION_NODE_HPP