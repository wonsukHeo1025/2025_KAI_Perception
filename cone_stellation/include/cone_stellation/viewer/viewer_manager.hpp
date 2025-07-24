#ifndef CONE_STELLATION_VIEWER_MANAGER_HPP
#define CONE_STELLATION_VIEWER_MANAGER_HPP

#include "cone_stellation/viewer/viewer_base.hpp"
#include "cone_stellation/viewer/cone_viewer.hpp"
#include "cone_stellation/viewer/track_viewer.hpp"
#include "cone_stellation/viewer/optimization_viewer.hpp"
#include "cone_stellation/viewer/pose_viewer.hpp"
#include <memory>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Manager class for all visualization components
 */
class ViewerManager {
public:
    ViewerManager(rclcpp::Node* node);
    ~ViewerManager();

    /**
     * @brief Initialize all viewers
     * @return True if all viewers initialized successfully
     */
    bool initialize();

    /**
     * @brief Shutdown all viewers
     */
    void shutdown();

    /**
     * @brief Update all active viewers
     */
    void updateAll();

    /**
     * @brief Clear all visualizations
     */
    void clearAll();

    /**
     * @brief Get cone viewer
     * @return Shared pointer to cone viewer
     */
    std::shared_ptr<ConeViewer> getConeViewer() { return cone_viewer_; }

    /**
     * @brief Get track viewer
     * @return Shared pointer to track viewer
     */
    std::shared_ptr<TrackViewer> getTrackViewer() { return track_viewer_; }

    /**
     * @brief Get optimization viewer
     * @return Shared pointer to optimization viewer
     */
    std::shared_ptr<OptimizationViewer> getOptimizationViewer() { return optimization_viewer_; }

    /**
     * @brief Get pose viewer
     * @return Shared pointer to pose viewer
     */
    std::shared_ptr<PoseViewer> getPoseViewer() { return pose_viewer_; }

    /**
     * @brief Enable/disable specific viewer
     * @param viewer_name Name of the viewer
     * @param enable Enable flag
     */
    void enableViewer(const std::string& viewer_name, bool enable);

    /**
     * @brief Check if viewer is enabled
     * @param viewer_name Name of the viewer
     * @return True if enabled
     */
    bool isViewerEnabled(const std::string& viewer_name) const;

    /**
     * @brief Set global frame ID for all viewers
     * @param frame_id Frame ID
     */
    void setGlobalFrameId(const std::string& frame_id);

    /**
     * @brief Set update rate for periodic updates
     * @param rate_hz Update rate in Hz
     */
    void setUpdateRate(double rate_hz);

    /**
     * @brief Start periodic update timer
     */
    void startPeriodicUpdate();

    /**
     * @brief Stop periodic update timer
     */
    void stopPeriodicUpdate();

private:
    /**
     * @brief Timer callback for periodic updates
     */
    void timerCallback();

    rclcpp::Node* node_;
    
    // Individual viewers
    std::shared_ptr<ConeViewer> cone_viewer_;
    std::shared_ptr<TrackViewer> track_viewer_;
    std::shared_ptr<OptimizationViewer> optimization_viewer_;
    std::shared_ptr<PoseViewer> pose_viewer_;
    
    // Viewer registry
    std::unordered_map<std::string, std::shared_ptr<ViewerBase>> viewers_;
    std::unordered_map<std::string, bool> viewer_enabled_;
    
    // Update timer
    rclcpp::TimerBase::SharedPtr update_timer_;
    double update_rate_hz_{30.0};
    
    bool initialized_{false};
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_VIEWER_MANAGER_HPP