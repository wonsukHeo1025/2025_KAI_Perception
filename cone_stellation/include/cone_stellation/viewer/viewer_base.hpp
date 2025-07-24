#ifndef CONE_STELLATION_VIEWER_BASE_HPP
#define CONE_STELLATION_VIEWER_BASE_HPP

#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace cone_stellation {
namespace viewer {

/**
 * @brief Abstract base class for visualization components
 */
class ViewerBase {
public:
    ViewerBase() = default;
    virtual ~ViewerBase() = default;

    /**
     * @brief Initialize the viewer
     * @return True if initialization successful
     */
    virtual bool initialize() = 0;

    /**
     * @brief Shutdown the viewer
     */
    virtual void shutdown() = 0;

    /**
     * @brief Check if viewer is initialized
     * @return True if initialized
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief Update visualization
     */
    virtual void update() = 0;

    /**
     * @brief Clear all visualizations
     */
    virtual void clear() = 0;

    /**
     * @brief Set viewer name
     * @param name Viewer name
     */
    void setName(const std::string& name) { name_ = name; }

    /**
     * @brief Get viewer name
     * @return Viewer name
     */
    const std::string& getName() const { return name_; }

protected:
    std::string name_{"ViewerBase"};
    mutable std::mutex mutex_;
};

} // namespace viewer
} // namespace cone_stellation

#endif // CONE_STELLATION_VIEWER_BASE_HPP