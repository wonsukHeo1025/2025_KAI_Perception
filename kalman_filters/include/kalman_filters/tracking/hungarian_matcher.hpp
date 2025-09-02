#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include <unordered_map>
#include <Eigen/Core>
#include "tracking_types.hpp"

namespace kalman_filters {
namespace tracking {

// Forward declaration
class UKFTrack;

/**
 * @brief Hungarian algorithm implementation for data association
 * 
 * This class provides optimal assignment between detections and tracks
 * using the Hungarian algorithm (also known as Kuhn-Munkres algorithm).
 * 
 * Note: This implementation requires dlib library for the core algorithm.
 * If dlib is not available, the library will fall back to greedy matching.
 */
class HungarianMatcher {
public:
    HungarianMatcher() = default;
    ~HungarianMatcher() = default;
    
    /**
     * @brief Perform Hungarian matching on cost matrix
     * @param cost_matrix Distance/cost matrix (rows: detections, cols: tracks)
     * @param max_distance Maximum allowed matching distance
     * @return Association result with matched pairs and unmatched indices
     */
    AssociationResult match(const Eigen::MatrixXd& cost_matrix, 
                           double max_distance = 50.0);
    
    /**
     * @brief Compute cost matrix between 2D points
     * @param detections Vector of 2D detection points
     * @param tracks Vector of 2D track points
     * @return Cost matrix with Euclidean distances
     */
    static Eigen::MatrixXd computeCostMatrix(
        const std::vector<std::pair<double, double>>& detections,
        const std::vector<std::pair<double, double>>& tracks);
    
    /**
     * @brief Create association function for MultiTracker
     * @return Function that can be used with MultiTracker::setAssociationFunction
     */
    static std::function<AssociationResult(
        const std::vector<Detection>&,
        const std::unordered_map<int, std::shared_ptr<UKFTrack>>&,
        double)> createAssociationFunction();
    
private:
    /**
     * @brief Internal Hungarian algorithm implementation
     * @param cost_matrix Input cost matrix
     * @return Vector of assignments (row_idx -> col_idx)
     */
    std::vector<int> solveHungarian(const Eigen::MatrixXd& cost_matrix);
    
    /**
     * @brief Filter matches based on maximum distance threshold
     * @param assignments Raw assignments from Hungarian algorithm
     * @param cost_matrix Original cost matrix
     * @param max_distance Maximum allowed distance
     * @return Filtered association result
     */
    AssociationResult filterMatches(const std::vector<int>& assignments,
                                   const Eigen::MatrixXd& cost_matrix,
                                   double max_distance);
                                   
private:
    double max_matching_distance_ = 50.0;  // Default max distance
};

} // namespace tracking
} // namespace kalman_filters