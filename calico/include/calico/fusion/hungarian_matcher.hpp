#ifndef CALICO_FUSION_HUNGARIAN_MATCHER_HPP
#define CALICO_FUSION_HUNGARIAN_MATCHER_HPP

#include <vector>
#include <utility>
#include <Eigen/Core>

namespace calico {
namespace fusion {

/**
 * @brief Result structure for Hungarian matching
 */
struct MatchResult {
    std::vector<std::pair<int, int>> matches;     // (detection_idx, track_idx) pairs
    std::vector<int> unmatched_detections;        // Indices of unmatched detections
    std::vector<int> unmatched_tracks;            // Indices of unmatched tracks
};

/**
 * @brief Hungarian algorithm implementation for data association
 * 
 * This class provides optimal assignment between detections and tracks
 * using the Hungarian algorithm (also known as Kuhn-Munkres algorithm)
 */
class HungarianMatcher {
public:
    HungarianMatcher() = default;
    ~HungarianMatcher() = default;
    
    /**
     * @brief Perform Hungarian matching on cost matrix
     * @param cost_matrix Distance/cost matrix (rows: detections, cols: tracks)
     * @param max_distance Maximum allowed matching distance
     * @return Matching result with matched pairs and unmatched indices
     */
    MatchResult match(const Eigen::MatrixXd& cost_matrix, 
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
    
private:
    /**
     * @brief Internal Hungarian algorithm implementation using dlib
     * @param cost_matrix Input cost matrix
     * @return Vector of assignments (row_idx -> col_idx)
     */
    std::vector<int> solveHungarian(const Eigen::MatrixXd& cost_matrix);
    
    
    /**
     * @brief Filter matches based on maximum distance threshold
     * @param assignments Raw assignments from Hungarian algorithm
     * @param cost_matrix Original cost matrix
     * @param max_distance Maximum allowed distance
     * @return Filtered match result
     */
    MatchResult filterMatches(const std::vector<int>& assignments,
                             const Eigen::MatrixXd& cost_matrix,
                             double max_distance);
                             
private:
    double max_matching_distance_ = 50.0;  // Default max distance
};

} // namespace fusion
} // namespace calico

#endif // CALICO_FUSION_HUNGARIAN_MATCHER_HPP