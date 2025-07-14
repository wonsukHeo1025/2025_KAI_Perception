#include "calico/fusion/hungarian_matcher.hpp"
#include <limits>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <rclcpp/rclcpp.hpp>
#include <dlib/optimization/max_cost_assignment.h>

namespace calico {
namespace fusion {

MatchResult HungarianMatcher::match(const Eigen::MatrixXd& cost_matrix, double max_distance) {
    MatchResult result;
    
    // Store max distance for use in solver
    max_matching_distance_ = max_distance;
    
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        // Empty cost matrix - no matches possible
        for (int i = 0; i < cost_matrix.rows(); ++i) {
            result.unmatched_detections.push_back(i);
        }
        for (int j = 0; j < cost_matrix.cols(); ++j) {
            result.unmatched_tracks.push_back(j);
        }
        return result;
    }
    
    // Solve Hungarian algorithm
    std::vector<int> assignments = solveHungarian(cost_matrix);
    
    // Filter matches based on maximum distance threshold
    result = filterMatches(assignments, cost_matrix, max_distance);
    
    return result;
}

Eigen::MatrixXd HungarianMatcher::computeCostMatrix(
    const std::vector<std::pair<double, double>>& detections,
    const std::vector<std::pair<double, double>>& tracks) {
    
    size_t n_detections = detections.size();
    size_t n_tracks = tracks.size();
    
    Eigen::MatrixXd cost_matrix(n_detections, n_tracks);
    
    // Compute Euclidean distances
    for (size_t i = 0; i < n_detections; ++i) {
        for (size_t j = 0; j < n_tracks; ++j) {
            double dx = detections[i].first - tracks[j].first;
            double dy = detections[i].second - tracks[j].second;
            cost_matrix(i, j) = std::sqrt(dx * dx + dy * dy);
        }
    }
    
    return cost_matrix;
}

std::vector<int> HungarianMatcher::solveHungarian(const Eigen::MatrixXd& cost_matrix) {
    int n_rows = cost_matrix.rows();
    int n_cols = cost_matrix.cols();
    
    // Handle empty matrix
    if (n_rows == 0 || n_cols == 0) {
        return std::vector<int>(n_rows, -1);
    }
    
    // dlib requires square matrix or more columns than rows
    // If we have more rows than columns, we need to handle it differently
    if (n_rows > n_cols) {
        // Add dummy columns with high cost
        Eigen::MatrixXd padded_matrix(n_rows, n_rows);
        padded_matrix.leftCols(n_cols) = cost_matrix;
        padded_matrix.rightCols(n_rows - n_cols).setConstant(max_matching_distance_ * 10);
        
        // Solve with padded matrix (recursive call)
        auto padded_result = solveHungarian(padded_matrix);
        
        // Remove assignments to dummy columns
        std::vector<int> result(n_rows, -1);
        for (int i = 0; i < n_rows; ++i) {
            if (padded_result[i] >= 0 && padded_result[i] < n_cols) {
                result[i] = padded_result[i];
            }
        }
        return result;
    }
    
    // dlib uses max-cost assignment, so we need to negate the costs
    // First find the maximum value to ensure all values are positive after transformation
    double max_val = cost_matrix.maxCoeff();
    
    // Create dlib matrix (must use integer type)
    dlib::matrix<long> dlib_cost(n_rows, n_cols);
    
    // Scale factor for converting double to integer
    const long scale = 1000;
    
    // Convert Eigen matrix to dlib matrix with negated costs
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            // Transform min-cost to max-cost problem
            // Use max_val - cost to ensure positive values
            // Scale to integer
            double cost_normalized = max_val - cost_matrix(i, j);
            // Ensure non-negative values
            if (cost_normalized < 0) cost_normalized = 0;
            dlib_cost(i, j) = static_cast<long>(cost_normalized * scale);
        }
    }
    
    // Solve using dlib
    std::vector<long> dlib_assignments;
    try {
        // dlib::max_cost_assignment returns the assignment vector directly
        dlib_assignments = dlib::max_cost_assignment(dlib_cost);
        RCLCPP_DEBUG(rclcpp::get_logger("hungarian_matcher"),
                    "dlib assignment completed successfully for %dx%d matrix", n_rows, n_cols);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("hungarian_matcher"),
                    "dlib max_cost_assignment failed for %dx%d matrix: %s", 
                    n_rows, n_cols, e.what());
        return std::vector<int>(n_rows, -1);
    }
    
    // Convert back to our format
    std::vector<int> assignments(n_rows, -1);
    for (size_t i = 0; i < dlib_assignments.size() && i < static_cast<size_t>(n_rows); ++i) {
        if (dlib_assignments[i] >= 0 && dlib_assignments[i] < n_cols) {
            // Check if the original cost is within threshold
            if (cost_matrix(i, dlib_assignments[i]) <= max_matching_distance_) {
                assignments[i] = static_cast<int>(dlib_assignments[i]);
            }
        }
    }
    
    return assignments;
}

MatchResult HungarianMatcher::filterMatches(const std::vector<int>& assignments,
                                           const Eigen::MatrixXd& cost_matrix,
                                           double max_distance) {
    MatchResult result;
    
    // Track which detections and tracks were matched
    std::vector<bool> det_matched(cost_matrix.rows(), false);
    std::vector<bool> track_matched(cost_matrix.cols(), false);
    
    // Process assignments
    for (size_t i = 0; i < assignments.size(); ++i) {
        int j = assignments[i];
        
        if (j >= 0 && cost_matrix(i, j) < max_distance) {
            // Valid match within threshold
            result.matches.emplace_back(i, j);
            det_matched[i] = true;
            track_matched[j] = true;
        }
    }
    
    // Find unmatched detections
    for (size_t i = 0; i < det_matched.size(); ++i) {
        if (!det_matched[i]) {
            result.unmatched_detections.push_back(i);
        }
    }
    
    // Find unmatched tracks
    for (size_t j = 0; j < track_matched.size(); ++j) {
        if (!track_matched[j]) {
            result.unmatched_tracks.push_back(j);
        }
    }
    
    return result;
}

} // namespace fusion
} // namespace calico