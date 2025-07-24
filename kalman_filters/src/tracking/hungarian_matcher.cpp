#include "kalman_filters/tracking/hungarian_matcher.hpp"
#include "kalman_filters/tracking/ukf_track.hpp"
#include <limits>
#include <algorithm>
#include <cmath>
#include <set>
#include <queue>

// No external dependencies - using our own Hungarian implementation

namespace kalman_filters {
namespace tracking {

AssociationResult HungarianMatcher::match(const Eigen::MatrixXd& cost_matrix, double max_distance) {
    AssociationResult result;
    
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
    
    // Make the cost matrix square by padding if necessary
    int n = std::max(n_rows, n_cols);
    Eigen::MatrixXd square_cost(n, n);
    square_cost.setConstant(max_matching_distance_ * 10);  // High cost for dummy elements
    square_cost.block(0, 0, n_rows, n_cols) = cost_matrix;
    
    // Hungarian algorithm implementation
    Eigen::MatrixXd cost = square_cost;
    
    // Step 1: Subtract row minima
    for (int i = 0; i < n; ++i) {
        double row_min = cost.row(i).minCoeff();
        if (row_min != std::numeric_limits<double>::infinity()) {
            cost.row(i).array() -= row_min;
        }
    }
    
    // Step 2: Subtract column minima
    for (int j = 0; j < n; ++j) {
        double col_min = cost.col(j).minCoeff();
        if (col_min != std::numeric_limits<double>::infinity()) {
            cost.col(j).array() -= col_min;
        }
    }
    
    // Initialize assignments
    std::vector<int> row_assignment(n, -1);
    std::vector<int> col_assignment(n, -1);
    
    // Augmenting path algorithm
    bool done = false;
    while (!done) {
        // Find zeros and try to make assignments
        std::vector<bool> row_covered(n, false);
        std::vector<bool> col_covered(n, false);
        
        // Try to find an assignment for each row
        for (int iter = 0; iter < n; ++iter) {
            // Find uncovered zeros
            std::vector<std::pair<int, int>> zeros;
            for (int i = 0; i < n; ++i) {
                if (row_covered[i]) continue;
                for (int j = 0; j < n; ++j) {
                    if (col_covered[j]) continue;
                    if (std::abs(cost(i, j)) < 1e-10) {  // Effectively zero
                        zeros.emplace_back(i, j);
                    }
                }
            }
            
            if (zeros.empty()) {
                // Need to create more zeros
                double min_uncovered = std::numeric_limits<double>::max();
                
                // Find minimum uncovered value
                for (int i = 0; i < n; ++i) {
                    if (row_covered[i]) continue;
                    for (int j = 0; j < n; ++j) {
                        if (col_covered[j]) continue;
                        min_uncovered = std::min(min_uncovered, cost(i, j));
                    }
                }
                
                // Subtract from uncovered elements
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (!row_covered[i] && !col_covered[j]) {
                            cost(i, j) -= min_uncovered;
                        } else if (row_covered[i] && col_covered[j]) {
                            cost(i, j) += min_uncovered;
                        }
                    }
                }
            } else {
                // Make an assignment
                for (const auto& [i, j] : zeros) {
                    if (!row_covered[i] && !col_covered[j]) {
                        row_assignment[i] = j;
                        col_assignment[j] = i;
                        row_covered[i] = true;
                        col_covered[j] = true;
                        break;
                    }
                }
            }
        }
        
        // Check if we have a complete assignment
        done = true;
        for (int i = 0; i < n; ++i) {
            if (row_assignment[i] == -1) {
                // Try to find an augmenting path
                std::vector<bool> visited(n, false);
                std::vector<int> path_col(n, -1);
                
                // BFS to find augmenting path
                std::queue<int> q;
                q.push(i);
                visited[i] = true;
                
                int found_col = -1;
                while (!q.empty() && found_col == -1) {
                    int curr_row = q.front();
                    q.pop();
                    
                    for (int j = 0; j < n; ++j) {
                        if (std::abs(cost(curr_row, j)) < 1e-10) {
                            if (col_assignment[j] == -1) {
                                found_col = j;
                                path_col[curr_row] = j;
                                break;
                            } else if (col_assignment[j] != curr_row) {
                                int next_row = col_assignment[j];
                                if (!visited[next_row]) {
                                    visited[next_row] = true;
                                    path_col[curr_row] = j;
                                    q.push(next_row);
                                }
                            }
                        }
                    }
                }
                
                if (found_col != -1) {
                    // Augment along the path
                    int curr_row = i;
                    while (curr_row != -1) {
                        int j = path_col[curr_row];
                        int next_row = col_assignment[j];
                        row_assignment[curr_row] = j;
                        col_assignment[j] = curr_row;
                        curr_row = next_row;
                    }
                    done = false;
                    break;
                }
            }
        }
        
        // If no augmenting path found but not all assigned, create more zeros
        if (done) {
            for (int i = 0; i < n; ++i) {
                if (row_assignment[i] == -1) {
                    done = false;
                    break;
                }
            }
        }
    }
    
    // Extract assignments for original rows
    std::vector<int> assignments(n_rows, -1);
    for (int i = 0; i < n_rows; ++i) {
        if (row_assignment[i] >= 0 && row_assignment[i] < n_cols) {
            // Check if the assignment cost is within threshold
            if (cost_matrix(i, row_assignment[i]) <= max_matching_distance_) {
                assignments[i] = row_assignment[i];
            }
        }
    }
    
    return assignments;
}

AssociationResult HungarianMatcher::filterMatches(const std::vector<int>& assignments,
                                                  const Eigen::MatrixXd& cost_matrix,
                                                  double max_distance) {
    AssociationResult result;
    
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

// Static factory function for creating association function
std::function<AssociationResult(
    const std::vector<Detection>&,
    const std::unordered_map<int, std::shared_ptr<UKFTrack>>&,
    double)> 
HungarianMatcher::createAssociationFunction() {
    
    return [](const std::vector<Detection>& detections,
              const std::unordered_map<int, std::shared_ptr<UKFTrack>>& tracks,
              double max_distance) -> AssociationResult {
        
        // Extract positions for cost matrix computation
        std::vector<std::pair<double, double>> detection_positions;
        for (const auto& det : detections) {
            detection_positions.emplace_back(det.x, det.y);
        }
        
        std::vector<std::pair<double, double>> track_positions;
        std::vector<int> track_ids;
        for (const auto& [id, track] : tracks) {
            auto pos = track->getPosition();
            track_positions.emplace_back(pos(0), pos(1));
            track_ids.push_back(id);
        }
        
        if (detection_positions.empty() || track_positions.empty()) {
            // No association possible
            AssociationResult result;
            for (size_t i = 0; i < detections.size(); ++i) {
                result.unmatched_detections.push_back(i);
            }
            for (const auto& [id, track] : tracks) {
                result.unmatched_tracks.push_back(id);
            }
            return result;
        }
        
        // Compute cost matrix
        auto cost_matrix = HungarianMatcher::computeCostMatrix(
            detection_positions, track_positions);
        
        // Perform Hungarian matching
        HungarianMatcher matcher;
        auto match_result = matcher.match(cost_matrix, max_distance);
        
        // Convert track indices to track IDs
        AssociationResult final_result;
        for (const auto& [det_idx, track_idx] : match_result.matches) {
            if (track_idx < track_ids.size()) {
                final_result.matches.emplace_back(det_idx, track_ids[track_idx]);
            }
        }
        final_result.unmatched_detections = match_result.unmatched_detections;
        
        // Convert unmatched track indices to IDs
        for (int track_idx : match_result.unmatched_tracks) {
            if (track_idx < track_ids.size()) {
                final_result.unmatched_tracks.push_back(track_ids[track_idx]);
            }
        }
        
        return final_result;
    };
}

} // namespace tracking
} // namespace kalman_filters