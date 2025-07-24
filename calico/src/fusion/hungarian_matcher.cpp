#include "calico/fusion/hungarian_matcher.hpp"
#include <kalman_filters/tracking/hungarian_matcher.hpp>
#include <kalman_filters/tracking/tracking_types.hpp>
#include <limits>
#include <algorithm>
#include <cmath>

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
    
    // Use kalman_filters Hungarian matcher
    kalman_filters::tracking::HungarianMatcher kf_matcher;
    kalman_filters::tracking::AssociationResult kf_result;
    
    try {
        kf_result = kf_matcher.match(cost_matrix, max_distance);
    } catch (const std::exception& e) {
        // If kalman_filters matcher fails, return empty result
        for (int i = 0; i < cost_matrix.rows(); ++i) {
            result.unmatched_detections.push_back(i);
        }
        for (int j = 0; j < cost_matrix.cols(); ++j) {
            result.unmatched_tracks.push_back(j);
        }
        return result;
    }
    
    // Convert kalman_filters::tracking::AssociationResult to calico::fusion::MatchResult
    result.matches.reserve(kf_result.matches.size());
    for (const auto& match : kf_result.matches) {
        result.matches.push_back(match);
    }
    
    result.unmatched_detections.reserve(kf_result.unmatched_detections.size());
    for (int idx : kf_result.unmatched_detections) {
        result.unmatched_detections.push_back(idx);
    }
    
    result.unmatched_tracks.reserve(kf_result.unmatched_tracks.size());
    for (int idx : kf_result.unmatched_tracks) {
        result.unmatched_tracks.push_back(idx);
    }
    
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

std::vector<int> HungarianMatcher::solveHungarian(const Eigen::MatrixXd& /*cost_matrix*/) {
    // This is now handled internally by kalman_filters::tracking::HungarianMatcher
    // We don't need to expose this anymore
    throw std::runtime_error("solveHungarian should not be called directly. Use match() instead.");
}

MatchResult HungarianMatcher::filterMatches(const std::vector<int>& /*assignments*/,
                                           const Eigen::MatrixXd& /*cost_matrix*/,
                                           double /*max_distance*/) {
    // This is now handled internally by kalman_filters::tracking::HungarianMatcher
    // We don't need to expose this anymore
    throw std::runtime_error("filterMatches should not be called directly. Use match() instead.");
}

} // namespace fusion
} // namespace calico