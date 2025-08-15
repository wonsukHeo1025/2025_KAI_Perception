#include "cone_detection/dbscan_clusterer.h"
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <queue>

namespace LIDAR {

DBSCANClusterer::DBSCANClusterer(float eps, int min_points) 
    : eps_(eps), min_points_(min_points) {
}

void DBSCANClusterer::setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    cloud_ = cloud;
    labels_.clear();
    labels_.resize(cloud_->points.size(), -1); // -1 means unvisited
}

void DBSCANClusterer::setSearchMethod(pcl::search::KdTree<pcl::PointXYZI>::Ptr tree) {
    tree_ = tree;
}

void DBSCANClusterer::extract(std::vector<pcl::PointIndices>& cluster_indices) {
    if (!cloud_ || !tree_) {
        return;
    }
    
    cluster_indices.clear();
    int cluster_id = 0;
    
    // Process each point
    for (size_t i = 0; i < cloud_->points.size(); ++i) {
        // Skip already processed points
        if (labels_[i] != -1) {
            continue;
        }
        
        // Get neighbors within epsilon distance
        std::vector<int> neighbors = regionQuery(i);
        
        // If not enough neighbors, mark as noise
        if (static_cast<int>(neighbors.size()) < min_points_) {
            labels_[i] = -2; // noise
            continue;
        }
        
        // Create new cluster
        labels_[i] = cluster_id;
        expandCluster(i, cluster_id, neighbors);
        cluster_id++;
    }
    
    // Convert labels to pcl::PointIndices format
    std::vector<std::vector<int>> clusters(cluster_id);
    
    for (size_t i = 0; i < labels_.size(); ++i) {
        int label = labels_[i];
        if (label >= 0) { // Valid cluster (not noise)
            clusters[label].push_back(i);
        }
    }
    
    // Convert to pcl::PointIndices
    for (const auto& cluster : clusters) {
        if (!cluster.empty()) {
            pcl::PointIndices indices;
            indices.indices = cluster;
            cluster_indices.push_back(indices);
        }
    }
}

void DBSCANClusterer::expandCluster(int point_idx, int cluster_id, 
                                    const std::vector<int>& neighbors) {
    std::queue<int> to_process;
    
    // Add all neighbors to processing queue
    for (int neighbor_idx : neighbors) {
        to_process.push(neighbor_idx);
    }
    
    while (!to_process.empty()) {
        int current_idx = to_process.front();
        to_process.pop();
        
        // Skip if already processed
        if (labels_[current_idx] >= 0) {
            continue;
        }
        
        // Add to current cluster
        labels_[current_idx] = cluster_id;
        
        // Get neighbors of current point
        std::vector<int> current_neighbors = regionQuery(current_idx);
        
        // If current point has enough neighbors, add them to queue
        if (static_cast<int>(current_neighbors.size()) >= min_points_) {
            for (int neighbor_idx : current_neighbors) {
                // Add unvisited or noise points to queue
                if (labels_[neighbor_idx] == -1 || labels_[neighbor_idx] == -2) {
                    to_process.push(neighbor_idx);
                }
            }
        }
    }
}

std::vector<int> DBSCANClusterer::regionQuery(int point_idx) {
    std::vector<int> neighbors;
    std::vector<float> distances;
    
    // Use KdTree for efficient radius search
    tree_->radiusSearch(point_idx, eps_, neighbors, distances);
    
    return neighbors;
}

} // namespace LIDAR