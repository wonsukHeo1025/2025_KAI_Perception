#pragma once
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <unordered_set>

namespace LIDAR {

class DBSCANClusterer {
public:
    DBSCANClusterer(float eps, int min_points);
    
    void setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void setSearchMethod(pcl::search::KdTree<pcl::PointXYZI>::Ptr tree);
    void extract(std::vector<pcl::PointIndices>& cluster_indices);
    
private:
    float eps_;
    int min_points_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree_;
    
    std::vector<int> labels_;  // -1: unvisited, -2: noise, 0+: cluster id
    
    void expandCluster(int point_idx, int cluster_id, 
                       const std::vector<int>& neighbors);
    std::vector<int> regionQuery(int point_idx);
};

} // namespace LIDAR