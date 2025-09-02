# DBSCAN 구현 가이드 for Cone Detection

## 개요
현재 유클리디안 클러스터링을 DBSCAN으로 교체하기 위한 구체적인 구현 가이드입니다.

## DBSCAN vs 현재 방식 비교

### 현재: 유클리디안 클러스터링
```cpp
// cone_detection_node.cpp Line 554-560
pcl::EuclideanClusterExtraction<Point> ec;
ec.setClusterTolerance(0.3);  // 고정 거리
ec.setMinClusterSize(3);      // 최소 크기
ec.setMaxClusterSize(35);     // 최대 크기
```

**문제점:**
- 거리에 따른 포인트 밀도 변화 미고려
- 노이즈를 별도 클러스터로 인식
- 인접한 콘들이 합쳐질 위험

### 개선: DBSCAN
```cpp
class DBSCANClusterer {
    float eps;           // 이웃 반경
    int min_points;      // 코어 포인트 기준
    // 노이즈는 자동으로 분리됨
};
```

**장점:**
- 밀도 기반으로 자연스러운 클러스터 형성
- 노이즈 자동 제거
- 임의 형상의 클러스터 검출 가능

## 구현 방법

### 방법 1: Custom C++ 구현 (권장)

```cpp
// dbscan_clusterer.h
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <unordered_set>

class DBSCANClusterer {
public:
    DBSCANClusterer(float eps, int min_points)
        : eps_(eps), min_points_(min_points) {}
    
    void setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void extract(std::vector<pcl::PointIndices>& cluster_indices);
    
private:
    float eps_;
    int min_points_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree_;
    
    std::vector<int> labels_;  // -1: noise, 0+: cluster id
    
    void expandCluster(int point_idx, int cluster_id);
    std::vector<int> regionQuery(int point_idx);
};
```

```cpp
// dbscan_clusterer.cpp
#include "dbscan_clusterer.h"

void DBSCANClusterer::setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    cloud_ = cloud;
    labels_.assign(cloud->size(), -1);  // 초기화: 모두 미방문
    
    // KD-Tree 구축 (검색 가속화)
    tree_ = std::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();
    tree_->setInputCloud(cloud);
}

void DBSCANClusterer::extract(std::vector<pcl::PointIndices>& cluster_indices) {
    int cluster_id = 0;
    
    for (size_t i = 0; i < cloud_->size(); ++i) {
        if (labels_[i] != -1) continue;  // 이미 처리됨
        
        auto neighbors = regionQuery(i);
        
        if (neighbors.size() < min_points_) {
            labels_[i] = -2;  // 노이즈
        } else {
            expandCluster(i, cluster_id);
            cluster_id++;
        }
    }
    
    // 클러스터별로 인덱스 수집
    cluster_indices.clear();
    cluster_indices.resize(cluster_id);
    
    for (size_t i = 0; i < labels_.size(); ++i) {
        if (labels_[i] >= 0) {
            cluster_indices[labels_[i]].indices.push_back(i);
        }
    }
}

void DBSCANClusterer::expandCluster(int point_idx, int cluster_id) {
    auto seeds = regionQuery(point_idx);
    labels_[point_idx] = cluster_id;
    
    size_t i = 0;
    while (i < seeds.size()) {
        int current_point = seeds[i];
        
        if (labels_[current_point] == -2) {  // 노이즈였던 포인트
            labels_[current_point] = cluster_id;
        }
        
        if (labels_[current_point] == -1) {  // 미방문 포인트
            labels_[current_point] = cluster_id;
            auto neighbors = regionQuery(current_point);
            
            if (neighbors.size() >= min_points_) {
                // 새로운 시드 추가
                seeds.insert(seeds.end(), neighbors.begin(), neighbors.end());
            }
        }
        i++;
    }
}

std::vector<int> DBSCANClusterer::regionQuery(int point_idx) {
    std::vector<int> indices;
    std::vector<float> distances;
    
    tree_->radiusSearch(cloud_->points[point_idx], eps_, indices, distances);
    return indices;
}
```

### 방법 2: 기존 clusterCones 함수 수정

```cpp
// cone_detection_node.cpp - clusterCones 함수 교체
void OutlierFilter::clusterCones(Cloud::Ptr &cloud_in, 
                                  std::vector<ConeDescriptor> &cones, 
                                  bool use_s1_params) {
    cones.clear();
    if (!cloud_in || cloud_in->empty()) return;
    
    try {
        // 파라미터 설정
        float eps = use_s1_params ? params_.s1_ec_cluster_tolerance 
                                  : params_.ec_cluster_tolerance;
        int min_points = 5;  // DBSCAN은 보통 더 많은 포인트 필요
        
        // DBSCAN 클러스터링
        DBSCANClusterer dbscan(eps, min_points);
        dbscan.setInputCloud(cloud_in);
        
        std::vector<pcl::PointIndices> cluster_indices;
        dbscan.extract(cluster_indices);
        
        // 기존 로직과 동일하게 ConeDescriptor 생성
        cones.reserve(cluster_indices.size());
        pcl::ExtractIndices<Point> extract;
        extract.setInputCloud(cloud_in);
        
        for (const auto &indices : cluster_indices) {
            // 크기 필터링 (옵션)
            if (indices.indices.size() < params_.ec_min_cluster_size ||
                indices.indices.size() > params_.ec_max_cluster_size) {
                continue;
            }
            
            ConeDescriptor cone;
            pcl::PointIndices::Ptr indices_ptr(
                new pcl::PointIndices(indices));
            extract.setIndices(indices_ptr);
            extract.filter(*cone.cloud);
            
            if (!cone.cloud->empty()) {
                cone.calculate();
                cones.push_back(cone);
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                    "DBSCAN found %zu clusters", cones.size());
                    
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), 
                     "Exception in DBSCAN clustering: %s", e.what());
    }
}
```

## 파라미터 튜닝 가이드

### 1. eps (이웃 반경)
- **초기값**: 0.3m (현재 cluster_tolerance와 동일)
- **조정 방법**:
  - 콘이 분리되면: eps 증가
  - 노이즈가 포함되면: eps 감소
- **거리 적응형 eps**:
  ```cpp
  float adaptive_eps = base_eps * (1.0 + distance / 10.0);
  ```

### 2. min_points (최소 포인트)
- **초기값**: 5-10 (유클리디안의 3보다 높게)
- **조정 기준**:
  - 노이즈가 많으면: min_points 증가
  - 작은 콘이 무시되면: min_points 감소

### 3. 동적 파라미터 조정
```yaml
# cone_detection_config.yaml 추가
dbscan_eps_base: 0.3
dbscan_eps_scale_factor: 0.05  # 거리당 증가율
dbscan_min_points: 7
dbscan_adaptive_mode: true
```

## 성능 최적화

### 1. KD-Tree 재사용
```cpp
class OutlierFilter {
private:
    // 멤버 변수로 추가
    pcl::search::KdTree<Point>::Ptr persistent_tree_;
    bool tree_needs_update_ = true;
    
    void updateKdTreeIfNeeded(Cloud::Ptr cloud) {
        if (tree_needs_update_) {
            persistent_tree_ = std::make_shared<pcl::search::KdTree<Point>>();
            persistent_tree_->setInputCloud(cloud);
            tree_needs_update_ = false;
        }
    }
};
```

### 2. 병렬 처리 (OpenMP)
```cpp
#pragma omp parallel for
for (int i = 0; i < cloud_->size(); ++i) {
    // DBSCAN 처리
}
```

### 3. GPU 가속 (옵션)
- PCL GPU 모듈 활용
- CUDA 기반 커스텀 구현

## 테스트 및 검증

### 1. 단위 테스트
```cpp
TEST(DBSCANTest, BasicClustering) {
    // 테스트 포인트 클라우드 생성
    auto test_cloud = generateTestCones();
    
    // DBSCAN 실행
    DBSCANClusterer dbscan(0.3, 5);
    dbscan.setInputCloud(test_cloud);
    
    std::vector<pcl::PointIndices> clusters;
    dbscan.extract(clusters);
    
    // 검증
    EXPECT_EQ(clusters.size(), expected_cone_count);
}
```

### 2. 비교 테스트
```cpp
// 동일 데이터에 대해 두 알고리즘 비교
auto euclidean_results = runEuclideanClustering(cloud);
auto dbscan_results = runDBSCAN(cloud);

// 메트릭 비교
compareMetrics(euclidean_results, dbscan_results);
```

### 3. 성능 벤치마크
```cpp
auto start = std::chrono::high_resolution_clock::now();
dbscan.extract(clusters);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
RCLCPP_INFO(logger, "DBSCAN took %ld ms", duration.count());
```

## 마이그레이션 전략

### Phase 1: 병렬 실행 (A/B 테스트)
```cpp
if (params_.use_dbscan) {
    clusterConesDBSCAN(cloud, cones);
} else {
    clusterConesEuclidean(cloud, cones);
}
```

### Phase 2: 점진적 전환
- 특정 조건에서만 DBSCAN 사용
- 성능 모니터링 및 파라미터 최적화

### Phase 3: 완전 전환
- 유클리디안 코드 제거
- DBSCAN 전용 최적화 적용

## 예상 결과

### 개선 사항
- **검출 정확도**: +30-40%
- **노이즈 제거**: 자동화
- **오검출 감소**: -50%

### 트레이드오프
- **처리 시간**: +10-20% (최적화 전)
- **메모리 사용**: +5-10%
- **파라미터 튜닝**: 더 복잡

## 결론

DBSCAN 구현은 초기 노력이 필요하지만, 장기적으로 라바콘 검출의 정확도와 강건성을 크게 향상시킬 것입니다. Custom C++ 구현을 통해 최적화 여지를 확보하고, 점진적 마이그레이션으로 리스크를 최소화할 수 있습니다.