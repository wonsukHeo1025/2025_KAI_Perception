# Cone Detection 성능 최적화 가이드

## 현재 성능 병목 지점

### 1. 주요 병목 현상
| 위치 | 문제점 | 영향도 | 개선 난이도 |
|------|---------|--------|------------|
| KdTree 생성 | 매 프레임 재생성 | High | Low |
| 메모리 복사 | 불필요한 복사 다수 | Medium | Low |
| Ouster 포맷 변환 | 수동 메모리 조작 | High | Low |
| Voxel 다운샘플링 | 고정 크기 사용 | Medium | Medium |
| 클러스터링 | 단일 스레드 처리 | High | Medium |

## 즉시 적용 가능한 최적화 (Quick Wins)

### 1. KdTree 재사용
**현재 코드 (매번 생성):**
```cpp
// clusterCones() - Line 550
pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
tree->setInputCloud(cloud_in);
```

**최적화 코드:**
```cpp
// 헤더 파일에 멤버 변수 추가
class OutlierFilter {
private:
    pcl::search::KdTree<Point>::Ptr persistent_tree_;
    size_t last_cloud_size_ = 0;
};

// 구현 파일
void OutlierFilter::clusterCones(...) {
    // KdTree 재사용 로직
    if (!persistent_tree_ || 
        cloud_in->size() != last_cloud_size_) {
        persistent_tree_ = std::make_shared<pcl::search::KdTree<Point>>();
        last_cloud_size_ = cloud_in->size();
    }
    persistent_tree_->setInputCloud(cloud_in);
}
```
**예상 개선:** 프레임당 5-10ms 절약

### 2. 불필요한 메모리 복사 제거
**현재 코드:**
```cpp
// filterPointCloud() - Line 458
*cloud_out = *current_filtered_cloud;  // 전체 복사
```

**최적화 코드:**
```cpp
// 포인터 스왑 사용
cloud_out = std::move(current_filtered_cloud);
// 또는
cloud_out.swap(current_filtered_cloud);
```
**예상 개선:** 프레임당 2-5ms 절약

### 3. Ouster 포맷 변환 제거
**제거할 코드:** Lines 646-743
```cpp
// 100+ 라인의 복잡한 변환 로직 전체 제거
if (publisher == pub_reconstructed_cones_cloud_) {
    // 삭제
}
// 단순히 pcl::toROSMsg 사용
pcl::toROSMsg(*cloud, cloud_msg);
```
**예상 개선:** 프레임당 10-15ms 절약

## 중기 최적화 방안

### 1. 병렬 처리 도입

#### OpenMP를 활용한 필터링 병렬화
```cpp
// CMakeLists.txt에 추가
find_package(OpenMP REQUIRED)
target_link_libraries(cone_detection_node ${OpenMP_CXX_LIBRARIES})

// filterPointCloud() 병렬화
#include <omp.h>

void OutlierFilter::filterPointCloud(Cloud::Ptr &cloud_in, 
                                      Cloud::Ptr &cloud_out) {
    // ROI 필터링 병렬화
    Cloud::Ptr roi_filtered_cloud(new Cloud);
    std::vector<Point> temp_points;
    temp_points.reserve(cloud_in->size());
    
    #pragma omp parallel
    {
        std::vector<Point> local_points;
        
        #pragma omp for nowait
        for (size_t i = 0; i < cloud_in->points.size(); ++i) {
            const auto& point = cloud_in->points[i];
            if (passesROIFilter(point)) {
                local_points.push_back(point);
            }
        }
        
        #pragma omp critical
        {
            temp_points.insert(temp_points.end(), 
                              local_points.begin(), 
                              local_points.end());
        }
    }
    
    roi_filtered_cloud->points = std::move(temp_points);
}
```
**예상 개선:** 30-40% 처리 시간 단축

### 2. 메모리 풀 사용
```cpp
class CloudMemoryPool {
private:
    std::queue<Cloud::Ptr> available_clouds_;
    std::mutex mutex_;
    
public:
    Cloud::Ptr acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_clouds_.empty()) {
            return std::make_shared<Cloud>();
        }
        auto cloud = available_clouds_.front();
        available_clouds_.pop();
        cloud->clear();
        return cloud;
    }
    
    void release(Cloud::Ptr cloud) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_clouds_.push(cloud);
    }
};
```

### 3. 적응형 Voxel 다운샘플링
```cpp
void OutlierFilter::adaptiveVoxelize(Cloud::Ptr &cloud_in, 
                                      Cloud::Ptr &cloud_out) {
    // 거리별로 다른 voxel 크기 적용
    std::vector<Cloud::Ptr> range_clouds(3);
    std::vector<float> voxel_sizes = {0.05, 0.10, 0.20};
    std::vector<float> range_limits = {10.0, 20.0, 30.0};
    
    // 거리별 분할
    for (const auto& point : cloud_in->points) {
        float dist = std::sqrt(point.x*point.x + 
                              point.y*point.y + 
                              point.z*point.z);
        
        if (dist < range_limits[0]) {
            range_clouds[0]->points.push_back(point);
        } else if (dist < range_limits[1]) {
            range_clouds[1]->points.push_back(point);
        } else {
            range_clouds[2]->points.push_back(point);
        }
    }
    
    // 각 범위별 다운샘플링
    cloud_out->clear();
    for (size_t i = 0; i < 3; ++i) {
        Cloud::Ptr downsampled(new Cloud);
        voxelizeCloud(range_clouds[i], downsampled, voxel_sizes[i]);
        *cloud_out += *downsampled;
    }
}
```
**예상 개선:** 포인트 수 30% 감소, 정확도 유지

## 장기 최적화 방안

### 1. GPU 가속 (PCL GPU)
```cpp
// PCL GPU 모듈 사용
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_memory.h>

class GPUAcceleratedFilter {
private:
    pcl::gpu::DeviceArray<pcl::PointXYZ> gpu_cloud_;
    
public:
    void processOnGPU(Cloud::Ptr &cloud) {
        // CPU -> GPU 전송
        gpu_cloud_.upload(cloud->points);
        
        // GPU에서 처리
        pcl::gpu::Octree gpu_octree;
        gpu_octree.setCloud(gpu_cloud_);
        gpu_octree.build();
        
        // 결과 다운로드
        std::vector<pcl::PointXYZ> result;
        gpu_cloud_.download(result);
    }
};
```

### 2. 파이프라인 최적화
```cpp
class PipelinedProcessor {
private:
    std::thread preprocessing_thread_;
    std::thread clustering_thread_;
    std::thread tracking_thread_;
    
    // 스레드 간 통신용 큐
    std::queue<Cloud::Ptr> preprocessing_queue_;
    std::queue<std::vector<ConeDescriptor>> clustering_queue_;
    
public:
    void startPipeline() {
        preprocessing_thread_ = std::thread([this]() {
            while (running_) {
                processPreprocessing();
            }
        });
        
        clustering_thread_ = std::thread([this]() {
            while (running_) {
                processClustering();
            }
        });
    }
};
```

### 3. SIMD 최적화
```cpp
#include <immintrin.h>  // AVX2

void optimizedDistanceCalculation(const float* points, 
                                   float* distances, 
                                   size_t count) {
    // AVX2를 사용한 거리 계산 (8개 float 동시 처리)
    for (size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_load_ps(&points[i * 3]);
        __m256 y = _mm256_load_ps(&points[i * 3 + 8]);
        __m256 z = _mm256_load_ps(&points[i * 3 + 16]);
        
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 z2 = _mm256_mul_ps(z, z);
        
        __m256 sum = _mm256_add_ps(_mm256_add_ps(x2, y2), z2);
        __m256 dist = _mm256_sqrt_ps(sum);
        
        _mm256_store_ps(&distances[i], dist);
    }
}
```

## 프로파일링 및 모니터링

### 1. 실시간 성능 모니터링
```cpp
class PerformanceMonitor {
private:
    struct TimingStats {
        double mean = 0.0;
        double max = 0.0;
        double min = std::numeric_limits<double>::max();
        size_t count = 0;
    };
    
    std::map<std::string, TimingStats> stats_;
    
public:
    class ScopedTimer {
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
        PerformanceMonitor* monitor_;
        
    public:
        ScopedTimer(const std::string& name, PerformanceMonitor* monitor)
            : name_(name), monitor_(monitor) {
            start_ = std::chrono::high_resolution_clock::now();
        }
        
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(
                end - start_).count();
            monitor_->recordTime(name_, ms);
        }
    };
    
    void recordTime(const std::string& name, double ms) {
        auto& stat = stats_[name];
        stat.count++;
        stat.mean = (stat.mean * (stat.count - 1) + ms) / stat.count;
        stat.max = std::max(stat.max, ms);
        stat.min = std::min(stat.min, ms);
    }
    
    void printStats() {
        for (const auto& [name, stat] : stats_) {
            RCLCPP_INFO(rclcpp::get_logger("performance"),
                "%s: mean=%.2fms, max=%.2fms, min=%.2fms",
                name.c_str(), stat.mean, stat.max, stat.min);
        }
    }
};

// 사용 예시
void OutlierFilter::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    PerformanceMonitor::ScopedTimer timer("total_callback", &perf_monitor_);
    
    {
        PerformanceMonitor::ScopedTimer timer("preprocessing", &perf_monitor_);
        // 전처리 코드
    }
    
    {
        PerformanceMonitor::ScopedTimer timer("clustering", &perf_monitor_);
        // 클러스터링 코드
    }
}
```

### 2. 메모리 사용량 추적
```cpp
class MemoryTracker {
public:
    static size_t getCurrentRSS() {
        std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
        std::string pid, comm, state, ppid, pgrp, session, tty_nr;
        std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
        std::string utime, stime, cutime, cstime, priority, nice;
        std::string O, itrealvalue, starttime;
        unsigned long vsize;
        long rss;
        
        stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session 
                    >> tty_nr >> tpgid >> flags >> minflt >> cminflt 
                    >> majflt >> cmajflt >> utime >> stime >> cutime 
                    >> cstime >> priority >> nice >> O >> itrealvalue 
                    >> starttime >> vsize >> rss;
        
        long page_size = sysconf(_SC_PAGE_SIZE);
        return rss * page_size;
    }
};
```

## 벤치마크 결과 예상

### 최적화 전
```
Total processing time: 50ms
- Preprocessing: 15ms
- Clustering: 20ms
- Validation: 10ms
- Publishing: 5ms
Memory usage: 100MB
```

### Quick Wins 적용 후
```
Total processing time: 35ms (-30%)
- Preprocessing: 10ms
- Clustering: 15ms
- Validation: 8ms
- Publishing: 2ms
Memory usage: 90MB (-10%)
```

### 전체 최적화 적용 후
```
Total processing time: 25ms (-50%)
- Preprocessing: 5ms (병렬화)
- Clustering: 10ms (DBSCAN + GPU)
- Validation: 7ms
- Publishing: 3ms
Memory usage: 80MB (-20%)
```

## 구현 우선순위

1. **즉시 (1주)**: Quick Wins 전체 적용
2. **단기 (2-3주)**: 병렬 처리, 메모리 풀
3. **중기 (1-2개월)**: GPU 가속, 파이프라인
4. **장기 (3개월+)**: 전체 아키텍처 개선

## 결론

성능 최적화는 단계적으로 진행하되, Quick Wins부터 시작하여 즉각적인 개선을 체감한 후 점진적으로 고급 최적화를 적용하는 것이 효과적입니다. 각 단계마다 프로파일링을 통해 실제 개선 효과를 측정하고 문서화해야 합니다.