# GLIM Features Integration Roadmap for ConeSTELLATION

## Overview
This document provides a detailed integration plan for advanced SLAM features from GLIM into ConeSTELLATION, with specific implementation strategies and priorities.

## Current Status
- ✅ Basic factor graph SLAM
- ✅ Inter-landmark factors
- ✅ Tentative landmark with basic color voting
- ⚠️ Missing: Loop closure, advanced optimization
- ⚠️ Missing: Multi-threading, memory management, robust kernels
- 📝 Note: Fixed-lag smoother POSTPONED (using external odometry)

## Priority 1: Core Performance Features (Must Have)

### 1. ~~Fixed-Lag Smoother Implementation~~ (POSTPONED)
**Status**: POSTPONED based on GLIM architecture analysis (2025-07-21)

**Reasoning**:
- GLIM uses fixed-lag smoother for IMU odometry estimation, NOT landmark SLAM
- Since we have external IMU+GPS odometry at 100Hz, this is not needed
- Landmark SLAM should remain unbounded for maximum accuracy
- May revisit if memory becomes issue in very long operations

**Alternative Approach**:
- Use external odometry for real-time pose (100Hz)
- SLAM focuses on mapping and drift correction only
- Consider periodic landmark pruning if memory becomes an issue
  void marginalize_old_states();
  // Implementation details removed - feature postponed
};
```

**Original Benefits** (for future reference):
- Would provide bounded memory usage O(k) instead of O(n)
- Would ensure consistent computation time
- Would preserve loop closure information through marginalization

### 2. Multi-threaded Architecture
**GLIM Reference**: `glim/src/glim/odometry/async_odometry_estimation.cpp`

**Implementation Plan**:
```cpp
// Async wrapper pattern from GLIM
template<typename T>
class AsyncProcessor {
  void start() {
    thread_ = std::thread([this] {
      while (!kill_switch_) {
        auto input = input_queue_.pop();
        if (input) {
          auto output = processor_->process(*input);
          output_queue_.push(output);
        }
      }
    });
  }
  
private:
  std::thread thread_;
  ConcurrentQueue<Input> input_queue_;
  ConcurrentQueue<Output> output_queue_;
  std::unique_ptr<T> processor_;
};

// Apply to cone processing
using AsyncConePreprocessor = AsyncProcessor<ConePreprocessor>;
using AsyncConeMapper = AsyncProcessor<ConeMapping>;
```

**Thread Architecture**:
```
Main Thread
├── ROS2 callbacks (cone detection, odometry)
├── Keyframe selection
└── Visualization

Preprocessing Thread
├── Cone filtering
├── Pattern detection
└── Color validation

Mapping Thread
├── Data association
├── Factor graph update
└── Optimization

Loop Closure Thread
├── Candidate detection
├── Validation
└── Constraint addition
```

### 3. Robust Optimization
**GLIM Reference**: `glim/include/glim/util/robust_kernel.hpp`

**Implementation Plan**:
```cpp
// Robust kernel factory
class RobustKernelFactory {
  static gtsam::noiseModel::Robust::shared_ptr create(
      const std::string& type,  // "huber", "tukey", "dcs"
      double parameter,
      gtsam::SharedNoiseModel base_model) {
    
    if (type == "huber") {
      return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(parameter),
        base_model);
    }
    // ... other kernels
  }
};

// Apply to cone factors
auto cone_noise = gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));
auto robust_cone_noise = RobustKernelFactory::create("huber", 1.0, cone_noise);
graph.add(ConeObservationFactor(x1, l1, measurement, robust_cone_noise));
```

## Priority 2: Advanced Features (Should Have)

### 4. Loop Closure with Cone Constellations
**GLIM Reference**: `glim/src/glim/mapping/global_mapping_pose_graph.cpp`

**Implementation Plan**:
```cpp
class ConeLoopDetector {
  struct ConstellationDescriptor {
    std::vector<RelativeConePosition> relative_positions;
    std::vector<ConeColor> colors;
    Eigen::Matrix3d covariance;
    
    double distance_to(const ConstellationDescriptor& other) const;
  };
  
  // Build descriptor from local cone map
  ConstellationDescriptor build_descriptor(
      const std::vector<ConeLandmark>& local_cones,
      const gtsam::Pose2& reference_pose);
  
  // Find loop candidates
  std::vector<LoopCandidate> find_candidates(
      const ConstellationDescriptor& query,
      double min_travel_distance = 10.0) {
    // Parallel evaluation using TBB
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, descriptors_.size()),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          if (travel_distance(current_id, i) > min_travel_distance) {
            double score = query.distance_to(descriptors_[i]);
            if (score < threshold) {
              candidates.push_back({i, score});
            }
          }
        }
      });
  }
};
```

### 5. Memory Management Strategy
**GLIM Reference**: `glim/include/glim/util/data_store.hpp`

**Implementation Plan**:
```cpp
class ConeMapDataStore {
  struct Policy {
    size_t max_cones = 1000;
    size_t max_observations_per_cone = 50;
    size_t lru_cache_size = 100;
    bool enable_compression = true;
  };
  
  // LRU cache for cone descriptors
  template<typename Key, typename Value>
  class LRUCache {
    void put(const Key& key, const Value& value) {
      cache_[key] = value;
      access_order_.push_front(key);
      
      if (cache_.size() > max_size_) {
        auto oldest = access_order_.back();
        cache_.erase(oldest);
        access_order_.pop_back();
      }
    }
  };
  
  // Compressed cone storage
  struct CompressedCone {
    uint16_t x, y;  // Fixed-point position
    uint8_t color;
    uint8_t confidence;
  };
};
```

## Priority 3: Production Features (Nice to Have)

### 8. Enhanced Color Voting System (Low Priority)
**Current Implementation**: Basic voting in TentativeLandmark

**Enhancement Plan**:
```cpp
class AdvancedColorVoting {
  struct ColorHypothesis {
    ConeColor color;
    double probability;
    double confidence;
    std::vector<double> observation_weights;
  };
  
  // Bayesian color classification
  ColorHypothesis update_color_belief(
      const ColorHypothesis& prior,
      const ConeObservation& observation) {
    // Distance-based weight
    double distance_weight = exp(-observation.distance / 10.0);
    
    // Temporal consistency weight
    double time_weight = exp(-time_since_last / 1.0);
    
    // Viewing angle weight
    double angle_weight = cos(observation.viewing_angle);
    
    // Combined weight
    double weight = distance_weight * time_weight * angle_weight;
    
    // Bayesian update
    return bayesian_update(prior, observation.color, weight);
  }
  
  // Multi-hypothesis tracking
  std::vector<ColorHypothesis> maintain_hypotheses(
      const std::vector<ConeObservation>& observations) {
    // Keep top-k color hypotheses
    // Prune low-probability hypotheses
    // Handle ambiguous cases
  }
};
```

### 9. Configuration Management System
**GLIM Reference**: `glim/src/glim/util/config.cpp`

```yaml
# config/cone_slam_profile.yaml
profiles:
  fast:
    optimization:
      skip_frames: 5
      max_iterations: 10
      use_robust_kernels: false
    loop_closure:
      enabled: false
      
  accurate:
    optimization:
      skip_frames: 1
      max_iterations: 50
      use_robust_kernels: true
    loop_closure:
      enabled: true
      min_score: 0.7
      
  race:
    optimization:
      # fixed_lag_window: 10.0  # POSTPONED - not needed with external odometry
      marginalize_old: true
    multi_threading:
      preprocessing_threads: 2
      mapping_threads: 1
```

### 10. Serialization and Map Sharing
**GLIM Reference**: `glim/src/glim/io/`

```cpp
class ConeMapSerializer {
  // Save cone map for track learning
  void save_map(const std::string& filename) {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);
    
    packer.pack_map(3);
    packer.pack("version"); packer.pack(1);
    packer.pack("landmarks"); pack_landmarks(packer);
    packer.pack("metadata"); pack_metadata(packer);
    
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(buffer.data(), buffer.size());
  }
  
  // Share between vehicles
  TrackedConeArray create_sharing_message() {
    // Create compressed representation
    // Include only high-confidence cones
    // Add track-specific metadata
  }
};
```

## Implementation Timeline

### Sprint 1 (2 weeks): Core Performance
- [ ] ~~Fixed-lag smoother integration~~ POSTPONED
- [ ] Basic multi-threading (2 threads)
- [ ] Huber robust kernels

### Sprint 2 (2 weeks): Advanced Features  
- [ ] Loop closure detection
- [ ] Memory management
- [ ] Production configurations

### Sprint 3 (1 week): Testing & Optimization
- [ ] Performance profiling
- [ ] Parameter tuning
- [ ] Race simulation testing

### Sprint 4 (1 week): Production Features
- [ ] Configuration profiles
- [ ] Map serialization
- [ ] Documentation

## Success Metrics

1. **Performance**:
   - Optimization time < 50ms per keyframe
   - Memory usage < 1GB for 1000 landmarks
   - CPU usage < 80% on target hardware

2. **Accuracy**:
   - Loop closure success rate > 90%
   - Color classification accuracy > 95%
   - Trajectory error < 0.5m RMSE

3. **Robustness**:
   - Handle 30% cone occlusion
   - Recover from 5-second detection blackout
   - Operate in rain/dust conditions

## Testing Strategy

1. **Unit Tests**:
   - ~~Fixed-lag smoother convergence~~ N/A
   - Thread safety verification
   - Memory leak detection

2. **Integration Tests**:
   - Multi-threaded pipeline
   - Loop closure scenarios
   - Long-duration stability

3. **Performance Tests**:
   - Benchmark against requirements
   - Profile bottlenecks
   - Optimize critical paths

## Risk Mitigation

1. **Numerical Stability**:
   - Use GLIM's fallback mechanisms
   - Monitor condition numbers
   - Implement recovery strategies

2. **Real-time Constraints**:
   - Adaptive quality settings
   - Computation budgets
   - Graceful degradation

3. **Integration Complexity**:
   - Incremental integration
   - Feature flags
   - Rollback capability