# CLAUDE.md - Kalman Filters Library

## Overview
The kalman_filters library is a header-only C++ library providing Kalman filter implementations for state estimation and object tracking in robotics applications. It's designed to be ROS2-independent for maximum reusability.

## Current Status (2025-07-23)
- ✅ **Refactoring Complete**: Major improvements to code quality, naming consistency, and robustness
- ✅ **dlib Dependency Removed**: Implemented standalone Hungarian algorithm
- ✅ **Numerical Stability Enhanced**: Added SVD fallbacks for matrix operations
- ✅ **Error Handling Improved**: Proper handling of ill-conditioned matrices
- ✅ **Performance Optimized**: Reduced memory allocations with thread_local storage

## Key Components

### Core Filters
- **UKF** (Unscented Kalman Filter): Renamed from UnscentedKF for consistency
- **EKF** (Extended Kalman Filter): Renamed from ExtendedKF for consistency
- **SystemModel**: Abstract base class for defining system dynamics and observations

### Tracking System
- **UKFTrack**: Single object tracker using UKF with constant velocity model
- **MultiTracker**: Multi-object tracking with data association
- **HungarianMatcher**: Optimal assignment algorithm (now dlib-free)

## Recent Changes (2025-07-23)

### 1. Naming Consistency
- `UnscentedKF` → `UKF`
- `ExtendedKF` → `EKF`
- All classes now use consistent abbreviations

### 2. Hungarian Algorithm
- Removed dlib dependency completely
- Implemented proper O(n³) Hungarian algorithm using augmenting paths
- Works with non-square cost matrices
- Includes distance threshold filtering

### 3. Numerical Robustness
- Added SVD fallback when Cholesky decomposition fails
- Regularization for nearly singular matrices
- Condition number checking for matrix inversions
- Pseudo-inverse for ill-conditioned matrices

### 4. API Improvements
- Const correctness: getters return const references
- Thread-safe memory optimizations
- Better error handling throughout

## Building the Library

```bash
cd /home/user1/ROS2_Workspace/ros2_ws/src/kalman_filters
mkdir build && cd build
cmake ..
make
sudo make install
```

## Usage in Other Projects

### CMakeLists.txt
```cmake
find_library(KALMAN_FILTERS_LIB kalman_filters_lib REQUIRED)
target_link_libraries(your_target ${KALMAN_FILTERS_LIB})
```

### Include Headers
```cpp
#include <kalman_filters/ukf.hpp>
#include <kalman_filters/tracking/multi_tracker.hpp>
```

## Integration with calico

The calico package uses this library for UKF tracking with IMU compensation. The integration is seamless:
- calico's `ukf_tracking_node.cpp` uses `MultiTracker` 
- Hungarian matching available for both tracking and sensor fusion
- No code changes needed in calico after the refactoring

## Integration with cone_detection

The cone_detection package uses basic tracking functionality:
- Simple `MultiTracker` integration for LiDAR cone tracking
- Works with the refactored library without modifications

## Testing Recommendations

1. **Unit Tests**: Test matrix operations with ill-conditioned matrices
2. **Integration Tests**: Verify tracking performance with noisy data
3. **Performance Tests**: Measure improvement from memory optimizations

## Future Enhancements

1. **Additional Filters**: Particle Filter, IMM (Interacting Multiple Model)
2. **GPU Acceleration**: CUDA support for large-scale tracking
3. **Python Bindings**: pybind11 wrapper for Python users
4. **Documentation**: Doxygen documentation and usage examples