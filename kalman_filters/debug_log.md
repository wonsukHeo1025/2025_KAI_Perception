# Kalman Filters Library Debug Log

## 2025-07-23 - Library Refactoring Started

### Issue: Naming Inconsistency
- **Problem**: Library uses both full names (UnscentedKF) and abbreviations (UKFTrack) inconsistently
- **Solution**: Standardizing to use abbreviations (UKF, EKF) throughout the library for consistency
- **Status**: In progress

### Issue: Hungarian Matching Implementation
- **Problem**: Current implementation relies on dlib when available, falls back to greedy matching
- **Solution**: Implementing a standalone Hungarian algorithm to remove dlib dependency
- **Status**: Pending

### Issue: Numerical Stability
- **Problem**: Cholesky decomposition can fail without fallback, causing crashes
- **Solution**: Adding SVD fallback and regularization for numerical stability
- **Status**: Pending

## 2025-07-23 - Library Refactoring Completed

### Issue: Naming Inconsistency
- **Problem**: Inconsistent naming between UnscentedKF/ExtendedKF and UKFTrack
- **Solution**: Renamed UnscentedKF to UKF and ExtendedKF to EKF for consistency
- **Result**: Resolved - All classes now use consistent abbreviations

### Issue: Hungarian Algorithm Implementation
- **Problem**: Dependency on dlib library for Hungarian matching
- **Solution**: Implemented a complete Hungarian algorithm from scratch using augmenting path method
- **Result**: Resolved - dlib dependency removed, proper O(n³) Hungarian algorithm implemented

### Issue: Numerical Stability in Matrix Operations
- **Problem**: Cholesky decomposition failures causing crashes
- **Solution**: Added SVD fallback with regularization when Cholesky fails
- **Result**: Resolved - Both UKF and UKFTrack now handle ill-conditioned matrices gracefully

### Issue: Error Handling in Matrix Inversions
- **Problem**: Direct matrix inversions without condition checking
- **Solution**: Added condition number checking and pseudo-inverse fallback for nearly singular matrices
- **Result**: Resolved - Both UKF and EKF now use SVD-based pseudo-inverse when needed

### Issue: Const Correctness
- **Problem**: Getter methods returning by value instead of const reference
- **Solution**: Changed get_state() and get_cov() to return const references
- **Result**: Resolved - Improved performance and API design

### Issue: Memory Allocations
- **Problem**: Frequent vector reallocations in predict/update cycles
- **Solution**: Added thread_local storage and reserve() calls for frequently used vectors
- **Result**: Resolved - Reduced memory allocations in hot paths

## 2025-07-23 - Build Configuration Fixed

### Issue: Static Library Linking Error
- **Problem**: kalman_filters built as static library without -fPIC, causing link errors in shared libraries
- **Solution**: Changed from STATIC to SHARED library and added CMAKE_POSITION_INDEPENDENT_CODE
- **Result**: Resolved - Library can now be linked into shared libraries

### Issue: CMake Finding Wrong Library Type
- **Problem**: CMake was still looking for .a file instead of .so file
- **Solution**: Updated find_library in both packages to explicitly search for the library
- **Result**: Resolved - Both cone_detection and calico now build successfully with shared library