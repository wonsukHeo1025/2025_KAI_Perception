# Calico Debug Log

## 2025-07-23 - Removed dlib Dependency

### Issue: dlib Dependency in CMakeLists.txt
- **Problem**: Calico was depending on dlib for Hungarian algorithm
- **Solution**: Replaced with kalman_filters library's own Hungarian implementation
- **Result**: Resolved - dlib dependency removed from CMakeLists.txt

### Issue: Hungarian Matcher Implementation
- **Problem**: Hungarian matcher was using dlib directly
- **Solution**: Created adapter that delegates to kalman_filters::tracking::HungarianMatcher
- **Result**: Resolved - Now uses kalman_filters implementation without dlib

### Issue: Build Success
- **Problem**: Initial build failed due to static library linking issues
- **Solution**: Fixed kalman_filters to build as shared library and updated CMake configuration
- **Result**: Resolved - calico now builds successfully without dlib dependency