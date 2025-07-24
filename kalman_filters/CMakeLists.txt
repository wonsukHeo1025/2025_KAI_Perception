## kalman_filters library
set(PROJECT_NAME kalman_filters)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Enable -fPIC for all targets
project(${PROJECT_NAME})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Optional dlib support (no longer needed - we have our own Hungarian implementation)
# find_package(dlib QUIET)
# if(dlib_FOUND)
#     message(STATUS "dlib found - but not used, we have our own Hungarian implementation")
# endif()

# include_directories(include)

## Define library
add_library(${PROJECT_NAME}_lib SHARED
  src/ekf.cpp
  src/ukf.cpp
  src/system_model.cpp
  src/tracking/ukf_track.cpp
  src/tracking/multi_tracker.cpp
  src/tracking/hungarian_matcher.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

# No external dependencies needed for Hungarian algorithm


install(TARGETS ${PROJECT_NAME}_lib
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)


install(DIRECTORY include/kalman_filters/
    DESTINATION include/kalman_filters
    FILES_MATCHING PATTERN "*.hpp"
)

# Export library
install(EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Configure and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)
