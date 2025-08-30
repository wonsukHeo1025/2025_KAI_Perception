cmake_minimum_required(VERSION 3.8)
project(filc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 강화된 성능 최적화 플래그 (무조건 적용하면 좋은 최적화들)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -funroll-loops -ffast-math -flto")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native -funroll-loops")

# 추가 성능 최적화 플래그
if(CMAKE_COMPILER_IS_GNUCXX)
    # GCC 전용 최적화
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-functions -fprefetch-loop-arrays")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing")
    
    # 브랜치 예측 최적화
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use")
    
    # 벡터화 최적화
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fvect-cost-model=unlimited")
endif()

# 링크 타임 최적화 (LTO)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find PCL
find_package(PCL REQUIRED)

# Find yaml-cpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# Find OpenMP for multithreading optimization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling multithreading optimization.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DWITH_OPENMP)
else()
    message(WARNING "OpenMP not found. Multithreading optimization disabled.")
endif()

# Check for SIMD support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)

if(COMPILER_SUPPORTS_AVX2)
    add_definitions(-DWITH_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    message(STATUS "AVX2 support enabled.")
elseif(COMPILER_SUPPORTS_SSE41)
    add_definitions(-DWITH_SSE41)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
    message(STATUS "SSE4.1 support enabled.")
endif()

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Create library for utilities
add_library(${PROJECT_NAME}_utils
  src/config_loader.cpp
  src/projection_utils.cpp
  src/interpolation_utils.cpp
)

ament_target_dependencies(${PROJECT_NAME}_utils
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
)

target_link_libraries(${PROJECT_NAME}_utils
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_utils OpenMP::OpenMP_CXX)
endif()

target_include_directories(${PROJECT_NAME}_utils PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Main fusion node
add_executable(main_fusion_node src/main_fusion_node.cpp)
ament_target_dependencies(main_fusion_node
  rclcpp
  ament_index_cpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

target_link_libraries(main_fusion_node
  ${PROJECT_NAME}_utils
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(main_fusion_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(main_fusion_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Range Image Interpolation node (Camera 없이 보간만)
add_executable(filc_interpolation_node src/main_fusion_node.cpp)
ament_target_dependencies(filc_interpolation_node
  rclcpp
  ament_index_cpp
  sensor_msgs
  std_msgs
  cv_bridge
)

target_link_libraries(filc_interpolation_node
  ${PROJECT_NAME}_utils
  ${OpenCV_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(filc_interpolation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(filc_interpolation_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Test interpolation node
add_executable(test_interpolation_node src/test_interpolation_node.cpp)
ament_target_dependencies(test_interpolation_node
  rclcpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

target_link_libraries(test_interpolation_node
  ${PCL_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_interpolation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(test_interpolation_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

# Image interpolation node
add_executable(image_interpolation_node src/image_interpolation_node.cpp)
ament_target_dependencies(image_interpolation_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
)

target_link_libraries(image_interpolation_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(image_interpolation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(image_interpolation_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Spherical interpolation node
add_executable(spherical_interpolation_node src/spherical_interpolation_node.cpp)
ament_target_dependencies(spherical_interpolation_node
  rclcpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

target_link_libraries(spherical_interpolation_node
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(spherical_interpolation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(spherical_interpolation_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Improved interpolation node
add_executable(improved_interpolation_node src/improved_interpolation_node.cpp)
ament_target_dependencies(improved_interpolation_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
)

target_link_libraries(improved_interpolation_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(improved_interpolation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(improved_interpolation_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Visualization node
add_executable(visualization_node src/visualization_node.cpp)
ament_target_dependencies(visualization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  visualization_msgs
)

target_link_libraries(visualization_node
  ${PROJECT_NAME}_utils
  ${OpenCV_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(visualization_node OpenMP::OpenMP_CXX)
endif()

# Install executables
install(TARGETS
  main_fusion_node
  filc_interpolation_node
  test_interpolation_node
  image_interpolation_node
  spherical_interpolation_node
  improved_interpolation_node
  visualization_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install library
install(TARGETS ${PROJECT_NAME}_utils
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install Python scripts
install(PROGRAMS
  scripts/visualize_interpolation.py
  scripts/test_functionality.py
  scripts/benchmark_interpolation.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_utils)
ament_export_dependencies(
  rclcpp
  ament_index_cpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  OpenCV
  PCL
  Eigen3
)

ament_package()
