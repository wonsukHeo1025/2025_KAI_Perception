cmake_minimum_required(VERSION 3.18)
project(cone_stellation)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(custom_interface REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM 4.2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem thread system)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Link directories
link_directories(${GTSAM_LIBRARY_DIRS})

# Core library (mostly header-only, but with some implementations)
add_library(${PROJECT_NAME}_core SHARED
  src/cone_stellation/preprocessing/cone_preprocessor.cpp
  src/cone_stellation/factors/inter_landmark_factors.cpp
  src/loop_closure_detector.cpp
)
target_link_libraries(${PROJECT_NAME}_core
  gtsam
  ${Boost_LIBRARIES}
)
ament_target_dependencies(${PROJECT_NAME}_core
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
)

# ROS2 node
add_executable(cone_slam_node src/cone_stellation/ros/cone_slam_node.cpp)
target_link_libraries(cone_slam_node 
  ${PROJECT_NAME}_core
  gtsam
  ${Boost_LIBRARIES}
  tbb
  pthread
)
ament_target_dependencies(cone_slam_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  custom_interface
)

# Install libraries and executables
install(TARGETS
  ${PROJECT_NAME}_core
  cone_slam_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install Python scripts
install(PROGRAMS
  scripts/dummy_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules (without ament_python_install_package)
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}/scripts
  FILES_MATCHING PATTERN "*.py"
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Unit tests for inter-landmark factors (commented out until test file is created)
  # ament_add_gtest(test_inter_landmark_factors
  #   test/test_inter_landmark_factors.cpp
  # )
  # target_link_libraries(test_inter_landmark_factors
  #   ${PROJECT_NAME}_core
  #   gtsam
  #   ${Boost_LIBRARIES}
  # )
  # ament_target_dependencies(test_inter_landmark_factors
  #   std_msgs
  #   geometry_msgs
  # )
endif()

ament_package()
