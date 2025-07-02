cmake_minimum_required(VERSION 3.18)
project(calico)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(kalman_filters REQUIRED)
# OR-Tools is used directly without find_package due to protobuf issues
# dlib for Hungarian algorithm fallback
find_package(dlib REQUIRED)

# Custom messages
find_package(custom_interface REQUIRED)
find_package(yolo_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${kalman_filters_INCLUDE_DIRS}
  /usr/local/include
)

# Link directories
link_directories(${EIGEN3_LIBRARY_DIRS})

# Define dependencies list
set(dependencies
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  custom_interface
  yolo_msgs
)

# Add library for utils
add_library(calico_utils SHARED
  src/utils/config_loader.cpp
  src/utils/message_converter.cpp
  src/utils/projection_utils.cpp
)
target_link_libraries(calico_utils
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
)
ament_target_dependencies(calico_utils ${dependencies})

# Add library for fusion
add_library(calico_fusion SHARED
  src/fusion/hungarian_matcher.cpp
  src/fusion/multi_camera_fusion.cpp
)
target_link_libraries(calico_fusion
  calico_utils
  ${OpenCV_LIBS}
  dlib::dlib
)
ament_target_dependencies(calico_fusion ${dependencies})

# Add library for tracking
add_library(calico_tracking SHARED
  src/tracking/track.cpp
  src/tracking/ukf_tracker.cpp
  src/tracking/imu_compensator.cpp
)
target_link_libraries(calico_tracking
  calico_utils
  calico_fusion
  ${kalman_filters_LIBRARIES}
)
ament_target_dependencies(calico_tracking ${dependencies})

# Add library for visualization
add_library(calico_visualization SHARED
  src/visualization/rviz_marker_publisher.cpp
)
target_link_libraries(calico_visualization
  calico_utils
)
ament_target_dependencies(calico_visualization ${dependencies})

# Add multi-camera fusion node
add_executable(multi_camera_fusion_node
  src/nodes/multi_camera_fusion_node.cpp
)
target_link_libraries(multi_camera_fusion_node
  calico_utils
  calico_fusion
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
)
ament_target_dependencies(multi_camera_fusion_node ${dependencies})

# Add UKF tracking node
add_executable(ukf_tracking_node
  src/nodes/ukf_tracking_node.cpp
)
target_link_libraries(ukf_tracking_node
  calico_utils
  calico_tracking
)
ament_target_dependencies(ukf_tracking_node ${dependencies})

# Add visualization node
add_executable(visualization_node
  src/nodes/visualization_node.cpp
)
target_link_libraries(visualization_node
  calico_utils
  calico_visualization
)
ament_target_dependencies(visualization_node ${dependencies})

# Add projection debug node
add_executable(projection_debug_node
  src/nodes/projection_debug_node.cpp
)
target_link_libraries(projection_debug_node
  calico_utils
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
)
ament_target_dependencies(projection_debug_node ${dependencies})

# Install targets
install(TARGETS
  calico_utils
  calico_fusion
  calico_tracking
  calico_visualization
  multi_camera_fusion_node
  ukf_tracking_node
  visualization_node
  projection_debug_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files from hungarian_association package
# Users can symlink or copy their config files
install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}/config
)

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(calico_utils calico_fusion calico_tracking calico_visualization)
ament_export_dependencies(${dependencies})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
