cmake_minimum_required(VERSION 3.8)
project(cone_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(custom_interface REQUIRED)

# Find kalman_filters library (installed separately)
find_library(KALMAN_FILTERS_LIB 
  NAMES kalman_filters_lib
  PATHS /usr/local/lib
  PATH_SUFFIXES lib
  REQUIRED
  NO_DEFAULT_PATH
)

# dlib no longer needed - kalman_filters has its own Hungarian implementation
message(STATUS "Using kalman_filters Hungarian matching implementation")

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  /usr/local/include  # for kalman_filters
  /usr/include/pcl-1.12
)

link_directories(
${PCL_LIBRARY_DIRS}
)

add_definitions(
${PCL_DEFINITIONS}
)

# 소스 파일 목록 설정
set(SOURCES
  src/cone_detection_node.cpp
)

set(VIZ_SOURCES
  src/cone_detection_visualization_node.cpp
)

# 실행 파일 생성
add_executable(cone_detection_node ${SOURCES})
add_executable(cone_detection_visualization_node ${VIZ_SOURCES})

# Link libraries (custom_interface 제거, ament_target_dependencies로 처리)
target_link_libraries(cone_detection_node 
  ${PCL_LIBRARIES} 
  Eigen3::Eigen
  ${KALMAN_FILTERS_LIB}
)

target_link_libraries(cone_detection_visualization_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

# dlib no longer needed

# ROS2 의존성 연결
ament_target_dependencies(cone_detection_node
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  custom_interface
)

ament_target_dependencies(cone_detection_visualization_node
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  custom_interface
  geometry_msgs
)

# 설치 규칙
install(TARGETS
  cone_detection_node
  cone_detection_visualization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  include
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
