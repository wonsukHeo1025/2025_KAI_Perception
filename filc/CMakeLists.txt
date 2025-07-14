cmake_minimum_required(VERSION 3.8)
project(filc)

# C++17 표준 사용
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 컴파일러 경고 추가
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 의존성 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# PCL
find_package(PCL REQUIRED)

# Eigen3
find_package(Eigen3 REQUIRED)

# OpenMP (선택적)
find_package(OpenMP)

# 포함 디렉토리
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Improved interpolation node (메인 노드)
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

# OpenMP 링크 (가능한 경우)
if(OpenMP_CXX_FOUND)
  target_link_libraries(improved_interpolation_node OpenMP::OpenMP_CXX)
endif()

# 설치
install(TARGETS
  improved_interpolation_node
  DESTINATION lib/${PROJECT_NAME}
)

# 헤더 파일 설치
install(DIRECTORY include/
  DESTINATION include
)

# 설정 파일 설치
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Launch 파일 설치
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Python 스크립트 설치
install(PROGRAMS
  scripts/visualize_interpolation.py
  scripts/benchmark_interpolation.py
  DESTINATION lib/${PROJECT_NAME}
)

# 테스트
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  OpenCV
  PCL
  Eigen3
)

ament_package()