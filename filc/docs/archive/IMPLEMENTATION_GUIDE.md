# 포인트클라우드 수직 보간 구현 가이드

## 1. 즉시 시작할 수 있는 최소 구현 (MVP)

### Step 1: 테스트용 간단한 노드 생성

```cpp
// test_interpolation_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class TestInterpolationNode : public rclcpp::Node {
public:
    TestInterpolationNode() : Node("test_interpolation_node") {
        // 구독자
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", 10,
            std::bind(&TestInterpolationNode::pointCloudCallback, this, std::placeholders::_1));
        
        // 발행자
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ouster/interpolated_points", 10);
        
        RCLCPP_INFO(this->get_logger(), "Test Interpolation Node Started");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 1. 입력 확인
        RCLCPP_INFO(this->get_logger(), 
            "Received cloud: %dx%d", msg->height, msg->width);
        
        if (msg->height != 32) {
            RCLCPP_WARN(this->get_logger(), 
                "Expected 32 channels, got %d", msg->height);
            return;
        }
        
        // 2. PCL 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // 3. 간단한 보간 테스트
        auto interpolated = simpleInterpolation(cloud);
        
        // 4. 발행
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*interpolated, output);
        output.header = msg->header;
        pub_->publish(output);
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr simpleInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        // 출력 클라우드 생성 (128x1024)
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
        output->width = 1024;
        output->height = 128;
        output->points.resize(128 * 1024);
        output->is_dense = false;
        
        // 원본 포인트 복사 및 보간
        for (int col = 0; col < 1024; ++col) {
            for (int row = 0; row < 32; ++row) {
                int input_idx = row * 1024 + col;
                int output_idx = (row * 4) * 1024 + col;
                
                // 원본 포인트 복사
                output->points[output_idx] = input->points[input_idx];
                
                // 다음 채널과의 보간 (마지막 채널 제외)
                if (row < 31) {
                    auto& p1 = input->points[input_idx];
                    auto& p2 = input->points[(row + 1) * 1024 + col];
                    
                    // 3개의 보간 포인트 생성
                    for (int i = 1; i <= 3; ++i) {
                        float t = i / 4.0f;
                        int interp_idx = (row * 4 + i) * 1024 + col;
                        
                        // 선형 보간 (일단 간단히)
                        output->points[interp_idx].x = p1.x * (1-t) + p2.x * t;
                        output->points[interp_idx].y = p1.y * (1-t) + p2.y * t;
                        output->points[interp_idx].z = p1.z * (1-t) + p2.z * t;
                        output->points[interp_idx].intensity = 
                            p1.intensity * (1-t) + p2.intensity * t;
                    }
                }
            }
        }
        
        return output;
    }
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestInterpolationNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Step 2: CMakeLists.txt에 추가

```cmake
# 테스트 노드 추가
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
install(TARGETS test_interpolation_node
  DESTINATION lib/${PROJECT_NAME}
)
```

### Step 3: 빌드 및 실행

```bash
# 빌드
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select filc

# 실행
source install/setup.bash
ros2 run filc test_interpolation_node

# 확인
ros2 topic echo /ouster/interpolated_points --no-arr
```

## 2. 개선된 구현 - 구면 좌표 기반

### Step 1: 구면 좌표 변환 추가

```cpp
// spherical_interpolation.hpp
#pragma once
#include <vector>
#include <cmath>

struct SphericalPoint {
    float range;
    float azimuth;
    float altitude;
    float intensity;
};

class SphericalInterpolator {
private:
    // OS1-32 빔 고도각 (도)
    const std::vector<float> beam_altitudes_ = {
        -16.611f, -16.084f, -15.557f, -15.029f, -14.502f, -13.975f,
        -13.447f, -12.920f, -12.393f, -11.865f, -11.338f, -10.811f,
        -10.283f, -9.756f, -9.229f, -8.701f, -8.174f, -7.646f,
        -7.119f, -6.592f, -6.064f, -5.537f, -5.010f, -4.482f,
        -3.955f, -3.428f, -2.900f, -2.373f, -1.846f, -1.318f,
        -0.791f, -0.264f
    };
    
    const float beam_origin_offset_ = 0.015806f; // 미터

public:
    // XYZ를 구면 좌표로 변환
    SphericalPoint cartesianToSpherical(float x, float y, float z, float intensity) {
        SphericalPoint sp;
        sp.range = std::sqrt(x*x + y*y + z*z);
        sp.azimuth = std::atan2(y, x);
        sp.altitude = std::asin(z / sp.range);
        sp.intensity = intensity;
        return sp;
    }
    
    // 구면 좌표를 XYZ로 변환 (Ouster 공식)
    void sphericalToCartesian(const SphericalPoint& sp, int col, 
                            float& x, float& y, float& z) {
        // 엔코더 각도
        float theta_encoder = 2.0f * M_PI * (1.0f - float(col) / 1024.0f);
        
        // 실제 방위각 = 엔코더 각도 (방위각 오프셋은 0으로 가정)
        float theta = theta_encoder;
        
        // 거리 보정
        float corrected_range = sp.range - beam_origin_offset_;
        
        // XYZ 계산
        float cos_phi = std::cos(sp.altitude);
        x = corrected_range * std::cos(theta) * cos_phi;
        y = corrected_range * std::sin(theta) * cos_phi;
        z = corrected_range * std::sin(sp.altitude);
    }
    
    // 고도각 보간 (큐빅)
    float interpolateAltitude(int ch_below, int ch_above, float t) {
        // 간단한 큐빅 보간
        int ch0 = std::max(0, ch_below - 1);
        int ch1 = ch_below;
        int ch2 = ch_above;
        int ch3 = std::min(31, ch_above + 1);
        
        float alt0 = beam_altitudes_[ch0] * M_PI / 180.0f;
        float alt1 = beam_altitudes_[ch1] * M_PI / 180.0f;
        float alt2 = beam_altitudes_[ch2] * M_PI / 180.0f;
        float alt3 = beam_altitudes_[ch3] * M_PI / 180.0f;
        
        // Catmull-Rom 스플라인
        float t2 = t * t;
        float t3 = t2 * t;
        
        return 0.5f * ((2 * alt1) +
                      (-alt0 + alt2) * t +
                      (2*alt0 - 5*alt1 + 4*alt2 - alt3) * t2 +
                      (-alt0 + 3*alt1 - 3*alt2 + alt3) * t3);
    }
    
    // Range 보간
    float interpolateRange(float r1, float r2, float t, float threshold = 0.5f) {
        if (std::abs(r2 - r1) > threshold) {
            // 불연속성: nearest neighbor
            return (t < 0.5f) ? r1 : r2;
        }
        // 연속적: 선형 보간
        return r1 * (1.0f - t) + r2 * t;
    }
};
```

### Step 2: 개선된 보간 노드

```cpp
// improved_interpolation_node.cpp
class ImprovedInterpolationNode : public rclcpp::Node {
private:
    SphericalInterpolator interpolator_;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr sphericalInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
        output->width = 1024;
        output->height = 128;
        output->points.resize(128 * 1024);
        output->is_dense = false;
        
        // 1단계: 구면 좌표로 변환
        std::vector<std::vector<SphericalPoint>> spherical_grid(32, 
            std::vector<SphericalPoint>(1024));
        
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 1024; ++col) {
                int idx = row * 1024 + col;
                auto& p = input->points[idx];
                spherical_grid[row][col] = 
                    interpolator_.cartesianToSpherical(p.x, p.y, p.z, p.intensity);
            }
        }
        
        // 2단계: 보간 수행
        for (int col = 0; col < 1024; ++col) {
            for (int row = 0; row < 32; ++row) {
                // 원본 포인트
                int out_row = row * 4;
                auto& sp = spherical_grid[row][col];
                interpolator_.sphericalToCartesian(sp, col,
                    output->points[out_row * 1024 + col].x,
                    output->points[out_row * 1024 + col].y,
                    output->points[out_row * 1024 + col].z);
                output->points[out_row * 1024 + col].intensity = sp.intensity;
                
                // 보간 포인트 (마지막 행 제외)
                if (row < 31) {
                    auto& sp1 = spherical_grid[row][col];
                    auto& sp2 = spherical_grid[row + 1][col];
                    
                    for (int i = 1; i <= 3; ++i) {
                        float t = i / 4.0f;
                        int interp_row = row * 4 + i;
                        
                        // 구면 좌표 보간
                        SphericalPoint interp_sp;
                        interp_sp.range = interpolator_.interpolateRange(
                            sp1.range, sp2.range, t);
                        interp_sp.altitude = interpolator_.interpolateAltitude(
                            row, row + 1, t);
                        interp_sp.azimuth = sp1.azimuth; // 같은 열이므로 동일
                        interp_sp.intensity = sp1.intensity * (1-t) + sp2.intensity * t;
                        
                        // XYZ 변환
                        interpolator_.sphericalToCartesian(interp_sp, col,
                            output->points[interp_row * 1024 + col].x,
                            output->points[interp_row * 1024 + col].y,
                            output->points[interp_row * 1024 + col].z);
                        output->points[interp_row * 1024 + col].intensity = 
                            interp_sp.intensity;
                    }
                }
            }
        }
        
        // 마지막 4개 행 처리 (채널 31의 보간)
        for (int col = 0; col < 1024; ++col) {
            for (int i = 125; i < 128; ++i) {
                output->points[i * 1024 + col] = output->points[124 * 1024 + col];
            }
        }
        
        return output;
    }
};
```

## 3. 검증 및 디버깅

### 디버깅용 시각화 노드

```python
#!/usr/bin/env python3
# visualize_interpolation.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np

class InterpolationVisualizer(Node):
    def __init__(self):
        super().__init__('interpolation_visualizer')
        
        self.original_sub = self.create_subscription(
            PointCloud2, '/ouster/points', self.original_callback, 10)
        self.interp_sub = self.create_subscription(
            PointCloud2, '/ouster/interpolated_points', self.interp_callback, 10)
        
        self.original_stats = None
        self.interp_stats = None
        
        self.timer = self.create_timer(1.0, self.print_stats)
    
    def calculate_stats(self, msg):
        # 포인트클라우드 통계 계산
        return {
            'height': msg.height,
            'width': msg.width,
            'total_points': msg.height * msg.width,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }
    
    def original_callback(self, msg):
        self.original_stats = self.calculate_stats(msg)
    
    def interp_callback(self, msg):
        self.interp_stats = self.calculate_stats(msg)
    
    def print_stats(self):
        self.get_logger().info('-' * 50)
        if self.original_stats:
            self.get_logger().info(f'Original: {self.original_stats["height"]}x{self.original_stats["width"]} = {self.original_stats["total_points"]} points')
        if self.interp_stats:
            self.get_logger().info(f'Interpolated: {self.interp_stats["height"]}x{self.interp_stats["width"]} = {self.interp_stats["total_points"]} points')
            if self.original_stats:
                ratio = self.interp_stats["total_points"] / self.original_stats["total_points"]
                self.get_logger().info(f'Interpolation ratio: {ratio:.2f}x')

def main():
    rclpy.init()
    node = InterpolationVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. 테스트 명령어

```bash
# 1. 빌드
cd /home/user1/ROS2_Workspace/ros2_ws
colcon build --packages-select filc

# 2. 첫 번째 터미널: 테스트 노드 실행
source install/setup.bash
ros2 run filc test_interpolation_node

# 3. 두 번째 터미널: 시각화
source install/setup.bash
python3 src/filc/scripts/visualize_interpolation.py

# 4. 세 번째 터미널: 토픽 확인
ros2 topic list
ros2 topic echo /ouster/interpolated_points --no-arr --once

# 5. RViz2로 시각화
rviz2
# PointCloud2 추가하고 topic을 /ouster/interpolated_points로 설정
```

## 5. 다음 단계

1. **성능 최적화**
   - OpenMP 병렬화 추가
   - SIMD 명령어 활용
   - 메모리 풀 사용

2. **정확도 개선**
   - 더 정교한 스플라인 보간
   - 적응적 보간 전략
   - 에지 보존 알고리즘

3. **통합**
   - 기존 main_fusion_node에 통합
   - 파라미터 서버 연동
   - 실시간 설정 변경

이 가이드를 따라 단계별로 구현하면 됩니다!