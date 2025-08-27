#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>

class TestInterpolationNode : public rclcpp::Node {
public:
    TestInterpolationNode() : Node("test_interpolation_node") {
        // 파라미터 선언
        this->declare_parameter("input_topic", "/ouster/points");
        this->declare_parameter("output_topic", "/ouster/interpolated_points");
        this->declare_parameter("scale_factor", 4.0);
        
        // 파라미터 가져오기
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        
        // 구독자
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, rclcpp::QoS(10).best_effort(),
            std::bind(&TestInterpolationNode::pointCloudCallback, this, std::placeholders::_1));
        
        // 발행자
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic, rclcpp::QoS(10).reliable());
        
        // 통계 타이머
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&TestInterpolationNode::printStats, this));
        
        RCLCPP_INFO(this->get_logger(), 
            "Test Interpolation Node Started\n"
            "  Input: %s\n"
            "  Output: %s\n"
            "  Scale Factor: %.1f",
            input_topic.c_str(), output_topic.c_str(), scale_factor_);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 1. 입력 확인
        if (msg->height != 32 || msg->width != 1024) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Expected 32x1024 organized cloud, got %dx%d", msg->height, msg->width);
            return;
        }
        
        // 2. PCL 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // 3. 간단한 보간
        auto interpolated = simpleInterpolation(cloud);
        
        // 4. 발행
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*interpolated, output);
        output.header = msg->header;
        pub_->publish(output);
        
        // 5. 통계 업데이트
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        frame_count_++;
        total_processing_time_ += duration.count();
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Processed frame in %ld ms", duration.count());
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr simpleInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        // 출력 클라우드 생성
        int output_height = static_cast<int>(input->height * scale_factor_);
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
        output->width = input->width;
        output->height = output_height;
        output->points.resize(output_height * input->width);
        output->is_dense = false;
        
        // 원본 포인트 복사 및 보간
        for (size_t col = 0; col < input->width; ++col) {
            for (size_t row = 0; row < input->height; ++row) {
                size_t input_idx = row * input->width + col;
                size_t output_idx = static_cast<size_t>(row * scale_factor_) * input->width + col;
                
                // 원본 포인트 복사
                output->points[output_idx] = input->points[input_idx];
                
                // 다음 채널과의 보간 (마지막 채널 제외)
                if (row < input->height - 1) {
                    const auto& p1 = input->points[input_idx];
                    const auto& p2 = input->points[(row + 1) * input->width + col];
                    
                    // 보간 포인트 생성
                    for (int i = 1; i < static_cast<int>(scale_factor_); ++i) {
                        float t = static_cast<float>(i) / scale_factor_;
                        size_t interp_idx = (static_cast<size_t>(row * scale_factor_) + i) * input->width + col;
                        
                        // 유효한 포인트인지 확인
                        if (!std::isfinite(p1.x) || !std::isfinite(p2.x)) {
                            output->points[interp_idx].x = std::numeric_limits<float>::quiet_NaN();
                            output->points[interp_idx].y = std::numeric_limits<float>::quiet_NaN();
                            output->points[interp_idx].z = std::numeric_limits<float>::quiet_NaN();
                            output->points[interp_idx].intensity = 0;
                            continue;
                        }
                        
                        // 선형 보간
                        output->points[interp_idx].x = p1.x * (1-t) + p2.x * t;
                        output->points[interp_idx].y = p1.y * (1-t) + p2.y * t;
                        output->points[interp_idx].z = p1.z * (1-t) + p2.z * t;
                        output->points[interp_idx].intensity = p1.intensity * (1-t) + p2.intensity * t;
                    }
                }
            }
        }
        
        // 마지막 몇 개 행 처리 (마지막 원본 행 이후)
        size_t last_original_row = input->height - 1;
        size_t last_output_row = static_cast<size_t>(last_original_row * scale_factor_);
        for (size_t row = last_output_row + 1; row < output->height; ++row) {
            for (size_t col = 0; col < output->width; ++col) {
                size_t idx = row * output->width + col;
                size_t last_idx = last_output_row * output->width + col;
                output->points[idx] = output->points[last_idx];
            }
        }
        
        return output;
    }
    
    void printStats() {
        if (frame_count_ > 0) {
            double avg_time = static_cast<double>(total_processing_time_) / frame_count_;
            double fps = 1000.0 / avg_time;
            
            RCLCPP_INFO(this->get_logger(), 
                "=== Interpolation Statistics ===\n"
                "  Frames processed: %ld\n"
                "  Average time: %.2f ms\n"
                "  Average FPS: %.1f\n"
                "  Scale factor: %.1fx (32->%d channels)",
                frame_count_, avg_time, fps, scale_factor_, 
                static_cast<int>(32 * scale_factor_));
        }
    }
    
    // ROS2 인터페이스
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // 파라미터
    double scale_factor_;
    
    // 통계
    size_t frame_count_ = 0;
    long total_processing_time_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestInterpolationNode>());
    rclcpp::shutdown();
    return 0;
}