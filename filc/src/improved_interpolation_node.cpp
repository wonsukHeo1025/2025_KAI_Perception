#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>

class ImprovedInterpolationNode : public rclcpp::Node {
public:
    ImprovedInterpolationNode() : Node("improved_interpolation_node") {
        // 파라미터 선언
        this->declare_parameter("scale_factor", 3.0);
        this->declare_parameter("use_image_features", true);
        this->declare_parameter("interpolation_method", "cubic");
        
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        use_image_features_ = this->get_parameter("use_image_features").as_bool();
        interpolation_method_ = this->get_parameter("interpolation_method").as_string();
        
        // OS1-32 고도각 초기화
        initializeBeamAltitudes();
        
        // 구독자
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", rclcpp::QoS(10).best_effort(),
            std::bind(&ImprovedInterpolationNode::pointCloudCallback, this, std::placeholders::_1));
        
        if (use_image_features_) {
            range_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/ouster/range_image", rclcpp::QoS(10).best_effort(),
                std::bind(&ImprovedInterpolationNode::rangeImageCallback, this, std::placeholders::_1));
            
            signal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/ouster/signal_image", rclcpp::QoS(10).best_effort(),
                std::bind(&ImprovedInterpolationNode::signalImageCallback, this, std::placeholders::_1));
        }
        
        // 발행자
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ouster/improved_interpolated_points", rclcpp::QoS(10).reliable());
        
        // 통계 타이머
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&ImprovedInterpolationNode::printStats, this));
        
        RCLCPP_INFO(this->get_logger(), 
            "Improved Interpolation Node Started\n"
            "  Scale Factor: %.1fx\n"
            "  Image Features: %s\n"
            "  Method: %s",
            scale_factor_, 
            use_image_features_ ? "Enabled" : "Disabled",
            interpolation_method_.c_str());
    }

private:
    void initializeBeamAltitudes() {
        // OS1-32 고도각 (도)
        std::vector<float> angles_deg = {
            -16.611f, -16.084f, -15.557f, -15.029f, -14.502f, -13.975f,
            -13.447f, -12.920f, -12.393f, -11.865f, -11.338f, -10.811f,
            -10.283f, -9.756f, -9.229f, -8.701f, -8.174f, -7.646f,
            -7.119f, -6.592f, -6.064f, -5.537f, -5.010f, -4.482f,
            -3.955f, -3.428f, -2.900f, -2.373f, -1.846f, -1.318f,
            -0.791f, -0.264f
        };
        
        // 라디안으로 변환
        beam_altitudes_rad_.resize(angles_deg.size());
        for (size_t i = 0; i < angles_deg.size(); ++i) {
            beam_altitudes_rad_[i] = angles_deg[i] * M_PI / 180.0f;
        }
        
        // 보간된 고도각 미리 계산
        int target_channels = static_cast<int>(32 * scale_factor_);
        interpolated_altitudes_ = interpolateAltitudesCubic(target_channels);
    }
    
    std::vector<float> interpolateAltitudesCubic(int target_size) {
        std::vector<float> result(target_size);
        
        // 큐빅 보간을 위한 계수 계산
        for (int i = 0; i < target_size; ++i) {
            float pos = i * (beam_altitudes_rad_.size() - 1) / static_cast<float>(target_size - 1);
            int idx = static_cast<int>(pos);
            float t = pos - idx;
            
            if (idx >= beam_altitudes_rad_.size() - 1) {
                result[i] = beam_altitudes_rad_.back();
            } else if (interpolation_method_ == "cubic" && idx > 0 && idx < beam_altitudes_rad_.size() - 2) {
                // Catmull-Rom 큐빅 보간
                float p0 = beam_altitudes_rad_[idx - 1];
                float p1 = beam_altitudes_rad_[idx];
                float p2 = beam_altitudes_rad_[idx + 1];
                float p3 = beam_altitudes_rad_[idx + 2];
                
                float t2 = t * t;
                float t3 = t2 * t;
                
                result[i] = 0.5f * ((2.0f * p1) +
                                   (-p0 + p2) * t +
                                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
            } else {
                // 선형 보간
                result[i] = beam_altitudes_rad_[idx] * (1.0f - t) + 
                           beam_altitudes_rad_[std::min(idx + 1, static_cast<int>(beam_altitudes_rad_.size() - 1))] * t;
            }
        }
        
        return result;
    }
    
    void rangeImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // OpenCV로 보간
            cv::resize(cv_ptr->image, interpolated_range_image_, 
                      cv::Size(msg->width, static_cast<int>(msg->height * scale_factor_)), 
                      0, 0, cv::INTER_CUBIC);
            
            has_range_image_ = true;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void signalImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // OpenCV로 보간
            cv::resize(cv_ptr->image, interpolated_signal_image_, 
                      cv::Size(msg->width, static_cast<int>(msg->height * scale_factor_)), 
                      0, 0, cv::INTER_CUBIC);
            
            has_signal_image_ = true;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 입력 확인
        if (msg->height != 32 || msg->width != 1024) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Expected 32x1024, got %dx%d", msg->height, msg->width);
            return;
        }
        
        // PCL 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr input(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *input);
        
        // 개선된 보간
        auto output = improvedInterpolation(input);
        
        // 발행
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*output, output_msg);
        output_msg.header = msg->header;
        pub_->publish(output_msg);
        
        // 통계
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        frame_count_++;
        total_processing_time_ += duration.count();
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr improvedInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        int output_height = static_cast<int>(input->height * scale_factor_);
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        output->width = input->width;
        output->height = output_height;
        output->points.resize(output_height * input->width);
        output->is_dense = false;
        
        // 각 열에 대해 처리
        for (size_t col = 0; col < input->width; ++col) {
            // 1. 원본 포인트 복사
            for (size_t row = 0; row < input->height; ++row) {
                size_t input_idx = row * input->width + col;
                size_t output_row = static_cast<size_t>(row * scale_factor_);
                size_t output_idx = output_row * input->width + col;
                
                // 원본 포인트는 그대로 복사
                output->points[output_idx] = input->points[input_idx];
            }
            
            // 2. 보간 포인트 생성
            for (size_t row = 0; row < input->height - 1; ++row) {
                size_t idx1 = row * input->width + col;
                size_t idx2 = (row + 1) * input->width + col;
                const auto& p1 = input->points[idx1];
                const auto& p2 = input->points[idx2];
                
                // 유효성 검사
                if (!std::isfinite(p1.x) || !std::isfinite(p2.x)) continue;
                
                // 거리 계산
                float r1 = std::sqrt(p1.x*p1.x + p1.y*p1.y + p1.z*p1.z);
                float r2 = std::sqrt(p2.x*p2.x + p2.y*p2.y + p2.z*p2.z);
                
                // 불연속성 검사
                bool is_discontinuous = std::abs(r2 - r1) > 0.5f;
                
                // 보간 포인트 생성
                size_t base_output_row = static_cast<size_t>(row * scale_factor_);
                for (int i = 1; i < static_cast<int>(scale_factor_); ++i) {
                    float t = static_cast<float>(i) / scale_factor_;
                    size_t interp_row = base_output_row + i;
                    size_t interp_idx = interp_row * input->width + col;
                    
                    if (is_discontinuous) {
                        // 불연속: nearest neighbor
                        if (t < 0.5f) {
                            output->points[interp_idx] = p1;
                        } else {
                            output->points[interp_idx] = p2;
                        }
                    } else {
                        // 연속: 선형 보간 (XYZ 직접)
                        output->points[interp_idx].x = p1.x * (1.0f - t) + p2.x * t;
                        output->points[interp_idx].y = p1.y * (1.0f - t) + p2.y * t;
                        output->points[interp_idx].z = p1.z * (1.0f - t) + p2.z * t;
                        output->points[interp_idx].intensity = p1.intensity * (1.0f - t) + p2.intensity * t;
                    }
                    
                    // 이미지 특징 사용 (옵션)
                    if (use_image_features_ && has_range_image_ && has_signal_image_) {
                        updatePointFromImages(output->points[interp_idx], interp_row, col);
                    }
                }
            }
        }
        
        // 마지막 행들 처리
        size_t last_input_row = input->height - 1;
        size_t last_output_row = static_cast<size_t>(last_input_row * scale_factor_);
        for (size_t row = last_output_row + 1; row < output->height; ++row) {
            for (size_t col = 0; col < output->width; ++col) {
                size_t idx = row * output->width + col;
                size_t last_idx = last_output_row * output->width + col;
                output->points[idx] = output->points[last_idx];
            }
        }
        
        return output;
    }
    
    void updatePointFromImages(pcl::PointXYZI& point, size_t row, size_t col) {
        // 이미지에서 추가 정보 가져오기
        if (row < interpolated_signal_image_.rows && col < interpolated_signal_image_.cols) {
            // Signal 이미지에서 intensity 업데이트
            if (interpolated_signal_image_.type() == CV_16U) {
                point.intensity = interpolated_signal_image_.at<uint16_t>(row, col) / 256.0f;
            } else if (interpolated_signal_image_.type() == CV_8U) {
                point.intensity = interpolated_signal_image_.at<uint8_t>(row, col);
            }
        }
    }
    
    void printStats() {
        if (frame_count_ > 0) {
            double avg_time = static_cast<double>(total_processing_time_) / frame_count_;
            double fps = 1000.0 / avg_time;
            
            RCLCPP_INFO(this->get_logger(), 
                "=== Improved Interpolation Statistics ===\n"
                "  Frames: %ld, Avg: %.2f ms, FPS: %.1f\n"
                "  Image features: %s (Range: %s, Signal: %s)",
                frame_count_, avg_time, fps,
                use_image_features_ ? "Enabled" : "Disabled",
                has_range_image_ ? "Ready" : "Waiting",
                has_signal_image_ ? "Ready" : "Waiting");
        }
    }
    
    // ROS2 인터페이스
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr range_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr signal_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // 파라미터
    double scale_factor_;
    bool use_image_features_;
    std::string interpolation_method_;
    
    // 고도각
    std::vector<float> beam_altitudes_rad_;
    std::vector<float> interpolated_altitudes_;
    
    // 이미지 데이터
    cv::Mat interpolated_range_image_;
    cv::Mat interpolated_signal_image_;
    bool has_range_image_ = false;
    bool has_signal_image_ = false;
    
    // 통계
    size_t frame_count_ = 0;
    long total_processing_time_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImprovedInterpolationNode>());
    rclcpp::shutdown();
    return 0;
}