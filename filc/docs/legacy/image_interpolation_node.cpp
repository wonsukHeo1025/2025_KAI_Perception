#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>
#include <cmath>

class ImageInterpolationNode : public rclcpp::Node {
public:
    ImageInterpolationNode() : Node("image_interpolation_node") {
        // 파라미터 선언
        this->declare_parameter("scale_factor", 4.0);
        this->declare_parameter("interpolation_method", "bicubic");
        this->declare_parameter("process_range", true);
        this->declare_parameter("process_intensity", true);
        this->declare_parameter("process_pointcloud", true);
        
        // 파라미터 가져오기
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        interpolation_method_ = this->get_parameter("interpolation_method").as_string();
        process_range_ = this->get_parameter("process_range").as_bool();
        process_intensity_ = this->get_parameter("process_intensity").as_bool();
        process_pointcloud_ = this->get_parameter("process_pointcloud").as_bool();
        
        // OpenCV 보간 방법 설정
        if (interpolation_method_ == "linear") {
            cv_interpolation_ = cv::INTER_LINEAR;
        } else if (interpolation_method_ == "cubic") {
            cv_interpolation_ = cv::INTER_CUBIC;
        } else if (interpolation_method_ == "lanczos") {
            cv_interpolation_ = cv::INTER_LANCZOS4;
        } else {
            cv_interpolation_ = cv::INTER_CUBIC;  // 기본값: bicubic
        }
        
        // 구독자
        if (process_range_) {
            range_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/ouster/range_image", rclcpp::QoS(10).best_effort(),
                std::bind(&ImageInterpolationNode::rangeImageCallback, this, std::placeholders::_1));
        }
        
        if (process_intensity_) {
            intensity_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/ouster/signal_image", rclcpp::QoS(10).best_effort(),
                std::bind(&ImageInterpolationNode::intensityImageCallback, this, std::placeholders::_1));
        }
        
        if (process_pointcloud_) {
            pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/ouster/points", rclcpp::QoS(10).best_effort(),
                std::bind(&ImageInterpolationNode::pointCloudCallback, this, std::placeholders::_1));
        }
        
        // 발행자
        if (process_range_) {
            range_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/ouster/interpolated_range_image", rclcpp::QoS(10).reliable());
        }
        
        if (process_intensity_) {
            intensity_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/ouster/interpolated_intensity_image", rclcpp::QoS(10).reliable());
        }
        
        if (process_pointcloud_) {
            pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/ouster/interpolated_points_from_images", rclcpp::QoS(10).reliable());
        }
        
        // 통계 타이머
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&ImageInterpolationNode::printStats, this));
        
        // Ouster OS1-32 고도각 (라디안)
        initializeBeamAngles();
        
        RCLCPP_INFO(this->get_logger(), 
            "Image Interpolation Node Started\n"
            "  Scale Factor: %.1fx\n"
            "  Method: %s\n"
            "  Processing: %s%s%s",
            scale_factor_, interpolation_method_.c_str(),
            process_range_ ? "Range " : "",
            process_intensity_ ? "Intensity " : "",
            process_pointcloud_ ? "PointCloud" : "");
    }

private:
    void initializeBeamAngles() {
        // Ouster OS1-32 빔 고도각 (도)
        std::vector<float> beam_angles_deg = {
            -16.611f, -16.084f, -15.557f, -15.029f, -14.502f, -13.975f,
            -13.447f, -12.920f, -12.393f, -11.865f, -11.338f, -10.811f,
            -10.283f, -9.756f, -9.229f, -8.701f, -8.174f, -7.646f,
            -7.119f, -6.592f, -6.064f, -5.537f, -5.010f, -4.482f,
            -3.955f, -3.428f, -2.900f, -2.373f, -1.846f, -1.318f,
            -0.791f, -0.264f
        };
        
        // 라디안으로 변환
        beam_angles_rad_.resize(beam_angles_deg.size());
        for (size_t i = 0; i < beam_angles_deg.size(); ++i) {
            beam_angles_rad_[i] = beam_angles_deg[i] * M_PI / 180.0f;
        }
    }
    
    void rangeImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // OpenCV로 변환
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // 보간 수행
            cv::Mat interpolated;
            interpolateImage(cv_ptr->image, interpolated);
            
            // ROS 메시지로 변환
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = msg->encoding;
            out_msg.image = interpolated;
            
            range_pub_->publish(*out_msg.toImageMsg());
            
            // 최신 Range 이미지 저장 (포인트클라우드 재구성용)
            latest_range_image_ = interpolated;
            latest_range_header_ = msg->header;
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        range_processing_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        range_frame_count_++;
    }
    
    void intensityImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // OpenCV로 변환
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // 보간 수행
            cv::Mat interpolated;
            interpolateImage(cv_ptr->image, interpolated);
            
            // ROS 메시지로 변환
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = msg->encoding;
            out_msg.image = interpolated;
            
            intensity_pub_->publish(*out_msg.toImageMsg());
            
            // 최신 Intensity 이미지 저장
            latest_intensity_image_ = interpolated;
            latest_intensity_header_ = msg->header;
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        intensity_processing_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        intensity_frame_count_++;
    }
    
    void interpolateImage(const cv::Mat& input, cv::Mat& output) {
        // 새로운 크기 계산
        int new_height = static_cast<int>(input.rows * scale_factor_);
        int new_width = input.cols;  // 수평은 그대로 유지
        
        // OpenCV resize 함수로 보간
        cv::resize(input, output, cv::Size(new_width, new_height), 0, 0, cv_interpolation_);
        
        // 에지 보존 처리 (선택적)
        if (interpolation_method_ == "edge_preserving") {
            applyEdgePreserving(output);
        }
    }
    
    void applyEdgePreserving(cv::Mat& image) {
        // 간단한 에지 보존 필터 (bilateral filter)
        cv::Mat temp;
        cv::bilateralFilter(image, temp, 5, 50, 50);
        image = temp;
    }
    
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Range와 Intensity 이미지가 모두 준비되었는지 확인
        if (latest_range_image_.empty() || latest_intensity_image_.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Waiting for range and intensity images...");
            return;
        }
        
        // 이미지 기반 포인트클라우드 재구성
        reconstructPointCloud(msg->header);
    }
    
    void reconstructPointCloud(const std_msgs::msg::Header& header) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 출력 포인트클라우드 생성
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        cloud->width = latest_range_image_.cols;
        cloud->height = latest_range_image_.rows;
        cloud->points.resize(cloud->width * cloud->height);
        cloud->is_dense = false;
        
        // 보간된 고도각 계산
        std::vector<float> interpolated_altitudes = interpolateAltitudes(cloud->height);
        
        // 각 픽셀을 3D 포인트로 변환
        for (int row = 0; row < cloud->height; ++row) {
            for (int col = 0; col < cloud->width; ++col) {
                int idx = row * cloud->width + col;
                
                // Range 값 가져오기
                float range = 0.0f;
                if (latest_range_image_.type() == CV_32F) {
                    range = latest_range_image_.at<float>(row, col);
                } else if (latest_range_image_.type() == CV_16U) {
                    range = latest_range_image_.at<uint16_t>(row, col) / 1000.0f; // mm to m
                }
                
                // Intensity 값 가져오기
                float intensity = 0.0f;
                if (latest_intensity_image_.type() == CV_32F) {
                    intensity = latest_intensity_image_.at<float>(row, col);
                } else if (latest_intensity_image_.type() == CV_8U) {
                    intensity = latest_intensity_image_.at<uint8_t>(row, col);
                } else if (latest_intensity_image_.type() == CV_16U) {
                    intensity = latest_intensity_image_.at<uint16_t>(row, col) / 256.0f;
                }
                
                // 유효하지 않은 포인트 처리
                if (range <= 0.0f || !std::isfinite(range)) {
                    cloud->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                    cloud->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                    cloud->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                    cloud->points[idx].intensity = 0;
                    continue;
                }
                
                // Ouster 좌표 변환
                float theta_encoder = 2.0f * M_PI * (1.0f - float(col) / float(cloud->width));
                float phi = interpolated_altitudes[row];
                float r = range - 0.015806f; // beam origin offset
                
                float cos_phi = std::cos(phi);
                cloud->points[idx].x = r * std::cos(theta_encoder) * cos_phi;
                cloud->points[idx].y = r * std::sin(theta_encoder) * cos_phi;
                cloud->points[idx].z = r * std::sin(phi);
                cloud->points[idx].intensity = intensity;
            }
        }
        
        // 발행
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = header;
        output.header.frame_id = "os_sensor";
        pointcloud_pub_->publish(output);
        
        auto end = std::chrono::high_resolution_clock::now();
        pointcloud_processing_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        pointcloud_frame_count_++;
    }
    
    std::vector<float> interpolateAltitudes(int target_size) {
        std::vector<float> result(target_size);
        
        // 선형 보간 (나중에 큐빅 스플라인으로 개선 예정)
        float scale = static_cast<float>(beam_angles_rad_.size() - 1) / (target_size - 1);
        
        for (int i = 0; i < target_size; ++i) {
            float pos = i * scale;
            int lower = static_cast<int>(pos);
            int upper = std::min(lower + 1, static_cast<int>(beam_angles_rad_.size() - 1));
            float t = pos - lower;
            
            result[i] = beam_angles_rad_[lower] * (1.0f - t) + beam_angles_rad_[upper] * t;
        }
        
        return result;
    }
    
    void printStats() {
        RCLCPP_INFO(this->get_logger(), "=== Image Interpolation Statistics ===");
        
        if (range_frame_count_ > 0) {
            double avg_time = static_cast<double>(range_processing_time_) / range_frame_count_;
            RCLCPP_INFO(this->get_logger(), 
                "Range Image: %ld frames, avg %.2f ms", range_frame_count_, avg_time);
        }
        
        if (intensity_frame_count_ > 0) {
            double avg_time = static_cast<double>(intensity_processing_time_) / intensity_frame_count_;
            RCLCPP_INFO(this->get_logger(), 
                "Intensity Image: %ld frames, avg %.2f ms", intensity_frame_count_, avg_time);
        }
        
        if (pointcloud_frame_count_ > 0) {
            double avg_time = static_cast<double>(pointcloud_processing_time_) / pointcloud_frame_count_;
            RCLCPP_INFO(this->get_logger(), 
                "PointCloud Reconstruction: %ld frames, avg %.2f ms", pointcloud_frame_count_, avg_time);
        }
    }
    
    // ROS2 인터페이스
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr range_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr intensity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr range_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr intensity_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // 파라미터
    double scale_factor_;
    std::string interpolation_method_;
    int cv_interpolation_;
    bool process_range_;
    bool process_intensity_;
    bool process_pointcloud_;
    
    // 최신 이미지 저장
    cv::Mat latest_range_image_;
    cv::Mat latest_intensity_image_;
    std_msgs::msg::Header latest_range_header_;
    std_msgs::msg::Header latest_intensity_header_;
    
    // 빔 고도각
    std::vector<float> beam_angles_rad_;
    
    // 통계
    size_t range_frame_count_ = 0;
    size_t intensity_frame_count_ = 0;
    size_t pointcloud_frame_count_ = 0;
    long range_processing_time_ = 0;
    long intensity_processing_time_ = 0;
    long pointcloud_processing_time_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageInterpolationNode>());
    rclcpp::shutdown();
    return 0;
}