#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "filc/spherical_interpolator.hpp"
#include <chrono>
#include <vector>
#include <omp.h>

class SphericalInterpolationNode : public rclcpp::Node {
public:
    SphericalInterpolationNode() : Node("spherical_interpolation_node") {
        // 파라미터 선언
        this->declare_parameter("input_topic", "/ouster/points");
        this->declare_parameter("output_topic", "/ouster/spherical_interpolated_points");
        this->declare_parameter("scale_factor", 4.0);
        this->declare_parameter("use_adaptive_interpolation", true);
        this->declare_parameter("validate_interpolation", true);
        this->declare_parameter("variance_threshold", 0.25);
        this->declare_parameter("num_threads", 0); // 0 = auto
        
        // 파라미터 가져오기
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        use_adaptive_ = this->get_parameter("use_adaptive_interpolation").as_bool();
        validate_interpolation_ = this->get_parameter("validate_interpolation").as_bool();
        variance_threshold_ = this->get_parameter("variance_threshold").as_double();
        int num_threads = this->get_parameter("num_threads").as_int();
        
        // OpenMP 스레드 설정
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        // 보간된 고도각 미리 계산
        int target_channels = static_cast<int>(32 * scale_factor_);
        interpolated_altitudes_ = interpolator_.interpolateAltitudesCubicSpline(target_channels);
        
        // 구독자
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, rclcpp::QoS(10).best_effort(),
            std::bind(&SphericalInterpolationNode::pointCloudCallback, this, std::placeholders::_1));
        
        // 발행자
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic, rclcpp::QoS(10).reliable());
        
        // 통계 타이머
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&SphericalInterpolationNode::printStats, this));
        
        RCLCPP_INFO(this->get_logger(), 
            "Spherical Interpolation Node Started\n"
            "  Input: %s\n"
            "  Output: %s\n"
            "  Scale Factor: %.1fx (32->%d channels)\n"
            "  Method: Cubic Spline (altitude) + %s (range)\n"
            "  Validation: %s",
            input_topic.c_str(), output_topic.c_str(), 
            scale_factor_, target_channels,
            use_adaptive_ ? "Adaptive" : "Linear",
            validate_interpolation_ ? "Enabled" : "Disabled");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 입력 확인
        if (msg->height != 32 || msg->width != 1024) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Expected 32x1024 organized cloud, got %dx%d", msg->height, msg->width);
            return;
        }
        
        // PCL 변환
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *input_cloud);
        
        // 구면 좌표계 기반 보간
        auto interpolated = sphericalInterpolation(input_cloud);
        
        // 발행
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*interpolated, output);
        output.header = msg->header;
        pub_->publish(output);
        
        // 통계 업데이트
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        frame_count_++;
        total_processing_time_ += duration.count();
        
        // 상세 시간 측정
        if (frame_count_ % 10 == 0) {
            RCLCPP_DEBUG(this->get_logger(), 
                "Processing times - Total: %ld ms, Convert: %.1f ms, Interp: %.1f ms, Validate: %.1f ms",
                duration.count(), 
                conversion_time_ / 10.0, 
                interpolation_time_ / 10.0,
                validation_time_ / 10.0);
            conversion_time_ = 0;
            interpolation_time_ = 0;
            validation_time_ = 0;
        }
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr sphericalInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        int output_height = static_cast<int>(input->height * scale_factor_);
        int output_width = input->width;
        
        // 출력 포인트클라우드 생성
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>);
        output->width = output_width;
        output->height = output_height;
        output->points.resize(output_height * output_width);
        output->is_dense = false;
        
        // 1단계: 구면 좌표 변환
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<filc::SphericalPoint>> spherical_grid(32, 
            std::vector<filc::SphericalPoint>(1024));
        
        #pragma omp parallel for
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 1024; ++col) {
                int idx = row * 1024 + col;
                const auto& p = input->points[idx];
                spherical_grid[row][col] = interpolator_.cartesianToSpherical(
                    p.x, p.y, p.z, p.intensity);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        conversion_time_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        
        // 2단계: 보간 수행
        t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int col = 0; col < output_width; ++col) {
            for (int row = 0; row < 32; ++row) {
                // 원본 포인트
                int out_row = static_cast<int>(row * scale_factor_);
                const auto& sp = spherical_grid[row][col];
                
                if (sp.valid) {
                    interpolator_.sphericalToCartesianOuster(sp, col, output_width,
                        output->points[out_row * output_width + col].x,
                        output->points[out_row * output_width + col].y,
                        output->points[out_row * output_width + col].z);
                    output->points[out_row * output_width + col].intensity = sp.intensity;
                } else {
                    output->points[out_row * output_width + col].x = std::numeric_limits<float>::quiet_NaN();
                    output->points[out_row * output_width + col].y = std::numeric_limits<float>::quiet_NaN();
                    output->points[out_row * output_width + col].z = std::numeric_limits<float>::quiet_NaN();
                    output->points[out_row * output_width + col].intensity = 0;
                }
                
                // 보간 포인트
                if (row < 31) {
                    const auto& sp1 = spherical_grid[row][col];
                    const auto& sp2 = spherical_grid[row + 1][col];
                    
                    // 보간할 포인트 수
                    int num_interp = static_cast<int>(scale_factor_) - 1;
                    
                    for (int i = 1; i <= num_interp; ++i) {
                        float t = static_cast<float>(i) / scale_factor_;
                        int interp_row = out_row + i;
                        
                        if (!sp1.valid || !sp2.valid) {
                            // 유효하지 않은 포인트
                            int idx = interp_row * output_width + col;
                            output->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                            output->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                            output->points[idx].z = std::numeric_limits<float>::quiet_NaN();
                            output->points[idx].intensity = 0;
                            continue;
                        }
                        
                        // 구면 좌표 보간
                        filc::SphericalPoint interp_sp;
                        
                        // Range 보간
                        if (use_adaptive_) {
                            interp_sp.range = interpolator_.interpolateRangeAdaptive(
                                sp1.range, sp2.range, t);
                        } else {
                            interp_sp.range = sp1.range * (1.0f - t) + sp2.range * t;
                        }
                        
                        // 고도각은 미리 계산된 값 사용
                        interp_sp.altitude = interpolated_altitudes_[interp_row];
                        
                        // 방위각과 강도는 선형 보간
                        interp_sp.azimuth = sp1.azimuth; // 같은 열이므로 동일
                        interp_sp.intensity = sp1.intensity * (1.0f - t) + sp2.intensity * t;
                        interp_sp.valid = true;
                        
                        // XYZ 변환
                        int idx = interp_row * output_width + col;
                        interpolator_.sphericalToCartesianOuster(interp_sp, col, output_width,
                            output->points[idx].x,
                            output->points[idx].y,
                            output->points[idx].z);
                        output->points[idx].intensity = interp_sp.intensity;
                    }
                }
            }
        }
        
        // 마지막 행들 처리
        int last_original_row = 31;
        int last_output_row = static_cast<int>(last_original_row * scale_factor_);
        for (int row = last_output_row + 1; row < output_height; ++row) {
            for (int col = 0; col < output_width; ++col) {
                int idx = row * output_width + col;
                int last_idx = last_output_row * output_width + col;
                output->points[idx] = output->points[last_idx];
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        interpolation_time_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        
        // 3단계: 검증 (선택적)
        if (validate_interpolation_) {
            t1 = std::chrono::high_resolution_clock::now();
            validateAndCorrect(output, spherical_grid);
            t2 = std::chrono::high_resolution_clock::now();
            validation_time_ += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        }
        
        return output;
    }
    
    void validateAndCorrect(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                           const std::vector<std::vector<filc::SphericalPoint>>& original_grid) {
        int scale = static_cast<int>(scale_factor_);
        
        #pragma omp parallel for
        for (int col = 0; col < cloud->width; ++col) {
            for (int row = 0; row < cloud->height; row += scale) {
                // 원본 행은 건드리지 않음
                if (row % scale == 0) continue;
                
                // 주변 4개 행의 range 값 수집
                std::vector<float> local_ranges;
                int start_row = std::max(0, row - scale + 1);
                int end_row = std::min(static_cast<int>(cloud->height), row + scale);
                
                for (int r = start_row; r < end_row; r += scale) {
                    int orig_row = r / scale;
                    if (orig_row < 32 && original_grid[orig_row][col].valid) {
                        local_ranges.push_back(original_grid[orig_row][col].range);
                    }
                }
                
                // 분산 검증
                if (!local_ranges.empty() && 
                    !interpolator_.validateInterpolation(local_ranges, variance_threshold_)) {
                    // 고분산 영역: 가장 가까운 원본 값으로 대체
                    int nearest_original = (row / scale) * scale;
                    for (int r = row; r < std::min(row + scale, static_cast<int>(cloud->height)); ++r) {
                        if (r % scale != 0) {
                            int idx = r * cloud->width + col;
                            int orig_idx = nearest_original * cloud->width + col;
                            cloud->points[idx] = cloud->points[orig_idx];
                        }
                    }
                }
            }
        }
    }
    
    void printStats() {
        if (frame_count_ > 0) {
            double avg_time = static_cast<double>(total_processing_time_) / frame_count_;
            double fps = 1000.0 / avg_time;
            
            RCLCPP_INFO(this->get_logger(), 
                "=== Spherical Interpolation Statistics ===\n"
                "  Frames processed: %ld\n"
                "  Average time: %.2f ms\n"
                "  Average FPS: %.1f\n"
                "  Scale factor: %.1fx (32->%d channels)\n"
                "  Method: Cubic Spline + %s\n"
                "  Validation: %s (threshold: %.2f)",
                frame_count_, avg_time, fps, scale_factor_, 
                static_cast<int>(32 * scale_factor_),
                use_adaptive_ ? "Adaptive" : "Linear",
                validate_interpolation_ ? "Enabled" : "Disabled",
                variance_threshold_);
        }
    }
    
    // ROS2 인터페이스
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // 보간기
    filc::SphericalInterpolator interpolator_;
    std::vector<float> interpolated_altitudes_;
    
    // 파라미터
    double scale_factor_;
    bool use_adaptive_;
    bool validate_interpolation_;
    double variance_threshold_;
    
    // 통계
    size_t frame_count_ = 0;
    long total_processing_time_ = 0;
    double conversion_time_ = 0;
    double interpolation_time_ = 0;
    double validation_time_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SphericalInterpolationNode>());
    rclcpp::shutdown();
    return 0;
}