#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <deque>
#include <numeric>
#include <algorithm>

class ImuPreprocessNode : public rclcpp::Node
{
public:
  ImuPreprocessNode() : Node("imu_preprocess_node")
  {
    /* ---------- 파라미터 ---------- */
    calib_duration_ = declare_parameter<double>("calib_duration", 20.0); // 정적 캘리브레이션 시간 [s]
    lpf_cutoff_     = declare_parameter<double>("lpf_cutoff",    15.0);  // 1차 IIR LPF 컷오프 [Hz]
    
    // 패키지 share 디렉토리를 찾아서 기본 경로 설정
    std::string default_calib_path;
    try {
      std::string package_share_dir = ament_index_cpp::get_package_share_directory("imu_preprocess");
      default_calib_path = package_share_dir + "/config/improved_imu_calibration.json";
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Failed to get package share directory: %s. Using relative path.", e.what());
      default_calib_path = "config/improved_imu_calibration.json";
    }
    
    calib_file_path_ = declare_parameter<std::string>("calibration_file", default_calib_path);
    use_json_bias_ = declare_parameter<bool>("use_json_bias", true);  // JSON 파일의 바이어스 사용 여부
    use_adaptive_filter_ = declare_parameter<bool>("use_adaptive_filter", true);  // Allan variance 기반 필터 사용
    bias_window_size_ = declare_parameter<int>("bias_window_size", 100);  // 동적 바이어스 추정 윈도우 크기

    /* ---------- 캘리브레이션 파일 로드 ---------- */
    if (!loadCalibrationData(calib_file_path_)) {
      RCLCPP_WARN(get_logger(), "Failed to load calibration file. Using runtime calibration.");
      use_json_bias_ = false;
    }

    /* ---------- 토픽 I/O ---------- */
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      std::bind(&ImuPreprocessNode::imuCallback, this, std::placeholders::_1));

    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
      "/imu/processed", rclcpp::SensorDataQoS());

    start_time_ = now();

    /* ---------- 초기화 ---------- */
    if (use_json_bias_) {
      calibrated_ = true;  // JSON 바이어스 사용 시 즉시 캘리브레이션 완료
      RCLCPP_INFO(get_logger(), "Using calibration from JSON file");
    }
  }

private:
  /* ===== IMU 콜백 ===== */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const double dt =
      (last_time_.nanoseconds() == 0) ? 0.01
      : (now() - last_time_).seconds();
    last_time_ = now();

    /* 1) 정적 캘리브레이션 단계 ------------------------- */
    if (!calibrated_) {
      accumulateBias(*msg);
      if ((now() - start_time_).seconds() >= calib_duration_) {
        finalizeBias();     // 편향 확정 + 로그 1회 출력
        calibrated_ = true;
      }
      return;               // 아직 퍼블리시하지 않음
    }

    /* 2) 편향 제거 -------------------------------------- */
    sensor_msgs::msg::Imu out = *msg;
    out.linear_acceleration.x -= bias_acc_[0];
    out.linear_acceleration.y -= bias_acc_[1];
    out.linear_acceleration.z -= bias_acc_[2];
    out.angular_velocity.x    -= bias_gyro_[0];
    out.angular_velocity.y    -= bias_gyro_[1];
    out.angular_velocity.z    -= bias_gyro_[2];

    /* 3) 동적 바이어스 추정 (Bias Stability 활용) ------- */
    if (use_adaptive_filter_ && bias_window_size_ > 0) {
      updateDynamicBias(out);
    }

    /* 4) Allan Variance 기반 Adaptive Noise Filtering --- */
    if (use_adaptive_filter_ && !allan_params_.empty()) {
      applyAdaptiveNoiseFilter(out, dt);
    } else if (lpf_cutoff_ > 0.0) {
      /* 기존 1차 IIR 저역통과필터 (fallback) ------------ */
      const double tau   = 1.0 / (2.0 * M_PI * lpf_cutoff_);
      const double alpha = dt / (tau + dt);

      auto lpf = [alpha](double x, double &prev) {
        double y = alpha * x + (1.0 - alpha) * prev;
        prev = y;
        return y;
      };

      out.linear_acceleration.x = lpf(out.linear_acceleration.x, acc_prev_[0]);
      out.linear_acceleration.y = lpf(out.linear_acceleration.y, acc_prev_[1]);
      out.linear_acceleration.z = lpf(out.linear_acceleration.z, acc_prev_[2]);
      out.angular_velocity.x    = lpf(out.angular_velocity.x,    gyro_prev_[0]);
      out.angular_velocity.y    = lpf(out.angular_velocity.y,    gyro_prev_[1]);
      out.angular_velocity.z    = lpf(out.angular_velocity.z,    gyro_prev_[2]);
    }

    /* 5) 공분산 업데이트 (Allan variance 정보 활용) ----- */
    if (use_adaptive_filter_) {
      updateCovariance(out);
    }

    imu_pub_->publish(out);
  }

  /* ===== 편향 누적 ===== */
  void accumulateBias(const sensor_msgs::msg::Imu &m)
  {
    sum_acc_[0]  += m.linear_acceleration.x;
    sum_acc_[1]  += m.linear_acceleration.y;
    sum_acc_[2]  += m.linear_acceleration.z - 9.81;         // 중력 제거(Z)
    sum_gyro_[0] += m.angular_velocity.x;
    sum_gyro_[1] += m.angular_velocity.y;
    sum_gyro_[2] += m.angular_velocity.z;
    ++sample_cnt_;
  }

  /* ===== 편향 확정 + 단 1회 로그 ===== */
  void finalizeBias()
  {
    for (int i = 0; i < 3; ++i) {
      bias_acc_[i]  = sum_acc_[i]  / sample_cnt_;
      bias_gyro_[i] = sum_gyro_[i] / sample_cnt_;
    }
    RCLCPP_INFO(get_logger(),
      "IMU bias calibrated:\n"
      "  acc  = [%.4f, %.4f, %.4f] m/s²\n"
      "  gyro = [%.4f, %.4f, %.4f] rad/s",
      bias_acc_[0],  bias_acc_[1],  bias_acc_[2],
      bias_gyro_[0], bias_gyro_[1], bias_gyro_[2]);
  }

  /* ===== 캘리브레이션 데이터 로드 ===== */
  bool loadCalibrationData(const std::string& path)
  {
    try {
      std::ifstream file(path);
      if (!file.is_open()) {
        RCLCPP_ERROR(get_logger(), "Cannot open calibration file: %s", path.c_str());
        return false;
      }

      nlohmann::json calib_data;
      file >> calib_data;

      // 바이어스 로드
      if (calib_data.contains("bias_estimation")) {
        auto bias = calib_data["bias_estimation"];
        bias_acc_[0] = bias["accel_bias"]["x"];
        bias_acc_[1] = bias["accel_bias"]["y"];
        bias_acc_[2] = bias["accel_bias"]["z"];
        bias_gyro_[0] = bias["gyro_bias"]["x"];
        bias_gyro_[1] = bias["gyro_bias"]["y"];
        bias_gyro_[2] = bias["gyro_bias"]["z"];
      }

      // 표준편차 로드
      if (calib_data.contains("statistics")) {
        auto stats = calib_data["statistics"];
        accel_std_[0] = stats["accel_std"][0];
        accel_std_[1] = stats["accel_std"][1];
        accel_std_[2] = stats["accel_std"][2];
        gyro_std_[0] = stats["gyro_std"][0];
        gyro_std_[1] = stats["gyro_std"][1];
        gyro_std_[2] = stats["gyro_std"][2];
      }

      // Allan variance 파라미터 로드
      if (calib_data.contains("allan_variance")) {
        allan_params_["accel_bias_stability"] = {
          calib_data["allan_variance"]["accel"]["bias_stability"].get<std::string>()
        };
        allan_params_["gyro_bias_stability"] = {
          calib_data["allan_variance"]["gyro"]["bias_stability"].get<std::string>()
        };
        
        // bias stability 값 파싱
        parseBiasStability(allan_params_["accel_bias_stability"][0], accel_bias_stability_);
        parseBiasStability(allan_params_["gyro_bias_stability"][0], gyro_bias_stability_);
      }

      RCLCPP_INFO(get_logger(), 
        "Loaded calibration:\n"
        "  Accel bias: [%.6f, %.6f, %.6f]\n"
        "  Gyro bias: [%.6f, %.6f, %.6f]\n"
        "  Accel std: [%.6f, %.6f, %.6f]\n"
        "  Gyro std: [%.6f, %.6f, %.6f]",
        bias_acc_[0], bias_acc_[1], bias_acc_[2],
        bias_gyro_[0], bias_gyro_[1], bias_gyro_[2],
        accel_std_[0], accel_std_[1], accel_std_[2],
        gyro_std_[0], gyro_std_[1], gyro_std_[2]);

      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Error loading calibration: %s", e.what());
      return false;
    }
  }

  /* ===== Bias Stability 문자열 파싱 ===== */
  void parseBiasStability(const std::string& str, std::array<double, 3>& result)
  {
    // "[0.00376809 0.00333192 0.00439254]" 형식을 파싱
    std::string cleaned = str;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    
    std::istringstream iss(cleaned);
    iss >> result[0] >> result[1] >> result[2];
  }

  /* ===== 동적 바이어스 추정 ===== */
  void updateDynamicBias(sensor_msgs::msg::Imu& msg)
  {
    // 가속도계 버퍼 업데이트
    accel_buffer_x_.push_back(msg.linear_acceleration.x);
    accel_buffer_y_.push_back(msg.linear_acceleration.y);
    accel_buffer_z_.push_back(msg.linear_acceleration.z);
    
    // 자이로 버퍼 업데이트
    gyro_buffer_x_.push_back(msg.angular_velocity.x);
    gyro_buffer_y_.push_back(msg.angular_velocity.y);
    gyro_buffer_z_.push_back(msg.angular_velocity.z);
    
    // 버퍼 크기 제한
    while (accel_buffer_x_.size() > static_cast<size_t>(bias_window_size_)) {
      accel_buffer_x_.pop_front();
      accel_buffer_y_.pop_front();
      accel_buffer_z_.pop_front();
      gyro_buffer_x_.pop_front();
      gyro_buffer_y_.pop_front();
      gyro_buffer_z_.pop_front();
    }
    
    // Bias stability를 고려한 동적 바이어스 조정
    if (accel_buffer_x_.size() >= static_cast<size_t>(bias_window_size_)) {
      // 현재 윈도우의 평균 계산
      double acc_mean_x = std::accumulate(accel_buffer_x_.begin(), accel_buffer_x_.end(), 0.0) / accel_buffer_x_.size();
      double acc_mean_y = std::accumulate(accel_buffer_y_.begin(), accel_buffer_y_.end(), 0.0) / accel_buffer_y_.size();
      double acc_mean_z = std::accumulate(accel_buffer_z_.begin(), accel_buffer_z_.end(), 0.0) / accel_buffer_z_.size();
      
      // Bias stability 범위 내에서만 바이어스 조정
      if (std::abs(acc_mean_x - bias_acc_[0]) < accel_bias_stability_[0] * 3.0) {
        bias_acc_[0] = 0.99 * bias_acc_[0] + 0.01 * acc_mean_x;
      }
      if (std::abs(acc_mean_y - bias_acc_[1]) < accel_bias_stability_[1] * 3.0) {
        bias_acc_[1] = 0.99 * bias_acc_[1] + 0.01 * acc_mean_y;
      }
      if (std::abs(acc_mean_z - bias_acc_[2]) < accel_bias_stability_[2] * 3.0) {
        bias_acc_[2] = 0.99 * bias_acc_[2] + 0.01 * acc_mean_z;
      }
      
      // 자이로도 동일하게 처리
      double gyro_mean_x = std::accumulate(gyro_buffer_x_.begin(), gyro_buffer_x_.end(), 0.0) / gyro_buffer_x_.size();
      double gyro_mean_y = std::accumulate(gyro_buffer_y_.begin(), gyro_buffer_y_.end(), 0.0) / gyro_buffer_y_.size();
      double gyro_mean_z = std::accumulate(gyro_buffer_z_.begin(), gyro_buffer_z_.end(), 0.0) / gyro_buffer_z_.size();
      
      if (std::abs(gyro_mean_x - bias_gyro_[0]) < gyro_bias_stability_[0] * 3.0) {
        bias_gyro_[0] = 0.99 * bias_gyro_[0] + 0.01 * gyro_mean_x;
      }
      if (std::abs(gyro_mean_y - bias_gyro_[1]) < gyro_bias_stability_[1] * 3.0) {
        bias_gyro_[1] = 0.99 * bias_gyro_[1] + 0.01 * gyro_mean_y;
      }
      if (std::abs(gyro_mean_z - bias_gyro_[2]) < gyro_bias_stability_[2] * 3.0) {
        bias_gyro_[2] = 0.99 * bias_gyro_[2] + 0.01 * gyro_mean_z;
      }
    }
  }

  /* ===== Allan Variance 기반 적응형 노이즈 필터 ===== */
  void applyAdaptiveNoiseFilter(sensor_msgs::msg::Imu& msg, double dt)
  {
    // 노이즈 레벨에 따른 적응형 필터링
    // 짧은 시간 상수에서는 약한 필터링, 긴 시간 상수에서는 강한 필터링
    
    // 가속도계 필터링
    double acc_filter_alpha_x = computeFilterAlpha(accel_std_[0], accel_bias_stability_[0], dt);
    double acc_filter_alpha_y = computeFilterAlpha(accel_std_[1], accel_bias_stability_[1], dt);
    double acc_filter_alpha_z = computeFilterAlpha(accel_std_[2], accel_bias_stability_[2], dt);
    
    msg.linear_acceleration.x = acc_filter_alpha_x * msg.linear_acceleration.x + 
                                (1.0 - acc_filter_alpha_x) * acc_prev_[0];
    msg.linear_acceleration.y = acc_filter_alpha_y * msg.linear_acceleration.y + 
                                (1.0 - acc_filter_alpha_y) * acc_prev_[1];
    msg.linear_acceleration.z = acc_filter_alpha_z * msg.linear_acceleration.z + 
                                (1.0 - acc_filter_alpha_z) * acc_prev_[2];
    
    // 자이로 필터링
    double gyro_filter_alpha_x = computeFilterAlpha(gyro_std_[0], gyro_bias_stability_[0], dt);
    double gyro_filter_alpha_y = computeFilterAlpha(gyro_std_[1], gyro_bias_stability_[1], dt);
    double gyro_filter_alpha_z = computeFilterAlpha(gyro_std_[2], gyro_bias_stability_[2], dt);
    
    msg.angular_velocity.x = gyro_filter_alpha_x * msg.angular_velocity.x + 
                             (1.0 - gyro_filter_alpha_x) * gyro_prev_[0];
    msg.angular_velocity.y = gyro_filter_alpha_y * msg.angular_velocity.y + 
                             (1.0 - gyro_filter_alpha_y) * gyro_prev_[1];
    msg.angular_velocity.z = gyro_filter_alpha_z * msg.angular_velocity.z + 
                             (1.0 - gyro_filter_alpha_z) * gyro_prev_[2];
    
    // 이전 값 업데이트
    acc_prev_[0] = msg.linear_acceleration.x;
    acc_prev_[1] = msg.linear_acceleration.y;
    acc_prev_[2] = msg.linear_acceleration.z;
    gyro_prev_[0] = msg.angular_velocity.x;
    gyro_prev_[1] = msg.angular_velocity.y;
    gyro_prev_[2] = msg.angular_velocity.z;
  }

  /* ===== 필터 알파 계산 ===== */
  double computeFilterAlpha(double noise_std, double bias_stability, double dt)
  {
    // Allan variance 정보를 활용한 적응형 필터 파라미터 계산
    // 노이즈가 크고 bias stability가 좋을수록 강한 필터링
    double noise_ratio = bias_stability / (noise_std + 1e-9);
    double tau = 0.1 + (1.0 - noise_ratio) * 0.9;  // 0.1 ~ 1.0 초
    return dt / (tau + dt);
  }

  /* ===== 공분산 업데이트 ===== */
  void updateCovariance(sensor_msgs::msg::Imu& msg)
  {
    // Allan variance 정보를 활용하여 IMU 공분산 설정
    // 가속도계 공분산
    msg.linear_acceleration_covariance[0] = accel_std_[0] * accel_std_[0];  // x
    msg.linear_acceleration_covariance[4] = accel_std_[1] * accel_std_[1];  // y
    msg.linear_acceleration_covariance[8] = accel_std_[2] * accel_std_[2];  // z
    
    // 자이로 공분산
    msg.angular_velocity_covariance[0] = gyro_std_[0] * gyro_std_[0];  // x
    msg.angular_velocity_covariance[4] = gyro_std_[1] * gyro_std_[1];  // y
    msg.angular_velocity_covariance[8] = gyro_std_[2] * gyro_std_[2];  // z
    
    // 방향 공분산은 기본값 사용 (IMU에서 방향은 추정하지 않음)
    msg.orientation_covariance[0] = -1;  // 사용하지 않음 표시
  }

  /* ===== 멤버 ===== */
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr     imu_pub_;
  rclcpp::Time start_time_, last_time_;

  double calib_duration_;
  double lpf_cutoff_;
  bool calibrated_{false};
  
  // 새로운 멤버 변수들
  std::string calib_file_path_;
  bool use_json_bias_;
  bool use_adaptive_filter_;
  int bias_window_size_;
  
  // 캘리브레이션 데이터
  std::array<double,3> accel_std_{}, gyro_std_{};
  std::array<double,3> accel_bias_stability_{}, gyro_bias_stability_{};
  std::map<std::string, std::vector<std::string>> allan_params_;
  
  // 동적 바이어스 추정을 위한 버퍼
  std::deque<double> accel_buffer_x_, accel_buffer_y_, accel_buffer_z_;
  std::deque<double> gyro_buffer_x_, gyro_buffer_y_, gyro_buffer_z_;

  std::array<double,3> sum_acc_{}, sum_gyro_{};
  std::array<double,3> bias_acc_{}, bias_gyro_{};
  std::array<double,3> acc_prev_{}, gyro_prev_{};
  size_t sample_cnt_{0};
};

/* ===== 메인 ===== */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuPreprocessNode>());
  rclcpp::shutdown();
  return 0;
}
