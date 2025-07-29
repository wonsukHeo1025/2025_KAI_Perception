#ifndef IMU_CALIBRATION_V2_HPP
#define IMU_CALIBRATION_V2_HPP

#include <string>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <rclcpp/rclcpp.hpp>

namespace kai {

struct ImuCalibrationData {
    struct Bias {
        double x;
        double y; 
        double z;
    };
    
    struct Statistics {
        std::vector<double> accel_std;
        std::vector<double> gyro_std;
        std::vector<double> bias_stability_accel;
        std::vector<double> bias_stability_gyro;
    };
    
    Bias accel_bias;
    Bias gyro_bias;
    Statistics stats;
    double measured_gravity;
    double collection_duration;
    int sample_count;
};

class ImuCalibrationLoader {
public:
    static std::optional<ImuCalibrationData> loadFromFile(const std::string& filepath) {
        try {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                RCLCPP_ERROR(rclcpp::get_logger("ImuCalibration"), 
                    "Calibration 파일을 열 수 없습니다: %s", filepath.c_str());
                return std::nullopt;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            
            ImuCalibrationData data;
            
            // bias_estimation 섹션 파싱
            data.accel_bias.x = extractNumber(content, "\"accel_bias\"[^}]*\"x\":\\s*([+-]?\\d+\\.?\\d*)");
            data.accel_bias.y = extractNumber(content, "\"accel_bias\"[^}]*\"y\":\\s*([+-]?\\d+\\.?\\d*)");
            data.accel_bias.z = extractNumber(content, "\"accel_bias\"[^}]*\"z\":\\s*([+-]?\\d+\\.?\\d*)");
            
            data.gyro_bias.x = extractNumber(content, "\"gyro_bias\"[^}]*\"x\":\\s*([+-]?\\d+\\.?\\d*)");
            data.gyro_bias.y = extractNumber(content, "\"gyro_bias\"[^}]*\"y\":\\s*([+-]?\\d+\\.?\\d*)");
            data.gyro_bias.z = extractNumber(content, "\"gyro_bias\"[^}]*\"z\":\\s*([+-]?\\d+\\.?\\d*)");
            
            // statistics 섹션 파싱
            data.measured_gravity = extractNumber(content, "\"measured_gravity\":\\s*([+-]?\\d+\\.?\\d*)");
            data.stats.accel_std = extractNumberArray(content, "\"accel_std\":\\s*\\[([^\\]]+)\\]");
            data.stats.gyro_std = extractNumberArray(content, "\"gyro_std\":\\s*\\[([^\\]]+)\\]");
            
            // allan_variance 섹션 파싱
            std::string accel_stability_str = extractString(content, "\"accel\"[^}]*\"bias_stability\":\\s*\"([^\"]+)\"");
            std::string gyro_stability_str = extractString(content, "\"gyro\"[^}]*\"bias_stability\":\\s*\"([^\"]+)\"");
            data.stats.bias_stability_accel = parseVectorString(accel_stability_str);
            data.stats.bias_stability_gyro = parseVectorString(gyro_stability_str);
            
            // collection_info 섹션 파싱
            data.collection_duration = extractNumber(content, "\"duration\":\\s*([+-]?\\d+\\.?\\d*)");
            data.sample_count = static_cast<int>(extractNumber(content, "\"sample_count\":\\s*([+-]?\\d+\\.?\\d*)"));
            
            RCLCPP_INFO(rclcpp::get_logger("ImuCalibration"), 
                "Calibration 데이터 로드 성공: %.0f초 동안 %d 샘플 수집",
                data.collection_duration, data.sample_count);
            RCLCPP_INFO(rclcpp::get_logger("ImuCalibration"),
                "가속도계 bias: [%.6f, %.6f, %.6f] m/s²",
                data.accel_bias.x, data.accel_bias.y, data.accel_bias.z);
            RCLCPP_INFO(rclcpp::get_logger("ImuCalibration"),
                "자이로 bias: [%.6f, %.6f, %.6f] rad/s", 
                data.gyro_bias.x, data.gyro_bias.y, data.gyro_bias.z);
            
            return data;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("ImuCalibration"),
                "Calibration 파일 파싱 오류: %s", e.what());
            return std::nullopt;
        }
    }
    
private:
    static double extractNumber(const std::string& content, const std::string& pattern) {
        std::regex regex(pattern);
        std::smatch match;
        if (std::regex_search(content, match, regex)) {
            return std::stod(match[1]);
        }
        return 0.0;
    }
    
    static std::string extractString(const std::string& content, const std::string& pattern) {
        std::regex regex(pattern);
        std::smatch match;
        if (std::regex_search(content, match, regex)) {
            return match[1];
        }
        return "";
    }
    
    static std::vector<double> extractNumberArray(const std::string& content, const std::string& pattern) {
        std::vector<double> result;
        std::regex regex(pattern);
        std::smatch match;
        if (std::regex_search(content, match, regex)) {
            std::string array_content = match[1];
            std::regex num_regex("([+-]?\\d+\\.?\\d*[eE]?[+-]?\\d*)");
            std::sregex_iterator it(array_content.begin(), array_content.end(), num_regex);
            std::sregex_iterator end;
            
            for (; it != end; ++it) {
                result.push_back(std::stod(it->str()));
            }
        }
        return result;
    }
    
    static std::vector<double> parseVectorString(const std::string& str) {
        std::vector<double> result;
        std::string cleaned = str;
        
        // 대괄호 제거
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
        
        // 공백으로 분리하여 숫자 추출
        std::stringstream ss(cleaned);
        double value;
        while (ss >> value) {
            result.push_back(value);
        }
        
        return result;
    }
};

} // namespace kai

#endif // IMU_CALIBRATION_V2_HPP