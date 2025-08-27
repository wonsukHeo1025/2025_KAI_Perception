#include "filc/interpolation_utils.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// 성능 최적화를 위한 하드코딩된 상수들 (무조건 적용하면 좋은 최적화)
namespace filc {
namespace PerformanceConstants {
    // 메모리 정렬 최적화 (AVX2용 32바이트 정렬)
    constexpr size_t MEMORY_ALIGNMENT = 32;
    
    // SIMD 최적화 활성화
    constexpr bool ENABLE_SIMD = true;
    
    // 캐시 최적화 - 캐시 라인 크기 (64 바이트가 일반적)
    constexpr size_t CACHE_LINE_SIZE = 64;
    
    // 스레드당 최소 처리할 포인트 수 (컨텍스트 스위칭 오버헤드 최소화)
    constexpr size_t MIN_POINTS_PER_THREAD = 1000;
    
    // 배치 처리 최적 크기 (L1 캐시에 맞는 크기)
    constexpr size_t OPTIMAL_BATCH_SIZE = 5000;
    
    // 브랜치 예측 최적화를 위한 likely/unlikely 매크로
    #ifdef __GNUC__
        #define LIKELY(x)   __builtin_expect(!!(x), 1)
        #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #else
        #define LIKELY(x)   (x)
        #define UNLIKELY(x) (x)
    #endif
    
    // 강제 인라인 (작은 함수들의 호출 오버헤드 제거)
    #ifdef __GNUC__
        #define FORCE_INLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline
    #endif
}
}

namespace filc {

InterpolationUtils::InterpolationUtils() {
}

InterpolationUtils::~InterpolationUtils() {
}

// Range 이미지만을 사용한 보간 함수 (새로 추가)
InterpolationUtils::InterpolationResult InterpolationUtils::interpolateRangeImageOnly(
    const cv::Mat& range_image,
    const InterpolationConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    InterpolationResult result;
    result.original_height = range_image.rows;
    result.interpolation_factor = config.vertical_scale_factor;
    result.interpolated_height = static_cast<size_t>(range_image.rows * config.vertical_scale_factor);
    
    // 입력 검증
    if (range_image.empty()) {
        std::cerr << "Error: Empty range image" << std::endl;
        return result;
    }
    
    // 보간 방법에 따른 처리
    switch (config.type) {
        case InterpolationType::LINEAR:
            result.interpolated_range = interpolateVerticalLinear(
                range_image, config.vertical_scale_factor, 
                config.preserve_edges, config.edge_threshold);
            break;
            
        case InterpolationType::ADAPTIVE:
            // Range 이미지만으로 적응형 보간
            result.interpolated_range = adaptiveInterpolationRangeOnly(range_image, config);
            break;
            
        case InterpolationType::CUBIC:
        case InterpolationType::BICUBIC:
        case InterpolationType::LANCZOS:
            // OpenCV의 resize 함수를 사용한 고급 보간
            {
                int interpolation_method;
                switch (config.type) {
                    case InterpolationType::CUBIC:
                        interpolation_method = cv::INTER_CUBIC;
                        break;
                    case InterpolationType::BICUBIC:
                        interpolation_method = cv::INTER_CUBIC; // OpenCV에서는 INTER_CUBIC이 bicubic
                        break;
                    case InterpolationType::LANCZOS:
                        interpolation_method = cv::INTER_LANCZOS4;
                        break;
                    default:
                        interpolation_method = cv::INTER_LINEAR;
                }
                
                cv::Size new_size(range_image.cols, 
                                static_cast<int>(range_image.rows * config.vertical_scale_factor));
                cv::resize(range_image, result.interpolated_range, new_size, 0, 0, interpolation_method);
            }
            break;
    }
    
    // 무효 픽셀 처리
    if (config.fill_invalid_pixels) {
        result.interpolated_range = fillInvalidPixels(result.interpolated_range);
    }
    
    // 유효성 마스크 생성
    result.validity_mask = cv::Mat::zeros(result.interpolated_range.size(), CV_8UC1);
    cv::threshold(result.interpolated_range, result.validity_mask, 0.1, 255, cv::THRESH_BINARY);
    
    // Intensity 이미지는 비어둠
    result.interpolated_intensity = cv::Mat();
    
    // 처리 시간 계산
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

InterpolationUtils::InterpolationResult InterpolationUtils::interpolateOusterImages(
    const cv::Mat& range_image,
    const cv::Mat& intensity_image,
    const InterpolationConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    InterpolationResult result;
    result.original_height = range_image.rows;
    result.interpolation_factor = config.vertical_scale_factor;
    result.interpolated_height = static_cast<size_t>(range_image.rows * config.vertical_scale_factor);
    
    // 입력 검증
    if (range_image.empty()) {
        std::cerr << "Error: Empty range image" << std::endl;
        return result;
    }
    
    // Intensity 이미지가 없으면 range만 처리
    if (intensity_image.empty()) {
        return interpolateRangeImageOnly(range_image, config);
    }
    
    if (range_image.size() != intensity_image.size()) {
        std::cerr << "Error: Range and intensity images have different sizes" << std::endl;
        return result;
    }
    
    // 보간 방법에 따른 처리
    switch (config.type) {
        case InterpolationType::LINEAR:
            result.interpolated_range = interpolateVerticalLinear(
                range_image, config.vertical_scale_factor, 
                config.preserve_edges, config.edge_threshold);
            result.interpolated_intensity = interpolateVerticalLinear(
                intensity_image, config.vertical_scale_factor, 
                config.preserve_edges, config.edge_threshold);
            break;
            
        case InterpolationType::ADAPTIVE:
            return adaptiveInterpolation(range_image, intensity_image, config);
            
        case InterpolationType::CUBIC:
        case InterpolationType::BICUBIC:
        case InterpolationType::LANCZOS:
            // OpenCV의 resize 함수를 사용한 고급 보간
            {
                int interpolation_method;
                switch (config.type) {
                    case InterpolationType::CUBIC:
                        interpolation_method = cv::INTER_CUBIC;
                        break;
                    case InterpolationType::BICUBIC:
                        interpolation_method = cv::INTER_CUBIC; // OpenCV에서는 INTER_CUBIC이 bicubic
                        break;
                    case InterpolationType::LANCZOS:
                        interpolation_method = cv::INTER_LANCZOS4;
                        break;
                    default:
                        interpolation_method = cv::INTER_LINEAR;
                }
                
                cv::Size new_size(range_image.cols, 
                                static_cast<int>(range_image.rows * config.vertical_scale_factor));
                cv::resize(range_image, result.interpolated_range, new_size, 0, 0, interpolation_method);
                cv::resize(intensity_image, result.interpolated_intensity, new_size, 0, 0, interpolation_method);
            }
            break;
    }
    
    // 무효 픽셀 처리
    if (config.fill_invalid_pixels) {
        result.interpolated_range = fillInvalidPixels(result.interpolated_range);
        result.interpolated_intensity = fillInvalidPixels(result.interpolated_intensity);
    }
    
    // 유효성 마스크 생성
    result.validity_mask = cv::Mat::zeros(result.interpolated_range.size(), CV_8UC1);
    cv::threshold(result.interpolated_range, result.validity_mask, 0.1, 255, cv::THRESH_BINARY);
    
    // 처리 시간 계산
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

// Range 이미지만을 위한 적응형 보간 함수 (새로 추가)
cv::Mat InterpolationUtils::adaptiveInterpolationRangeOnly(
    const cv::Mat& range_image,
    const InterpolationConfig& config) {
    
    cv::Mat result;
    cv::Size new_size(range_image.cols, 
                     static_cast<int>(range_image.rows * config.vertical_scale_factor));
    
    // 기본 선형 보간부터 시작
    cv::resize(range_image, result, new_size, 0, 0, cv::INTER_LINEAR);
    
    // Range 값 기반 적응형 가중치 적용
    for (int row = 1; row < result.rows - 1; ++row) {
        for (int col = 1; col < result.cols - 1; ++col) {
            float center_val = result.at<float>(row, col);
            
            if (center_val < 0.1f) continue; // 무효 픽셀 스킵
            
            // 주변 픽셀들의 range 값 차이를 분석
            std::vector<float> neighbors;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    float neighbor_val = result.at<float>(row + dr, col + dc);
                    if (neighbor_val > 0.1f) {
                        neighbors.push_back(neighbor_val);
                    }
                }
            }
            
            if (neighbors.empty()) continue;
            
            // 편차 계산
            float mean = std::accumulate(neighbors.begin(), neighbors.end(), 0.0f) / neighbors.size();
            float variance = 0.0f;
            for (float val : neighbors) {
                variance += (val - mean) * (val - mean);
            }
            variance /= neighbors.size();
            float std_dev = std::sqrt(variance);
            
            // 편차가 크면 (엣지 근처) 더 보수적인 보간 적용
            if (std_dev > config.edge_threshold) {
                // 가장 가까운 유효한 원본 픽셀 값 사용
                float original_row = static_cast<float>(row) / config.vertical_scale_factor;
                int nearest_original = static_cast<int>(std::round(original_row));
                if (nearest_original < range_image.rows) {
                    result.at<float>(row, col) = range_image.at<float>(nearest_original, col);
                }
            }
        }
    }
    
    return result;
}

// Helper function to create default config
InterpolationUtils::InterpolationConfig InterpolationUtils::createDefaultConfig() {
    InterpolationConfig config;
    config.type = InterpolationType::LINEAR;
    config.vertical_scale_factor = 2.0;
    config.horizontal_scale_factor = 1.0;
    config.preserve_edges = true;
    config.edge_threshold = 10.0;
    config.fill_invalid_pixels = true;
    config.max_interpolation_distance = 5.0;
    config.use_adaptive_weights = true;
    config.num_threads = 4;
    return config;
}

cv::Mat InterpolationUtils::interpolateVerticalLinear(
    const cv::Mat& image,
    double scale_factor,
    bool preserve_edges,
    double edge_threshold) {
    
    if (image.empty() || scale_factor <= 1.0) {
        return image.clone();
    }
    
    int original_height = image.rows;
    int original_width = image.cols;
    int new_height = static_cast<int>(original_height * scale_factor);
    
    // 출력 이미지 생성
    cv::Mat interpolated;
    if (image.type() == CV_32F) {
        interpolated = cv::Mat::zeros(new_height, original_width, CV_32F);
    } else if (image.type() == CV_16U) {
        interpolated = cv::Mat::zeros(new_height, original_width, CV_16U);
    } else {
        // 다른 타입은 CV_32F로 변환
        cv::Mat temp;
        image.convertTo(temp, CV_32F);
        cv::Mat result = interpolateVerticalLinear(temp, scale_factor, preserve_edges, edge_threshold);
        result.convertTo(interpolated, image.type());
        return interpolated;
    }
    
    // 원본 행들을 새로운 위치에 복사
    for (int original_row = 0; original_row < original_height; ++original_row) {
        int new_row = static_cast<int>(original_row * scale_factor);
        if (new_row < new_height) {
            image.row(original_row).copyTo(interpolated.row(new_row));
        }
    }
    
    // 중간 행들을 보간으로 채우기
    for (int new_row = 0; new_row < new_height; ++new_row) {
        double original_pos = static_cast<double>(new_row) / scale_factor;
        int lower_row = static_cast<int>(std::floor(original_pos));
        int upper_row = std::min(lower_row + 1, original_height - 1);
        
        // 원본 행인 경우 스킵
        if (std::abs(original_pos - lower_row) < 1e-6) {
            continue;
        }
        
        double weight = original_pos - lower_row;
        
        for (int col = 0; col < original_width; ++col) {
            float lower_val, upper_val;
            
            if (image.type() == CV_32F) {
                lower_val = image.at<float>(lower_row, col);
                upper_val = image.at<float>(upper_row, col);
            } else {
                lower_val = static_cast<float>(image.at<uint16_t>(lower_row, col));
                upper_val = static_cast<float>(image.at<uint16_t>(upper_row, col));
            }
            
            // 무효 값 처리 (0 또는 매우 작은 값)
            if (lower_val < 0.1f && upper_val < 0.1f) {
                if (interpolated.type() == CV_32F) {
                    interpolated.at<float>(new_row, col) = 0.0f;
                } else {
                    interpolated.at<uint16_t>(new_row, col) = 0;
                }
                continue;
            }
            
            // 엣지 보존 처리
            if (preserve_edges && std::abs(upper_val - lower_val) > edge_threshold) {
                // 엣지가 감지된 경우, 더 가까운 값 사용
                float interpolated_val = (weight < 0.5) ? lower_val : upper_val;
                
                if (interpolated.type() == CV_32F) {
                    interpolated.at<float>(new_row, col) = interpolated_val;
                } else {
                    interpolated.at<uint16_t>(new_row, col) = static_cast<uint16_t>(interpolated_val);
                }
            } else {
                // 일반 선형 보간
                float interpolated_val = lower_val * (1.0f - static_cast<float>(weight)) + 
                                       upper_val * static_cast<float>(weight);
                
                if (interpolated.type() == CV_32F) {
                    interpolated.at<float>(new_row, col) = interpolated_val;
                } else {
                    interpolated.at<uint16_t>(new_row, col) = static_cast<uint16_t>(interpolated_val);
                }
            }
        }
    }
    
    return interpolated;
}

InterpolationUtils::InterpolationResult InterpolationUtils::adaptiveInterpolation(
    const cv::Mat& range_image,
    const cv::Mat& intensity_image,
    const InterpolationConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    InterpolationResult result;
    result.original_height = range_image.rows;
    result.interpolation_factor = config.vertical_scale_factor;
    result.interpolated_height = static_cast<size_t>(range_image.rows * config.vertical_scale_factor);
    
    int new_height = static_cast<int>(range_image.rows * config.vertical_scale_factor);
    int width = range_image.cols;
    
    result.interpolated_range = cv::Mat::zeros(new_height, width, range_image.type());
    result.interpolated_intensity = cv::Mat::zeros(new_height, width, intensity_image.type());
    
    // 멀티스레딩 사용 여부 결정
    if (config.num_threads > 1) {
        return interpolateMultiThreaded(range_image, intensity_image, config);
    }
    
    // 각 픽셀에 대해 적응형 보간 수행
    for (int new_row = 0; new_row < new_height; ++new_row) {
        double original_pos = static_cast<double>(new_row) / config.vertical_scale_factor;
        int lower_row = static_cast<int>(std::floor(original_pos));
        
        for (int col = 0; col < width; ++col) {
            // 적응형 가중치 계산
            std::vector<double> weights = calculateAdaptiveWeights(
                range_image, intensity_image, lower_row, col, 2);
            
            // Range와 Intensity 값 보간
            float range_val = 0.0f, intensity_val = 0.0f;
            double total_weight = 0.0;
            
            // 이웃 픽셀들로부터 가중 평균 계산
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int src_row = std::max(0, std::min(range_image.rows - 1, lower_row + dy));
                    int src_col = std::max(0, std::min(width - 1, col + dx));
                    
                    size_t weight_idx = static_cast<size_t>((dy + 1) * 3 + (dx + 1));
                    double weight = (weight_idx < weights.size()) ? weights[weight_idx] : 1.0;
                    
                    if (range_image.type() == CV_32F) {
                        range_val += weight * range_image.at<float>(src_row, src_col);
                        intensity_val += weight * intensity_image.at<float>(src_row, src_col);
                    } else {
                        range_val += weight * range_image.at<uint16_t>(src_row, src_col);
                        intensity_val += weight * intensity_image.at<uint16_t>(src_row, src_col);
                    }
                    total_weight += weight;
                }
            }
            
            if (total_weight > 0) {
                range_val /= total_weight;
                intensity_val /= total_weight;
            }
            
            // 결과 저장
            if (result.interpolated_range.type() == CV_32F) {
                result.interpolated_range.at<float>(new_row, col) = range_val;
                result.interpolated_intensity.at<float>(new_row, col) = intensity_val;
            } else {
                result.interpolated_range.at<uint16_t>(new_row, col) = static_cast<uint16_t>(range_val);
                result.interpolated_intensity.at<uint16_t>(new_row, col) = static_cast<uint16_t>(intensity_val);
            }
        }
    }
    
    // 무효 픽셀 처리
    if (config.fill_invalid_pixels) {
        result.interpolated_range = fillInvalidPixels(result.interpolated_range);
        result.interpolated_intensity = fillInvalidPixels(result.interpolated_intensity);
    }
    
    // 유효성 마스크 생성
    result.validity_mask = cv::Mat::zeros(result.interpolated_range.size(), CV_8UC1);
    cv::threshold(result.interpolated_range, result.validity_mask, 0.1, 255, cv::THRESH_BINARY);
    
    // 처리 시간 계산
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

cv::Mat InterpolationUtils::fillInvalidPixels(const cv::Mat& image, int fill_method) {
    if (image.empty()) {
        return image.clone();
    }
    
    cv::Mat result = image.clone();
    cv::Mat mask;
    
    // 무효 픽셀 마스크 생성 (0값인 픽셀들)
    if (image.type() == CV_32F) {
        cv::threshold(image, mask, 0.1, 255, cv::THRESH_BINARY_INV);
    } else {
        cv::threshold(image, mask, 1, 255, cv::THRESH_BINARY_INV);
    }
    
    mask.convertTo(mask, CV_8UC1);
    
    // 마스크가 있는 영역이 있으면 inpaint 수행
    if (cv::countNonZero(mask) > 0) {
        cv::Mat temp;
        if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
            // inpaint는 8-bit 이미지만 지원하므로 변환
            double min_val, max_val;
            cv::minMaxLoc(image, &min_val, &max_val);
            image.convertTo(temp, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
            
            cv::Mat inpainted_8bit;
            cv::inpaint(temp, mask, inpainted_8bit, 3, fill_method);
            
            // 다시 원래 타입으로 변환
            inpainted_8bit.convertTo(result, image.type(), (max_val - min_val) / 255.0, min_val);
        } else {
            cv::inpaint(image, mask, result, 3, fill_method);
        }
    }
    
    return result;
}

std::map<std::string, double> InterpolationUtils::evaluateInterpolationQuality(
    const cv::Mat& original,
    const cv::Mat& interpolated) {
    
    std::map<std::string, double> metrics;
    
    if (original.empty() || interpolated.empty()) {
        return metrics;
    }
    
    // PSNR 계산
    cv::Mat diff;
    cv::absdiff(original, interpolated, diff);
    cv::Scalar mse_scalar = cv::mean(diff.mul(diff));
    double mse = mse_scalar[0];
    
    if (mse > 0) {
        double max_val = 255.0; // 8-bit 기준
        if (original.type() == CV_32F) {
            double min_val, max_val_actual;
            cv::minMaxLoc(original, &min_val, &max_val_actual);
            max_val = max_val_actual;
        } else if (original.type() == CV_16U) {
            max_val = 65535.0;
        }
        
        double psnr = 20.0 * std::log10(max_val / std::sqrt(mse));
        metrics["PSNR"] = psnr;
    } else {
        metrics["PSNR"] = std::numeric_limits<double>::infinity();
    }
    
    // MSE
    metrics["MSE"] = mse;
    
    // MAE (Mean Absolute Error)
    cv::Scalar mae_scalar = cv::mean(diff);
    metrics["MAE"] = mae_scalar[0];
    
    return metrics;
}

InterpolationUtils::InterpolationResult InterpolationUtils::interpolateMultiThreaded(
    const cv::Mat& range_image,
    const cv::Mat& intensity_image,
    const InterpolationConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    InterpolationResult result;
    result.original_height = range_image.rows;
    result.interpolation_factor = config.vertical_scale_factor;
    result.interpolated_height = static_cast<size_t>(range_image.rows * config.vertical_scale_factor);
    
    int new_height = static_cast<int>(range_image.rows * config.vertical_scale_factor);
    int width = range_image.cols;
    
    result.interpolated_range = cv::Mat::zeros(new_height, width, range_image.type());
    result.interpolated_intensity = cv::Mat::zeros(new_height, width, intensity_image.type());
    
    // 작업 분할
    std::vector<std::pair<int, int>> workload = divideWorkload(new_height, config.num_threads);
    
    // 스레드들을 이용한 병렬 처리
    std::vector<std::future<void>> futures;
    
    for (const auto& work : workload) {
        futures.emplace_back(std::async(std::launch::async, [&, work]() {
            for (int new_row = work.first; new_row < work.second; ++new_row) {
                interpolateRow(range_image, intensity_image, 
                             result.interpolated_range, result.interpolated_intensity,
                             new_row, config);
            }
        }));
    }
    
    // 모든 스레드 완료 대기
    for (auto& future : futures) {
        future.wait();
    }
    
    // 무효 픽셀 처리
    if (config.fill_invalid_pixels) {
        result.interpolated_range = fillInvalidPixels(result.interpolated_range);
        result.interpolated_intensity = fillInvalidPixels(result.interpolated_intensity);
    }
    
    // 유효성 마스크 생성
    result.validity_mask = cv::Mat::zeros(result.interpolated_range.size(), CV_8UC1);
    cv::threshold(result.interpolated_range, result.validity_mask, 0.1, 255, cv::THRESH_BINARY);
    
    // 처리 시간 계산
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

// Private helper functions implementation

double InterpolationUtils::calculateEdgePreservingWeight(
    const cv::Mat& range_image,
    int row,
    int col,
    double edge_threshold) {
    
    if (row <= 0 || row >= range_image.rows - 1 || col <= 0 || col >= range_image.cols - 1) {
        return 1.0;
    }
    
    float center_val = (range_image.type() == CV_32F) ? 
                       range_image.at<float>(row, col) : 
                       static_cast<float>(range_image.at<uint16_t>(row, col));
    
    // 이웃 픽셀들과의 차이 계산
    std::vector<float> differences;
    for (int dy = -1; dy <= 1; dy += 2) {
        for (int dx = -1; dx <= 1; dx += 2) {
            float neighbor_val = (range_image.type() == CV_32F) ? 
                                range_image.at<float>(row + dy, col + dx) : 
                                static_cast<float>(range_image.at<uint16_t>(row + dy, col + dx));
            differences.push_back(std::abs(center_val - neighbor_val));
        }
    }
    
    // 최대 차이를 기반으로 가중치 계산
    float max_diff = *std::max_element(differences.begin(), differences.end());
    
    if (max_diff > edge_threshold) {
        return 0.5; // 엣지 영역에서는 가중치 감소
    }
    
    return 1.0;
}

float InterpolationUtils::bilinearInterpolate(const cv::Mat& image, double x, double y) {
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = std::min(x1 + 1, image.cols - 1);
    int y2 = std::min(y1 + 1, image.rows - 1);
    
    double fx = x - x1;
    double fy = y - y1;
    
    float q11, q12, q21, q22;
    
    if (image.type() == CV_32F) {
        q11 = image.at<float>(y1, x1);
        q12 = image.at<float>(y2, x1);
        q21 = image.at<float>(y1, x2);
        q22 = image.at<float>(y2, x2);
    } else {
        q11 = static_cast<float>(image.at<uint16_t>(y1, x1));
        q12 = static_cast<float>(image.at<uint16_t>(y2, x1));
        q21 = static_cast<float>(image.at<uint16_t>(y1, x2));
        q22 = static_cast<float>(image.at<uint16_t>(y2, x2));
    }
    
    float result = q11 * (1 - fx) * (1 - fy) +
                   q21 * fx * (1 - fy) +
                   q12 * (1 - fx) * fy +
                   q22 * fx * fy;
    
    return result;
}

float InterpolationUtils::cubicInterpolate(const std::vector<float>& values, double t) {
    if (values.size() < 4) {
        return 0.0f;
    }
    
    float a = values[3] - values[2] - values[0] + values[1];
    float b = values[0] - values[1] - a;
    float c = values[2] - values[0];
    float d = values[1];
    
    return static_cast<float>(a * t * t * t + b * t * t + c * t + d);
}

std::vector<double> InterpolationUtils::calculateAdaptiveWeights(
    const cv::Mat& range_image,
    const cv::Mat& intensity_image,
    int row,
    int col,
    int neighbor_distance) {
    
    std::vector<double> weights(9, 1.0); // 3x3 이웃
    
    if (row < neighbor_distance || row >= range_image.rows - neighbor_distance ||
        col < neighbor_distance || col >= range_image.cols - neighbor_distance) {
        return weights;
    }
    
    float center_range = (range_image.type() == CV_32F) ? 
                        range_image.at<float>(row, col) : 
                        static_cast<float>(range_image.at<uint16_t>(row, col));
    
    float center_intensity = (intensity_image.type() == CV_32F) ? 
                            intensity_image.at<float>(row, col) : 
                            static_cast<float>(intensity_image.at<uint16_t>(row, col));
    
    int weight_idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (weight_idx >= static_cast<int>(weights.size())) break;
            
            float neighbor_range = (range_image.type() == CV_32F) ? 
                                  range_image.at<float>(row + dy, col + dx) : 
                                  static_cast<float>(range_image.at<uint16_t>(row + dy, col + dx));
            
            float neighbor_intensity = (intensity_image.type() == CV_32F) ? 
                                      intensity_image.at<float>(row + dy, col + dx) : 
                                      static_cast<float>(intensity_image.at<uint16_t>(row + dy, col + dx));
            
            // 거리와 intensity 차이를 기반으로 가중치 계산
            double range_diff = std::abs(center_range - neighbor_range);
            double intensity_diff = std::abs(center_intensity - neighbor_intensity);
            
            // 가중치는 유사도에 비례 (차이가 클수록 가중치 감소)
            double similarity = std::exp(-(range_diff * range_diff + intensity_diff * intensity_diff) / 1000.0);
            weights[weight_idx] = similarity;
            
            ++weight_idx;
        }
    }
    
    return weights;
}

std::vector<std::pair<int, int>> InterpolationUtils::divideWorkload(int image_height, int num_threads) {
    std::vector<std::pair<int, int>> workload;
    
    int rows_per_thread = image_height / num_threads;
    int remaining_rows = image_height % num_threads;
    
    int start_row = 0;
    for (int i = 0; i < num_threads; ++i) {
        int end_row = start_row + rows_per_thread;
        if (i < remaining_rows) {
            end_row += 1;
        }
        
        workload.emplace_back(start_row, std::min(end_row, image_height));
        start_row = end_row;
    }
    
    return workload;
}

void InterpolationUtils::interpolateRow(
    const cv::Mat& range_image,
    const cv::Mat& intensity_image,
    cv::Mat& interpolated_range,
    cv::Mat& interpolated_intensity,
    int target_row,
    const InterpolationConfig& config) {
    
    double original_pos = static_cast<double>(target_row) / config.vertical_scale_factor;
    int lower_row = static_cast<int>(std::floor(original_pos));
    int upper_row = std::min(lower_row + 1, range_image.rows - 1);
    double weight = original_pos - lower_row;
    
    for (int col = 0; col < range_image.cols; ++col) {
        float lower_range, upper_range, lower_intensity, upper_intensity;
        
        if (range_image.type() == CV_32F) {
            lower_range = range_image.at<float>(lower_row, col);
            upper_range = range_image.at<float>(upper_row, col);
            lower_intensity = intensity_image.at<float>(lower_row, col);
            upper_intensity = intensity_image.at<float>(upper_row, col);
        } else {
            lower_range = static_cast<float>(range_image.at<uint16_t>(lower_row, col));
            upper_range = static_cast<float>(range_image.at<uint16_t>(upper_row, col));
            lower_intensity = static_cast<float>(intensity_image.at<uint16_t>(lower_row, col));
            upper_intensity = static_cast<float>(intensity_image.at<uint16_t>(upper_row, col));
        }
        
        // 보간된 값 계산
        float interpolated_range_val = lower_range * (1.0f - static_cast<float>(weight)) + 
                                      upper_range * static_cast<float>(weight);
        float interpolated_intensity_val = lower_intensity * (1.0f - static_cast<float>(weight)) + 
                                          upper_intensity * static_cast<float>(weight);
        
        // 결과 저장
        if (interpolated_range.type() == CV_32F) {
            interpolated_range.at<float>(target_row, col) = interpolated_range_val;
            interpolated_intensity.at<float>(target_row, col) = interpolated_intensity_val;
        } else {
            interpolated_range.at<uint16_t>(target_row, col) = static_cast<uint16_t>(interpolated_range_val);
            interpolated_intensity.at<uint16_t>(target_row, col) = static_cast<uint16_t>(interpolated_intensity_val);
        }
    }
}

} // namespace filc 