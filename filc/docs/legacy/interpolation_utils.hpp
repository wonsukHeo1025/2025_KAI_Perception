#ifndef filc_INTERPOLATION_UTILS_HPP
#define filc_INTERPOLATION_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>

namespace filc {

/**
 * @brief Ouster LiDAR Range와 Intensity Image 보간 처리를 위한 유틸리티 클래스
 */
class InterpolationUtils {
public:
    // 보간 방법 타입 정의
    enum class InterpolationType {
        LINEAR,         // 선형 보간
        CUBIC,          // 3차 보간
        BICUBIC,        // 이중 3차 보간
        LANCZOS,        // Lanczos 보간
        ADAPTIVE        // 적응형 보간 (지역적 특성에 따라 방법 선택)
    };

    // 보간 결과 구조체
    struct InterpolationResult {
        cv::Mat interpolated_range;     // 보간된 Range Image
        cv::Mat interpolated_intensity; // 보간된 Intensity Image
        cv::Mat validity_mask;          // 유효한 픽셀 마스크
        double processing_time_ms;      // 처리 시간
        size_t original_height;         // 원본 이미지 높이
        size_t interpolated_height;     // 보간된 이미지 높이
        double interpolation_factor;    // 보간 배율
    };

    // 보간 설정 파라미터
    struct InterpolationConfig {
        InterpolationType type = InterpolationType::LINEAR;
        double vertical_scale_factor = 2.0;    // 수직 방향 확대 배율 (채널 수 증가)
        double horizontal_scale_factor = 1.0;  // 수평 방향 확대 배율
        bool preserve_edges = true;             // 엣지 보존 여부
        double edge_threshold = 10.0;           // 엣지 검출 임계값 (range 차이, mm)
        bool fill_invalid_pixels = true;        // 무효 픽셀 채우기 여부
        double max_interpolation_distance = 5.0; // 최대 보간 거리 (미터)
        bool use_adaptive_weights = true;       // 적응형 가중치 사용
        int num_threads = 4;                    // 병렬 처리 스레드 수
    };

public:
    InterpolationUtils();
    ~InterpolationUtils();

    /**
     * @brief 기본 설정 생성
     * @return 기본 보간 설정
     */
    static InterpolationConfig createDefaultConfig();

    /**
     * @brief Ouster Range Image와 Intensity Image의 밀도를 향상시키는 보간 처리
     * @param range_image 원본 Range Image (CV_32F 또는 CV_16U, 거리값 mm 단위)
     * @param intensity_image 원본 Intensity Image (CV_32F 또는 CV_16U)
     * @param config 보간 설정 파라미터
     * @return 보간 결과
     */
    static InterpolationResult interpolateOusterImages(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        const InterpolationConfig& config
    );

    /**
     * @brief Range Image만을 사용한 보간 처리 (Intensity 이미지 없이)
     * @param range_image 원본 Range Image (CV_32F 또는 CV_16U, 거리값 mm 단위)
     * @param config 보간 설정 파라미터
     * @return 보간 결과 (intensity는 비어있음)
     */
    static InterpolationResult interpolateRangeImageOnly(
        const cv::Mat& range_image,
        const InterpolationConfig& config
    );

    /**
     * @brief 수직 방향(채널 간) 선형 보간
     * @param image 입력 이미지
     * @param scale_factor 확대 배율
     * @param preserve_edges 엣지 보존 여부
     * @param edge_threshold 엣지 검출 임계값
     * @return 보간된 이미지
     */
    static cv::Mat interpolateVerticalLinear(
        const cv::Mat& image,
        double scale_factor,
        bool preserve_edges = true,
        double edge_threshold = 10.0
    );

    /**
     * @brief 적응형 보간 - 지역적 특성에 따라 보간 방법 선택
     * @param range_image Range Image
     * @param intensity_image Intensity Image
     * @param config 설정 파라미터
     * @return 보간 결과
     */
    static InterpolationResult adaptiveInterpolation(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        const InterpolationConfig& config
    );

    /**
     * @brief Range Image만을 위한 적응형 보간
     * @param range_image Range Image
     * @param config 설정 파라미터
     * @return 보간된 Range Image
     */
    static cv::Mat adaptiveInterpolationRangeOnly(
        const cv::Mat& range_image,
        const InterpolationConfig& config
    );

    /**
     * @brief 유효하지 않은 픽셀(0값 등) 처리
     * @param image 입력 이미지
     * @param fill_method 채우기 방법 (INPAINT_TELEA, INPAINT_NS)
     * @return 처리된 이미지
     */
    static cv::Mat fillInvalidPixels(
        const cv::Mat& image,
        int fill_method = cv::INPAINT_TELEA
    );

    /**
     * @brief 보간된 데이터의 품질 평가
     * @param original 원본 이미지
     * @param interpolated 보간된 이미지
     * @return 품질 메트릭 (PSNR, SSIM 등)
     */
    static std::map<std::string, double> evaluateInterpolationQuality(
        const cv::Mat& original,
        const cv::Mat& interpolated
    );

    /**
     * @brief 멀티스레드 보간 처리
     * @param range_image Range Image
     * @param intensity_image Intensity Image
     * @param config 설정 파라미터
     * @return 보간 결과
     */
    static InterpolationResult interpolateMultiThreaded(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        const InterpolationConfig& config
    );

private:
    /**
     * @brief 엣지 보존 가중치 계산
     * @param range_image Range Image
     * @param row 행 인덱스
     * @param col 열 인덱스
     * @param edge_threshold 엣지 임계값
     * @return 가중치 값
     */
    static double calculateEdgePreservingWeight(
        const cv::Mat& range_image,
        int row,
        int col,
        double edge_threshold
    );

    /**
     * @brief 이중선형 보간 (Bilinear Interpolation)
     * @param image 입력 이미지
     * @param x 보간할 x 좌표
     * @param y 보간할 y 좌표
     * @return 보간된 값
     */
    static float bilinearInterpolate(
        const cv::Mat& image,
        double x,
        double y
    );

    /**
     * @brief 3차 보간 (Cubic Interpolation)
     * @param values 4개의 인접 값
     * @param t 보간 위치 (0-1)
     * @return 보간된 값
     */
    static float cubicInterpolate(
        const std::vector<float>& values,
        double t
    );

    /**
     * @brief 적응형 가중치 계산
     * @param range_image Range Image
     * @param intensity_image Intensity Image
     * @param row 행 인덱스
     * @param col 열 인덱스
     * @param neighbor_distance 이웃 거리
     * @return 가중치 벡터
     */
    static std::vector<double> calculateAdaptiveWeights(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        int row,
        int col,
        int neighbor_distance = 2
    );

    /**
     * @brief 처리할 이미지 영역을 여러 스레드로 분할
     * @param image_height 이미지 높이
     * @param num_threads 스레드 수
     * @return 각 스레드의 처리 영역 (start_row, end_row)
     */
    static std::vector<std::pair<int, int>> divideWorkload(
        int image_height,
        int num_threads
    );

    /**
     * @brief 단일 행의 보간 처리
     * @param range_image Range Image
     * @param intensity_image Intensity Image
     * @param interpolated_range 출력 Range Image
     * @param interpolated_intensity 출력 Intensity Image
     * @param target_row 대상 행 인덱스
     * @param config 설정 파라미터
     */
    static void interpolateRow(
        const cv::Mat& range_image,
        const cv::Mat& intensity_image,
        cv::Mat& interpolated_range,
        cv::Mat& interpolated_intensity,
        int target_row,
        const InterpolationConfig& config
    );
};

} // namespace filc

#endif // filc_INTERPOLATION_UTILS_HPP 