#include "prism/interpolation/interpolation_engine.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <thread>
#include <cmath>


namespace prism {
namespace interpolation {

InterpolationEngine::InterpolationEngine(const InterpolationConfig& config)
    : config_(config)
    , beam_manager_(BeamAltitudeConfig{})
{
    if (!validateConfig(config_)) {
        throw std::invalid_argument("Invalid interpolation configuration");
    }
    
    // Create Catmull-Rom interpolator with compatible configuration
    CatmullRomConfig catmull_config;
    catmull_config.tension = config_.spline_tension;
    catmull_config.discontinuity_threshold = config_.discontinuity_threshold;
    interpolator_ = std::make_unique<CatmullRomInterpolator>(catmull_config);
    
    // Configure beam altitude manager
    BeamAltitudeConfig beam_config;
    beam_config.input_beams = config_.input_channels;
    beam_config.output_beams = config_.getOutputBeams();
    beam_manager_.configure(beam_config);
    
    // Initialize beam altitude manager with OS1-32 specifications
    if (!beam_manager_.initializeOS132Beams()) {
        throw std::runtime_error("Failed to initialize OS1-32 beam specifications");
    }
    
    if (!beam_manager_.generateInterpolatedBeams()) {
        throw std::runtime_error("Failed to generate interpolated beam configuration");
    }
    
    initialized_ = true;
    resetMetrics();
}

void InterpolationEngine::configure(const InterpolationConfig& config) {
    if (!validateConfig(config)) {
        throw std::invalid_argument("Invalid interpolation configuration");
    }
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    config_ = config;
    
    // Reconfigure Catmull-Rom interpolator
    if (interpolator_) {
        CatmullRomConfig catmull_config;
        catmull_config.tension = config_.spline_tension;
        catmull_config.discontinuity_threshold = config_.discontinuity_threshold;
        interpolator_->configure(catmull_config);
    }
    
    // Reinitialize beam altitude manager with new configuration
    // This is necessary because the beam manager needs to be properly initialized
    // with OS1-32 beam data before it can generate interpolated beams
    BeamAltitudeConfig beam_config;
    beam_config.input_beams = config_.input_channels;
    beam_config.output_beams = config_.getOutputBeams();
    
    // Create a new beam manager to ensure proper initialization
    beam_manager_ = BeamAltitudeManager(beam_config);
    
    // Properly initialize OS1-32 beams and generate interpolated beams
    try {
        if (!beam_manager_.initializeOS132Beams()) {
            throw std::runtime_error("initializeOS132Beams failed");
        }
        if (!beam_manager_.generateInterpolatedBeams()) {
            throw std::runtime_error("generateInterpolatedBeams failed");
        }
    } catch (const std::exception& e) {
        throw; // propagate to caller; configuration invalid
    }
}

InterpolationResult InterpolationEngine::interpolate(const core::PointCloudSoA& input) {
    return interpolate(input, config_.getOutputBeams());
}

InterpolationResult InterpolationEngine::interpolate(const core::PointCloudSoA& input, 
                                                   size_t output_channels) {
    InterpolationResult result;
    
    if (!initialized_) {
        result.error_message = "InterpolationEngine not properly initialized";
        return result;
    }
    
    // Validate input
    if (!validateInput(input)) {
        result.error_message = "Invalid input point cloud";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Acquire output point cloud from memory pool
        if (config_.memory_pool) {
            size_t estimated_size = estimateOutputSize(input.size(), output_channels);
            result.interpolated_cloud = config_.memory_pool->acquire(estimated_size);
        } else {
            // Create regular unique_ptr and wrap it in PooledPtr
            auto raw_cloud = std::make_unique<core::PointCloudSoA>(
                estimateOutputSize(input.size(), output_channels));
            result.interpolated_cloud = core::PooledPtr<core::PointCloudSoA>(
                raw_cloud.release(), core::PoolDeleter<core::PointCloudSoA>{nullptr});
        }
        
        if (!result.interpolated_cloud) {
            result.error_message = "Failed to acquire memory for output point cloud";
            return result;
        }
        
        // Clear output cloud
        result.interpolated_cloud->clear();
        
        // Separate input points by original beams (rings)
        auto beam_start = std::chrono::high_resolution_clock::now();
        auto beam_groups = separateBeams(input);
        auto beam_end = std::chrono::high_resolution_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            auto beam_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(beam_end - beam_start);
            double beam_ms = std::chrono::duration<double, std::milli>(beam_duration).count();
            metrics_.beam_processing_time_ms.set(metrics_.beam_processing_time_ms.get() + beam_ms);
        }
        
        // Process vertical beams only (no in-beam horizontal densification)
        result.beams_processed = output_channels;
        result.points_per_beam.reserve(output_channels);
        result.beam_altitudes.reserve(output_channels);
        
        for (size_t target_beam = 0; target_beam < output_channels; ++target_beam) {
            const auto& ib = beam_manager_.getInterpolatedBeam(target_beam);
            const auto& low = (ib.source_beam_low < beam_groups.size()) ? beam_groups[ib.source_beam_low] : std::vector<size_t>{};
            const auto& high = (ib.source_beam_high < beam_groups.size()) ? beam_groups[ib.source_beam_high] : std::vector<size_t>{};
            size_t added = 0;
            if (ib.is_original) {
                // Copy original beam points, update ring to target_beam
                for (size_t idx : low) {
                    result.interpolated_cloud->addPoint(
                        input.x[idx], input.y[idx], input.z[idx], input.intensity[idx],
                        input.hasColor() ? input.r[idx] : 0,
                        input.hasColor() ? input.g[idx] : 0,
                        input.hasColor() ? input.b[idx] : 0,
                        static_cast<uint16_t>(target_beam)
                    );
                    ++added;
                }
            } else {
                added = processBeam(low, high, ib.interpolation_weight, input,
                                    static_cast<uint16_t>(target_beam),
                                    *result.interpolated_cloud);
            }
            result.points_per_beam.push_back(added);
            result.beam_altitudes.push_back(ib.altitude_angle);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        // Update metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            double total_ms = std::chrono::duration<double, std::milli>(total_time).count();
            metrics_.interpolation_time_ms.set(total_ms);
            metrics_.input_points.set(input.size());
            metrics_.output_points.set(result.interpolated_cloud->size());
            
            // Record throughput operation
            metrics_.throughput.recordOperation(total_ms, result.interpolated_cloud->size());
            
            // Calculate and set interpolation ratio
            if (input.size() > 0) {
                double ratio = static_cast<double>(result.interpolated_cloud->size()) / input.size();
                metrics_.interpolation_ratio.set(ratio);
            }
        }
        
        result.metrics = metrics_;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.success = false;
    }
    
    return result;
}

size_t InterpolationEngine::processBeam(const std::vector<size_t>& beam_indices_low,
                                      const std::vector<size_t>& beam_indices_high,
                                      float interpolation_weight,
                                      const core::PointCloudSoA& input,
                                      uint16_t target_ring,
                                      core::PointCloudSoA& output) {
    if (!interpolator_) {
        return 0;
    }

    // If azimuth binning is enabled, avoid per-beam sorting by using uniform bins
    if (config_.azimuth_bins > 0) {
        const int bins = std::max(1, config_.azimuth_bins);
        const float two_pi = static_cast<float>(2.0 * M_PI);
        const float inv_two_pi = 1.0f / two_pi;
        auto wrap = [two_pi](float a){
            while (a >= static_cast<float>(M_PI)) a -= two_pi;
            while (a < -static_cast<float>(M_PI)) a += two_pi;
            return a;
        };

        std::vector<size_t> low_pick(bins, static_cast<size_t>(-1));
        std::vector<size_t> high_pick(bins, static_cast<size_t>(-1));
        std::vector<float> low_diff(bins, std::numeric_limits<float>::infinity());
        std::vector<float> high_diff(bins, std::numeric_limits<float>::infinity());

        auto choose_bin = [&](size_t idx, std::vector<size_t>& pick, std::vector<float>& diff_arr){
            float phi = std::atan2(input.y[idx], input.x[idx]);
            float nf = (phi + static_cast<float>(M_PI)) * (static_cast<float>(bins) * inv_two_pi);
            int b = static_cast<int>(std::floor(nf));
            if (b < 0) {
                b = 0;
            }
            if (b >= bins) {
                b = bins - 1;
            }
            float center = -static_cast<float>(M_PI) + (static_cast<float>(b) + 0.5f) * (two_pi / static_cast<float>(bins));
            float d = std::abs(wrap(phi - center));
            if (d < diff_arr[b]) { diff_arr[b] = d; pick[b] = idx; }
        };

        for (size_t idx : beam_indices_low)  choose_bin(idx, low_pick,  low_diff);
        for (size_t idx : beam_indices_high) choose_bin(idx, high_pick, high_diff);

        size_t points_added = 0;
        float w = std::clamp(interpolation_weight, 0.0f, 1.0f);
        for (int b = 0; b < bins; ++b) {
            size_t idxL = low_pick[b];
            size_t idxH = high_pick[b];
            if (idxL == static_cast<size_t>(-1) || idxH == static_cast<size_t>(-1)) continue;
            float x = (1.0f - w) * input.x[idxL] + w * input.x[idxH];
            float y = (1.0f - w) * input.y[idxL] + w * input.y[idxH];
            float z = (1.0f - w) * input.z[idxL] + w * input.z[idxH];
            float intensity = (1.0f - w) * input.intensity[idxL] + w * input.intensity[idxH];
            output.addPoint(x, y, z, intensity,
                            input.hasColor() ? static_cast<uint8_t>((input.r[idxL] + input.r[idxH]) / 2) : 0,
                            input.hasColor() ? static_cast<uint8_t>((input.g[idxL] + input.g[idxH]) / 2) : 0,
                            input.hasColor() ? static_cast<uint8_t>((input.b[idxL] + input.b[idxH]) / 2) : 0,
                            target_ring);
            ++points_added;
        }
        return points_added;
    }

    // Default path: Match nearest by azimuth after sorting
    std::vector<std::pair<float, size_t>> low_by_azimuth;
    std::vector<std::pair<float, size_t>> high_by_azimuth;
    low_by_azimuth.reserve(beam_indices_low.size());
    high_by_azimuth.reserve(beam_indices_high.size());
    auto azimuth_of = [&input](size_t idx) {
        return std::atan2(input.y[idx], input.x[idx]);
    };
    for (size_t idx : beam_indices_low) {
        low_by_azimuth.emplace_back(azimuth_of(idx), idx);
    }
    for (size_t idx : beam_indices_high) {
        high_by_azimuth.emplace_back(azimuth_of(idx), idx);
    }
    std::sort(low_by_azimuth.begin(), low_by_azimuth.end(), [](auto& a, auto& b){return a.first < b.first;});
    std::sort(high_by_azimuth.begin(), high_by_azimuth.end(), [](auto& a, auto& b){return a.first < b.first;});
    
    const size_t count = std::min(low_by_azimuth.size(), high_by_azimuth.size());
    size_t points_added = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t idxL = low_by_azimuth[i].second;
        size_t idxH = high_by_azimuth[i].second;
        float w = std::clamp(interpolation_weight, 0.0f, 1.0f);
        float x = (1.0f - w) * input.x[idxL] + w * input.x[idxH];
        float y = (1.0f - w) * input.y[idxL] + w * input.y[idxH];
        float z = (1.0f - w) * input.z[idxL] + w * input.z[idxH];
        float intensity = (1.0f - w) * input.intensity[idxL] + w * input.intensity[idxH];
        output.addPoint(x, y, z, intensity,
                        input.hasColor() ? static_cast<uint8_t>((input.r[idxL] + input.r[idxH]) / 2) : 0,
                        input.hasColor() ? static_cast<uint8_t>((input.g[idxL] + input.g[idxH]) / 2) : 0,
                        input.hasColor() ? static_cast<uint8_t>((input.b[idxL] + input.b[idxH]) / 2) : 0,
                        target_ring);
        ++points_added;
    }
    return points_added;
}

std::vector<size_t> InterpolationEngine::detectDiscontinuities(
    const std::vector<size_t>& beam_indices,
    const core::PointCloudSoA& input) const {
    
    std::vector<size_t> discontinuities;
    
    if (!config_.enable_discontinuity_detection || beam_indices.size() < 2) {
        return discontinuities;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 1; i < beam_indices.size(); ++i) {
        size_t idx1 = beam_indices[i - 1];
        size_t idx2 = beam_indices[i];
        
        // Squared-distance comparison (avoids sqrt)
        float dx = input.x[idx2] - input.x[idx1];
        float dy = input.y[idx2] - input.y[idx1];
        float dz = input.z[idx2] - input.z[idx1];
        float dist2 = dx*dx + dy*dy + dz*dz;
        float thr2 = config_.discontinuity_threshold * config_.discontinuity_threshold;
        
        if (dist2 > thr2) {
            discontinuities.push_back(i);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    // Update metrics in a non-const context
    const_cast<InterpolationEngine*>(this)->updateMetrics(duration, discontinuities.size());
    
    return discontinuities;
}

std::vector<std::vector<size_t>> InterpolationEngine::separateBeams(const core::PointCloudSoA& input) const {
    std::vector<std::vector<size_t>> beam_groups(config_.input_channels);
    
    for (size_t i = 0; i < input.size(); ++i) {
        // If ring data is available, use it. Otherwise, calculate from geometry
        uint16_t ring = 0;
        if (input.hasRing() && i < input.ring.size()) {
            ring = input.ring[i];
        } else {
            // Calculate ring from elevation angle
            float elevation = std::atan2(input.z[i], std::sqrt(input.x[i] * input.x[i] + input.y[i] * input.y[i]));
            float normalized_elevation = (elevation + M_PI/6) / (M_PI/3);  // Assuming ±30 degree FOV
            ring = static_cast<uint16_t>(normalized_elevation * (config_.input_channels - 1));
            ring = std::min(ring, static_cast<uint16_t>(config_.input_channels - 1));
        }
        
        if (ring < config_.input_channels) {
            beam_groups[ring].push_back(i);
        }
    }
    
    // Sort by azimuth only when binning is disabled
    if (config_.azimuth_bins == 0) {
        for (auto& beam : beam_groups) {
            if (beam.size() > 1) {
                std::sort(beam.begin(), beam.end(), [&input](size_t a, size_t b) {
                    float azimuth_a = std::atan2(input.y[a], input.x[a]);
                    float azimuth_b = std::atan2(input.y[b], input.x[b]);
                    return azimuth_a < azimuth_b;
                });
            }
        }
    }
    
    return beam_groups;
}


bool InterpolationEngine::validateConfig(const InterpolationConfig& config) const {
    if (config.input_channels == 0 || config.scale_factor < 1.0) {
        return false;
    }
    
    if (config.spline_tension < 0.0f || config.spline_tension > 1.0f) {
        return false;
    }
    
    if (config.discontinuity_threshold <= 0.0f) {
        return false;
    }
    
    // Memory pool is optional - nullptr is valid
    // The engine will allocate memory directly if no pool is provided
    
    return true;
}

bool InterpolationEngine::validateInput(const core::PointCloudSoA& input) const {
    if (input.empty()) {
        return false;
    }
    
    if (!input.validate()) {
        return false;
    }
    
    // Check if ring data is available for beam separation
    // NOTE: Ring data is generated in convertToSoA if not present in original data
    // So we'll proceed even without ring data in the input
    // if (!input.hasRing()) {
    //     // In production, we might generate ring data from geometry
    //     return false;
    // }
    
    return true;
}

size_t InterpolationEngine::estimateOutputSize(size_t input_size, size_t output_channels) const {
    if (input_size == 0 || config_.input_channels == 0) {
        return 0;
    }
    
    // Simple estimation: scale by channel ratio
    double channel_ratio = static_cast<double>(output_channels) / config_.input_channels;
    return static_cast<size_t>(input_size * channel_ratio * 1.2); // 20% buffer
}

void InterpolationEngine::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.reset();
}


void InterpolationEngine::updateMetrics(std::chrono::nanoseconds duration, size_t discontinuities) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    double duration_ms = std::chrono::duration<double, std::milli>(duration).count();
    metrics_.discontinuity_detection_time_ms.set(
        metrics_.discontinuity_detection_time_ms.get() + duration_ms);
    metrics_.discontinuities_detected.increment(discontinuities);
}

const InterpolationMetrics& InterpolationEngine::getMetrics() const noexcept {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

// Factory function
std::unique_ptr<InterpolationEngine> createInterpolationEngine(const InterpolationConfig& config) {
    return std::make_unique<InterpolationEngine>(config);
}

// Utility functions
namespace utils {

size_t calculateOptimalBatchSize(size_t input_size, size_t num_threads) {
    if (input_size == 0 || num_threads == 0) {
        return 1000; // Default batch size
    }
    
    // Aim for 4-8 batches per thread
    size_t target_batches = num_threads * 6;
    size_t batch_size = input_size / target_batches;
    
    // Clamp to reasonable range
    return std::max(size_t(100), std::min(size_t(10000), batch_size));
}

std::string metricsToString(const InterpolationMetrics& metrics) {
    std::ostringstream oss;
    oss << "InterpolationMetrics {\n";
    oss << "  Interpolation time: " << metrics.interpolation_time_ms.get() << " ms\n";
    oss << "  Beam processing time: " << metrics.beam_processing_time_ms.get() << " ms\n";
    oss << "  Discontinuity detection time: " << metrics.discontinuity_detection_time_ms.get() << " ms\n";
    oss << "  Input points: " << metrics.input_points.get() << "\n";
    oss << "  Output points: " << metrics.output_points.get() << "\n";
    oss << "  Discontinuities detected: " << metrics.discontinuities_detected.get() << "\n";
    oss << "  Interpolation ratio: " << metrics.interpolation_ratio.get() << "\n";
    oss << "  Throughput: " << metrics.throughput.getThroughput() << " points/sec\n";
    oss << "}";
    return oss.str();
}

bool validateOS132Compatibility(const InterpolationConfig& config) {
    return config.input_channels == 32 && 
           config.scale_factor >= 1.0 && 
           config.scale_factor <= 4.0;
}

} // namespace utils

} // namespace interpolation
} // namespace prism
