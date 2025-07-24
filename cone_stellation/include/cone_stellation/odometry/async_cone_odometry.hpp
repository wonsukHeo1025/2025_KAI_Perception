#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <rclcpp/rclcpp.hpp>

#include "cone_stellation/odometry/cone_odometry_base.hpp"
#include "cone_stellation/common/estimation_frame.hpp"

namespace cone_stellation {

/**
 * @brief Asynchronous wrapper for cone odometry estimation
 * 
 * Processes odometry estimation in a separate thread to avoid blocking
 * the main thread. Similar to GLIM's AsyncOdometryEstimation.
 */
class AsyncConeOdometry {
public:
  using Ptr = std::shared_ptr<AsyncConeOdometry>;
  
  /**
   * @brief Result of odometry estimation
   */
  struct OdometryResult {
    EstimationFrame::ConstPtr frame;          // Current frame
    Eigen::Isometry3d T_prev_curr;           // Transformation from previous
    Eigen::Isometry3d T_world_sensor;        // World to sensor transform
    int num_inliers;                         // Number of inlier matches
    double timestamp;                        // Frame timestamp
  };
  
  AsyncConeOdometry(ConeOdometryBase::Ptr odometry) 
    : odometry_(odometry), 
      running_(false),
      kill_switch_(false) {}
  
  ~AsyncConeOdometry() {
    stop();
  }
  
  /**
   * @brief Start the processing thread
   */
  void start() {
    if (running_) return;
    
    running_ = true;
    kill_switch_ = false;
    thread_ = std::thread(&AsyncConeOdometry::process_loop, this);
    
    RCLCPP_INFO(rclcpp::get_logger("async_cone_odometry"), 
                "Started async cone odometry thread");
  }
  
  /**
   * @brief Stop the processing thread
   */
  void stop() {
    if (!running_) return;
    
    kill_switch_ = true;
    frame_cv_.notify_all();
    
    if (thread_.joinable()) {
      thread_.join();
    }
    
    running_ = false;
    RCLCPP_INFO(rclcpp::get_logger("async_cone_odometry"), 
                "Stopped async cone odometry thread");
  }
  
  /**
   * @brief Insert a new frame for odometry estimation
   * @param frame Frame with cone observations
   * @return false if queue is full
   */
  bool insert_frame(const EstimationFrame::ConstPtr& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (frame_queue_.size() >= max_queue_size_) {
      RCLCPP_WARN_THROTTLE(rclcpp::get_logger("async_cone_odometry"), 
                          *rclcpp::Clock::make_shared(), 1000,
                          "Odometry frame queue full, dropping frame");
      return false;
    }
    
    frame_queue_.push_back(frame);
    frame_cv_.notify_one();
    return true;
  }
  
  /**
   * @brief Get the latest odometry result (non-blocking)
   * @return Latest result or nullptr if none available
   */
  std::shared_ptr<OdometryResult> get_result() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (!latest_result_) {
      return nullptr;
    }
    
    auto result = latest_result_;
    latest_result_ = nullptr;
    return result;
  }
  
  /**
   * @brief Get all available results (non-blocking)
   * @return Vector of results
   */
  std::vector<std::shared_ptr<OdometryResult>> get_all_results() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    std::vector<std::shared_ptr<OdometryResult>> results;
    results.swap(result_queue_);
    return results;
  }
  
  /**
   * @brief Check if processing thread is running
   */
  bool is_running() const { return running_; }
  
  /**
   * @brief Get the odometry estimator
   */
  ConeOdometryBase::Ptr get_odometry() const { return odometry_; }

private:
  void process_loop() {
    EstimationFrame::ConstPtr prev_frame = nullptr;
    Eigen::Isometry3d T_world_prev = Eigen::Isometry3d::Identity();
    
    while (!kill_switch_) {
      // Wait for new frame
      EstimationFrame::ConstPtr curr_frame = nullptr;
      {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait(lock, [this] { 
          return !frame_queue_.empty() || kill_switch_; 
        });
        
        if (kill_switch_) break;
        
        curr_frame = frame_queue_.front();
        frame_queue_.pop_front();
      }
      
      // First frame - no odometry to compute
      if (!prev_frame) {
        prev_frame = curr_frame;
        T_world_prev = curr_frame->T_world_sensor;
        
        // Create initial result
        auto result = std::make_shared<OdometryResult>();
        result->frame = curr_frame;
        result->T_prev_curr = Eigen::Isometry3d::Identity();
        result->T_world_sensor = T_world_prev;
        result->num_inliers = 0;
        result->timestamp = curr_frame->timestamp;
        
        publish_result(result);
        continue;
      }
      
      // Estimate odometry
      auto start_time = std::chrono::high_resolution_clock::now();
      
      Eigen::Isometry3d T_prev_curr = odometry_->estimate(prev_frame, curr_frame);
      int num_inliers = odometry_->num_inliers();
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time).count();
      
      // Update world transform
      Eigen::Isometry3d T_world_curr = T_world_prev * T_prev_curr;
      
      // Create result
      auto result = std::make_shared<OdometryResult>();
      result->frame = curr_frame;
      result->T_prev_curr = T_prev_curr;
      result->T_world_sensor = T_world_curr;
      result->num_inliers = num_inliers;
      result->timestamp = curr_frame->timestamp;
      
      publish_result(result);
      
      RCLCPP_DEBUG(rclcpp::get_logger("async_cone_odometry"),
                   "Odometry computed in %ld ms, inliers: %d", 
                   duration, num_inliers);
      
      // Update for next iteration
      prev_frame = curr_frame;
      T_world_prev = T_world_curr;
    }
  }
  
  void publish_result(const std::shared_ptr<OdometryResult>& result) {
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    latest_result_ = result;
    result_queue_.push_back(result);
    
    // Keep queue size limited
    while (result_queue_.size() > max_result_queue_size_) {
      result_queue_.erase(result_queue_.begin());
    }
  }
  
  // Odometry estimator
  ConeOdometryBase::Ptr odometry_;
  
  // Threading
  std::thread thread_;
  std::atomic<bool> running_;
  std::atomic<bool> kill_switch_;
  
  // Frame queue
  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  std::deque<EstimationFrame::ConstPtr> frame_queue_;
  const size_t max_queue_size_ = 10;
  
  // Result queue
  std::mutex result_mutex_;
  std::shared_ptr<OdometryResult> latest_result_;
  std::vector<std::shared_ptr<OdometryResult>> result_queue_;
  const size_t max_result_queue_size_ = 100;
};

} // namespace cone_stellation