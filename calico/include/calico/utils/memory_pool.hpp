#ifndef CALICO_UTILS_MEMORY_POOL_HPP
#define CALICO_UTILS_MEMORY_POOL_HPP

#include <vector>
#include <memory>
#include <mutex>

namespace calico {
namespace utils {

/**
 * @brief Simple memory pool for vector reuse to reduce allocations
 * @tparam T Element type
 */
template<typename T>
class VectorPool {
public:
    VectorPool(size_t initial_size = 10, size_t max_vector_capacity = 1000)
        : max_capacity_(max_vector_capacity) {
        // Pre-allocate some vectors
        for (size_t i = 0; i < initial_size; ++i) {
            std::vector<T> vec;
            vec.reserve(max_capacity_);
            pool_.push_back(std::move(vec));
        }
    }
    
    /**
     * @brief Get a vector from the pool or create a new one
     * @return Shared pointer to vector
     */
    std::shared_ptr<std::vector<T>> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!pool_.empty()) {
            auto vec = std::make_shared<std::vector<T>>(std::move(pool_.back()));
            pool_.pop_back();
            vec->clear();  // Clear contents but keep capacity
            return vec;
        }
        
        // Create new vector if pool is empty
        auto vec = std::make_shared<std::vector<T>>();
        vec->reserve(max_capacity_);
        return vec;
    }
    
    /**
     * @brief Return a vector to the pool
     * @param vec Vector to return
     */
    void release(std::shared_ptr<std::vector<T>> vec) {
        if (!vec) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Only keep vectors that aren't too large
        if (vec->capacity() <= max_capacity_ * 2) {
            vec->clear();
            pool_.push_back(std::move(*vec));
        }
        // Otherwise let it be destroyed
    }
    
    /**
     * @brief Clear the pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }
    
    /**
     * @brief Get current pool size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<std::vector<T>> pool_;
    size_t max_capacity_;
};

} // namespace utils
} // namespace calico

#endif // CALICO_UTILS_MEMORY_POOL_HPP