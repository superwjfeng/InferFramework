#ifndef INFER_INCLUDE_BASE_ALLOC_H
#define INFER_INCLUDE_BASE_ALLOC_H
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "base.h"

namespace base {

/**
 * @enum MemcpyKind
 * @brief Enumeration of memory copy operation types
 *
 * Specifies the source and destination device types for memory copy operations.
 * Used to determine which memory copy API to invoke (e.g., memcpy, cudaMemcpy).
 */
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,   ///< Copy from CPU memory to CPU memory (standard memcpy)
  kMemcpyCPU2CUDA = 1,  ///< Copy from CPU (host) memory to CUDA (device) memory
  kMemcpyCUDA2CPU = 2,  ///< Copy from CUDA (device) memory to CPU (host) memory
  kMemcpyCUDA2CUDA = 3, ///< Copy from CUDA memory to CUDA memory (device-to-device)
};

/**
 * @class DeviceAllocator
 * @brief Abstract base class for device-specific memory allocators
 *
 * This class provides a unified interface for memory allocation and management
 * across different device types (CPU, CUDA). It abstracts device-specific memory
 * operations including allocation, deallocation, memory copying, and initialization.
 *
 * The allocator supports:
 * - Device-specific memory allocation and deallocation
 * - Cross-device memory copying (CPU<->CUDA, CUDA<->CUDA)
 * - Memory initialization (zero-filling)
 * - Asynchronous operations with CUDA streams
 *
 * @note This is an abstract class. Use concrete implementations like
 *       CPUDeviceAllocator or CUDADeviceAllocator.
 * @see CPUDeviceAllocator For CPU memory allocation
 * @see CUDADeviceAllocator For CUDA GPU memory allocation
 */
class DeviceAllocator {
 public:
  /**
   * @brief Constructs a DeviceAllocator for the specified device type
   *
   * @param deviceType Type of device (CPU, CUDA, etc.)
   */
  explicit DeviceAllocator(DeviceType deviceType) : deviceType_(deviceType) {}

  /**
   * @brief Gets the device type of this allocator
   *
   * @return Device type (CPU, CUDA, etc.)
   */
  virtual DeviceType deviceType() const { return deviceType_; }

 public:
  /**
   * @brief Releases memory allocated by this allocator
   *
   * Pure virtual method that must be implemented by derived classes to
   * free memory in a device-appropriate manner.
   *
   * @param ptr Pointer to memory to release (must have been allocated by this allocator)
   */
  virtual void release(void* ptr) const = 0;

  /**
   * @brief Allocates memory on the device
   *
   * Pure virtual method that must be implemented by derived classes to
   * allocate memory in a device-appropriate manner.
   *
   * @param byteSize Number of bytes to allocate
   * @return Pointer to allocated memory, or nullptr on failure
   */
  virtual void* allocate(size_t byteSize) const = 0;

  /**
   * @brief Copies memory between devices
   *
   * Performs memory copy operations that may span different device types.
   * Supports synchronous and asynchronous copying with optional stream
   * synchronization.
   *
   * @param srcPtr Source pointer (must be valid for source device type)
   * @param destPtr Destination pointer (must be valid for destination device type)
   * @param byteSize Number of bytes to copy
   * @param memcpyKind Type of memory copy operation (CPU<->CPU, CPU<->CUDA, etc.)
   * @param stream Optional CUDA stream for asynchronous operations (CUDA only)
   * @param needSync If true, synchronize after copy operation completes
   */
  virtual void memcpy(const void* srcPtr, void* destPtr, size_t byteSize,
                      MemcpyKind memcpyKind = MemcpyKind::kMemcpyCPU2CPU,
                      void* stream = nullptr, bool needSync = false) const;

  /**
   * @brief Initializes memory region to zero
   *
   * Sets all bytes in the specified memory region to zero. Supports
   * asynchronous operations with CUDA streams.
   *
   * @param ptr Pointer to memory region to zero-initialize
   * @param byteSize Number of bytes to set to zero
   * @param stream Optional CUDA stream for asynchronous operations (CUDA only)
   * @param needSync If true, synchronize after memset completes
   */
  virtual void memsetZero(void* ptr, size_t byteSize, void* stream,
                          bool needSync = false);

 private:
  DeviceType deviceType_ = DeviceType::kDeviceUnknown;  ///< Device type for this allocator
};

/**
 * @class CPUDeviceAllocator
 * @brief CPU memory allocator implementation
 *
 * Concrete implementation of DeviceAllocator for CPU (host) memory.
 * Uses standard system memory allocation functions (malloc/free) to
 * manage CPU-accessible memory.
 *
 * Typical use cases:
 * - Allocating buffers for CPU-based inference
 * - Allocating temporary buffers for data preprocessing
 * - Allocating host memory for CPU-GPU data transfers
 *
 * @note Use CPUDeviceAllocatorFactory::getInstance() to obtain a
 *       shared singleton instance instead of creating directly.
 * @see DeviceAllocator Base class interface
 * @see CPUDeviceAllocatorFactory Factory for obtaining shared instances
 */
class CPUDeviceAllocator : public DeviceAllocator {
 public:
  /**
   * @brief Constructs a CPUDeviceAllocator instance
   */
  explicit CPUDeviceAllocator();

  /**
   * @brief Releases CPU memory
   *
   * Frees memory that was previously allocated by this allocator using free().
   *
   * @param ptr Pointer to CPU memory to release
   */
  void release(void* ptr) const override;

  /**
   * @brief Allocates CPU memory
   *
   * Allocates the requested number of bytes from CPU (host) memory using malloc().
   *
   * @param byteSize Number of bytes to allocate
   * @return Pointer to allocated CPU memory, or nullptr on allocation failure
   */
  void* allocate(size_t byteSize) const override;
};

/**
 * @struct CudaMemoryBuffer
 * @brief Descriptor for a CUDA memory buffer in the memory pool
 *
 * This structure tracks metadata for CUDA memory buffers managed by the
 * CUDADeviceAllocator. It maintains information about buffer location,
 * size, and availability status for efficient memory reuse and pooling.
 *
 * The allocator uses these descriptors to implement a memory pool strategy
 * that reduces the overhead of frequent CUDA memory allocations and deallocations.
 */
struct CudaMemoryBuffer {
  void* data;      ///< Pointer to CUDA device memory
  size_t byteSize; ///< Size of the buffer in bytes
  bool busy;       ///< Flag indicating if buffer is currently in use

  /**
   * @brief Default constructor
   */
  CudaMemoryBuffer() = default;

  /**
   * @brief Constructs a CudaMemoryBuffer with specified parameters
   *
   * @param data Pointer to CUDA device memory
   * @param byteSize Size of the buffer in bytes
   * @param busy Whether the buffer is currently allocated/busy
   */
  CudaMemoryBuffer(void* data, size_t byteSize, bool busy)
      : data(data), byteSize(byteSize), busy(busy) {}
};

/**
 * @class CUDADeviceAllocator
 * @brief CUDA GPU memory allocator with memory pooling
 *
 * Concrete implementation of DeviceAllocator for CUDA GPU memory.
 * Features a sophisticated memory pooling system to reduce allocation overhead:
 * - Maintains separate pools for different buffer sizes
 * - Reuses previously allocated buffers when possible
 * - Tracks buffer availability (busy/free status)
 * - Supports multiple CUDA devices
 *
 * The allocator uses cudaMalloc/cudaFree for actual GPU memory operations
 * but maintains pools to minimize the frequency of these expensive calls.
 *
 * Memory pool organization:
 * - cudaBuffersMap_: Pool for standard-sized buffers
 * - bigBuffersMap_: Pool for large buffers (different management strategy)
 * - noBusyCnt_: Tracks number of free buffers per device
 *
 * @note Use CUDADeviceAllocatorFactory::getInstance() to obtain a
 *       shared singleton instance instead of creating directly.
 * @see DeviceAllocator Base class interface
 * @see CudaMemoryBuffer Buffer descriptor used in memory pools
 * @see CUDADeviceAllocatorFactory Factory for obtaining shared instances
 */
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  /**
   * @brief Constructs a CUDADeviceAllocator instance
   *
   * Initializes the CUDA memory allocator and its internal memory pools.
   */
  explicit CUDADeviceAllocator();

  /**
   * @brief Releases CUDA GPU memory or returns it to the pool
   *
   * Depending on allocation strategy and buffer size, may either:
   * - Return the buffer to the memory pool for reuse
   * - Actually free the GPU memory using cudaFree
   *
   * @param ptr Pointer to CUDA device memory to release
   */
  void release(void* ptr) const override;

  /**
   * @brief Allocates CUDA GPU memory or reuses from pool
   *
   * Attempts to find a suitable buffer in the memory pool first.
   * If no suitable buffer exists, allocates new GPU memory using cudaMalloc.
   *
   * @param byteSize Number of bytes to allocate
   * @return Pointer to CUDA device memory, or nullptr on allocation failure
   */
  void* allocate(size_t byteSize) const override;

 private:
  mutable std::map<int, size_t> noBusyCnt_;  ///< Count of free buffers per device ID
  mutable std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap_;  ///< Pool of large buffers per device
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap_; ///< Pool of standard buffers per device
};

/**
 * @class CPUDeviceAllocatorFactory
 * @brief Singleton factory for CPUDeviceAllocator instances
 *
 * Implements the Singleton pattern to ensure a single shared instance of
 * CPUDeviceAllocator across the application. This design:
 * - Reduces memory overhead by sharing a single allocator
 * - Ensures consistent memory management behavior
 * - Thread-safe lazy initialization
 *
 * Usage:
 * @code
 * auto allocator = CPUDeviceAllocatorFactory::getInstance();
 * void* ptr = allocator->allocate(1024);
 * // ... use memory ...
 * allocator->release(ptr);
 * @endcode
 *
 * @note Uses Meyer's Singleton pattern for thread-safe initialization
 * @see CPUDeviceAllocator The allocator type produced by this factory
 */
class CPUDeviceAllocatorFactory {
 public:
  /**
   * @brief Gets the singleton CPUDeviceAllocator instance
   *
   * Creates the allocator on first call (lazy initialization).
   * Subsequent calls return the same shared instance.
   *
   * @return Shared pointer to the singleton CPUDeviceAllocator instance
   */
  static std::shared_ptr<CPUDeviceAllocator> getInstance() {
    if (!instance_) {
      instance_ = std::make_shared<CPUDeviceAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance_;  ///< Singleton instance
};

/**
 * @class CUDADeviceAllocatorFactory
 * @brief Singleton factory for CUDADeviceAllocator instances
 *
 * Implements the Singleton pattern to ensure a single shared instance of
 * CUDADeviceAllocator across the application. This is particularly important
 * for CUDA allocators because:
 * - Maintains a single unified memory pool across the application
 * - Prevents fragmentation from multiple independent allocators
 * - Optimizes memory reuse and reduces allocation overhead
 * - Ensures consistent memory management behavior
 *
 * Usage:
 * @code
 * auto allocator = CUDADeviceAllocatorFactory::getInstance();
 * void* ptr = allocator->allocate(1024 * 1024);  // 1MB on GPU
 * // ... use GPU memory ...
 * allocator->release(ptr);
 * @endcode
 *
 * @note Uses Meyer's Singleton pattern for thread-safe initialization
 * @see CUDADeviceAllocator The allocator type produced by this factory
 */
class CUDADeviceAllocatorFactory {
 public:
  /**
   * @brief Gets the singleton CUDADeviceAllocator instance
   *
   * Creates the allocator on first call (lazy initialization).
   * Subsequent calls return the same shared instance with its memory pools.
   *
   * @return Shared pointer to the singleton CUDADeviceAllocator instance
   */
  static std::shared_ptr<CUDADeviceAllocator> getInstance() {
    if (!instance_) {
      instance_ = std::make_shared<CUDADeviceAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance_;  ///< Singleton instance
};

}  // namespace base

#endif  // INFER_INCLUDE_BASE_ALLOC_H
