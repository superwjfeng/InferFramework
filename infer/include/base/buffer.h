/**
 * @file buffer.h
 * @brief Memory buffer management for tensors and data storage
 *
 * This file defines the Buffer class, which provides an abstraction for
 * managing memory buffers across different device types (CPU, CUDA). It handles
 * memory allocation, deallocation, and copying operations while supporting both
 * internally-managed and externally-provided memory.
 */

#ifndef INFER_INCLUDE_BUFFER_H
#define INFER_INCLUDE_BUFFER_H

#include <cstddef>
#include <memory>

#include "alloc.h"
#include "base/base.h"

namespace base {
/**
 * @class Buffer
 * @brief Memory buffer abstraction with device-aware allocation
 *
 * This class provides a managed memory buffer that can reside on different
 * device types (CPU, CUDA). It supports two modes of operation:
 * 1. Internal allocation: Memory is allocated and managed by the buffer
 * 2. External memory: Buffer wraps user-provided memory without ownership
 *
 * Key features:
 * - Device-aware memory allocation using DeviceAllocator
 * - Automatic memory deallocation for internally-allocated buffers
 * - Support for memory copying between buffers
 * - Integration with std::enable_shared_from_this for safe shared ownership
 * - Non-copyable to prevent accidental memory duplication
 *
 * The buffer is non-copyable (inherits from NoCopyable) to prevent unintended
 * memory duplication, but supports explicit copying via copyFrom() method.
 *
 * Usage examples:
 * @code
 * // Create a buffer with internal allocation
 * auto allocator = CPUDeviceAllocatorFactory::getInstance();
 * auto buffer = std::make_shared<Buffer>(1024, allocator);
 * buffer->allocate();
 *
 * // Create a buffer wrapping external memory
 * void* external_ptr = malloc(1024);
 * auto ext_buffer = std::make_shared<Buffer>(1024, nullptr, external_ptr, true);
 *
 * // Copy data between buffers
 * buffer->copyFrom(ext_buffer);
 * @endcode
 *
 * @see DeviceAllocator For memory allocation strategies
 * @see NoCopyable Base class preventing copy operations
 */
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an uninitialized buffer with zero size and no allocator.
   * Call allocate() after setting up the buffer to allocate memory.
   */
  explicit Buffer() = default;

  /**
   * @brief Constructs a buffer with specified size and allocator
   *
   * @param byteSize Size of the buffer in bytes
   * @param allocator Device allocator to use for memory management
   *                  (nullptr if using external memory)
   * @param ptr External memory pointer (nullptr to allocate internally)
   * @param useExternal If true, buffer wraps external memory without ownership;
   *                    if false, buffer allocates and manages its own memory
   *
   * @note When useExternal is true, the caller is responsible for the lifetime
   *       of the memory pointed to by ptr. The Buffer will not free this memory.
   * @note When useExternal is false, memory will be allocated on first call to
   *       allocate() and freed in the destructor.
   */
  explicit Buffer(size_t byteSize,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool useExternal = false);

  /**
   * @brief Destructor that releases internally-allocated memory
   *
   * If the buffer owns its memory (useExternal_ is false), this destructor
   * releases the memory using the allocator. External memory is not freed.
   */
  virtual ~Buffer();

  /**
   * @brief Allocates memory for the buffer
   *
   * Uses the configured allocator to allocate memory of the specified size.
   * This method should only be called for internally-managed buffers.
   * For external buffers, this is a no-op.
   *
   * @return true if allocation succeeded, false otherwise
   *
   * @note This method has no effect if the buffer uses external memory
   * @note Requires that allocator_ is not nullptr for internal allocation
   */
  bool allocate();

  /**
   * @brief Copies data from another buffer
   *
   * Copies the contents of the source buffer to this buffer.
   * The buffers may reside on different devices; the copy will
   * use the appropriate memory copy operation (CPU<->CPU, CPU<->CUDA, etc.).
   *
   * @param buffer Source buffer to copy from
   *
   * @pre Both buffers must have the same size
   * @pre Both buffers must have allocated memory (ptr_ != nullptr)
   */
  void copyFrom(const Buffer& buffer) const;

  /**
   * @brief Copies data from another buffer (pointer variant)
   *
   * Convenience overload that accepts a pointer to a buffer.
   *
   * @param buffer Pointer to source buffer to copy from
   *
   * @pre buffer must not be nullptr
   * @pre Both buffers must have the same size
   * @pre Both buffers must have allocated memory (ptr_ != nullptr)
   */
  void copyFrom(const Buffer* buffer) const;

 public:
  /**
   * @brief Gets a mutable pointer to the buffer memory
   *
   * @return Pointer to the buffer's memory region
   * @warning Use with caution; modifying memory directly bypasses safety checks
   */
  inline void* ptr() { return ptr_; }

  /**
   * @brief Gets a const pointer to the buffer memory
   *
   * @return Const pointer to the buffer's memory region
   */
  inline const void* ptr() const { return ptr_; }

  /**
   * @brief Gets the size of the buffer in bytes
   *
   * @return Size of the buffer in bytes
   */
  inline size_t byteSize() const { return byteSize_; }

  /**
   * @brief Gets the allocator used by this buffer
   *
   * @return Shared pointer to the device allocator, or nullptr for external buffers
   */
  inline std::shared_ptr<DeviceAllocator> allocator() const {
    return allocator_;
  }

  /**
   * @brief Gets the device type where this buffer resides
   *
   * @return Device type (CPU, CUDA, etc.)
   */
  inline DeviceType deviceType() const { return deviceType_; }

  /**
   * @brief Sets the device type for this buffer
   *
   * @param deviceType New device type
   *
   * @note This does not move the memory; it only updates the metadata.
   *       Use copyFrom() with a buffer on a different device to actually transfer data.
   */
  inline void setDeviceType(DeviceType deviceType) { deviceType_ = deviceType; }

  /**
   * @brief Gets a shared pointer to this buffer
   *
   * Utilizes std::enable_shared_from_this to safely obtain a shared_ptr
   * to this buffer. Useful when the buffer needs to extend its own lifetime.
   *
   * @return Shared pointer to this buffer
   *
   * @warning Only call this method if the buffer was created with std::make_shared
   *          or is already managed by a shared_ptr
   */
  inline std::shared_ptr<Buffer> get_shared_from_this() {
    return shared_from_this();
  }

  /**
   * @brief Checks if the buffer uses external memory
   *
   * @return true if buffer wraps external memory, false if internally allocated
   */
  inline bool isExternal() const { return useExternal_; }

 private:
  size_t byteSize_ = 0;  ///< Size of the buffer in bytes

  /**
   * @brief Pointer to the buffer memory
   *
   * Two scenarios:
   * 1. If useExternal_ is true, ptr_ points to user-provided memory
   * 2. If useExternal_ is false, ptr_ is allocated by the allocator
   */
  void* ptr_ = nullptr;

  bool useExternal_ = false;  ///< Flag indicating if the buffer uses external memory
  DeviceType deviceType_ = DeviceType::kDeviceUnknown;  ///< Device type where buffer resides
  std::shared_ptr<DeviceAllocator> allocator_;  ///< Allocator for managing buffer memory
};

}  // namespace base

#endif  // INFER_INCLUDE_BUFFER_H
