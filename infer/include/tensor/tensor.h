/**
 * @file tensor.h
 * @brief Multi-dimensional tensor data structure and operations
 *
 * This file defines the Tensor class, a fundamental data structure for
 * storing and manipulating multi-dimensional arrays in neural network
 * computations. Supports both CPU and CUDA memory management.
 */

#ifndef INFER_INCLUDE_TENSOR_TENSOR_H_
#define INFER_INCLUDE_TENSOR_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "base/alloc.h"
#include "base/base.h"
#include "base/buffer.h"
#include "driver_types.h"
#include "glog/logging.h"

/**
 * @namespace tensor
 * @brief Namespace for tensor data structures and operations
 */
namespace tensor {

/**
 * @brief Calculates the total number of elements from dimension sizes
 *
 * Multiplies all dimension sizes together to compute the total tensor size.
 * This is a helper function used internally for tensor size calculations.
 *
 * @tparam T Iterator type for dimension container
 * @tparam Tp Type of the initial value and result
 * @param begin Iterator to the beginning of dimensions
 * @param end Iterator to the end of dimensions
 * @param init Initial value for the accumulation (typically 1)
 * @return Total number of elements (product of all dimensions), or 0 if empty
 *
 * @note TODO: Consider using std::reduce for better performance
 */
template <typename T, typename Tp>
static size_t reduceDimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }

  size_t size = std::accumulate(begin, end, init, std::multiplies<Tp>());
  return size;
}

/**
 * @class Tensor
 * @brief Multi-dimensional array container for neural network computations
 *
 * The Tensor class represents a multi-dimensional array with support for
 * different data types and device memory (CPU/CUDA). It manages memory
 * allocation, data transfer between devices, and provides efficient access
 * to tensor elements.
 *
 * Key features:
 * - Supports 1D to 4D tensors (and arbitrary dimensions via vector constructor)
 * - Automatic memory management with shared_ptr
 * - CPU and CUDA memory support
 * - Flexible memory allocation strategies
 * - Type-safe element access
 */
class Tensor {
 public:
  /** @brief Default constructor creates an empty tensor */
  Tensor() = default;

  /**
   * @brief Constructs a 1D tensor
   * @param dataType Data type of tensor elements
   * @param dim0 Size of the first dimension
   * @param needAlloc Whether to allocate memory immediately
   * @param allocator Custom memory allocator (optional)
   * @param ptr Existing memory pointer to use (optional)
   */
  explicit Tensor(base::DataType dataType, int32_t dim0, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  /**
   * @brief Constructs a 2D tensor
   * @param dataType Data type of tensor elements
   * @param dim0 Size of the first dimension
   * @param dim1 Size of the second dimension
   * @param needAlloc Whether to allocate memory immediately
   * @param allocator Custom memory allocator (optional)
   * @param ptr Existing memory pointer to use (optional)
   */
  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  /**
   * @brief Constructs a 3D tensor
   * @param dataType Data type of tensor elements
   * @param dim0 Size of the first dimension
   * @param dim1 Size of the second dimension
   * @param dim2 Size of the third dimension
   * @param needAlloc Whether to allocate memory immediately
   * @param allocator Custom memory allocator (optional)
   * @param ptr Existing memory pointer to use (optional)
   */
  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  int32_t dim2, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  /**
   * @brief Constructs a 4D tensor
   * @param dataType Data type of tensor elements
   * @param dim0 Size of the first dimension
   * @param dim1 Size of the second dimension
   * @param dim2 Size of the third dimension
   * @param dim3 Size of the fourth dimension
   * @param needAlloc Whether to allocate memory immediately
   * @param allocator Custom memory allocator (optional)
   * @param ptr Existing memory pointer to use (optional)
   */
  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  int32_t dim2, int32_t dim3, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  /**
   * @brief Constructs a tensor with arbitrary dimensions
   * @param dataType Data type of tensor elements
   * @param dims Vector specifying size of each dimension
   * @param needAlloc Whether to allocate memory immediately
   * @param allocator Custom memory allocator (optional)
   * @param ptr Existing memory pointer to use (optional)
   */
  explicit Tensor(base::DataType dataType, std::vector<int32_t> dims,
                  bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

 public:
  /**
   * @brief Gets the total number of elements in the tensor
   * @return Total number of elements across all dimensions
   */
  inline size_t size() const { return size_; }

  /**
   * @brief Gets the total size in bytes of the tensor data
   * @return Size in bytes (number of elements × size of each element)
   */
  inline size_t byteSize() const { return size_ * dataTypeSize(dataType_); }

  /**
   * @brief Gets the dimension sizes of the tensor
   * @return Vector containing the size of each dimension
   */
  inline const std::vector<int32_t>& dims() const { return dims_; }

  /**
   * @brief Gets the number of dimensions of the tensor
   * @return Number of dimensions (e.g., 2 for a matrix, 3 for a 3D tensor)
   */
  inline int32_t dimSize() const { return dims_.size(); }

  /**
   * @brief Gets the size of a specific dimension
   * @param idx Index of the dimension (0-based)
   * @return Size of the specified dimension
   */
  inline int32_t getDim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, dims_.size());
    return dims_.at(idx);
  }

  /**
   * @brief Gets the underlying memory buffer
   * @return Shared pointer to the buffer managing tensor data
   */
  inline const std::shared_ptr<base::Buffer>& buffer() const { return buffer_; }

  /**
   * @brief Gets the data type of tensor elements
   * @return Data type enum (e.g., float32, int32)
   */
  inline base::DataType dataType() const { return dataType_; }

  /**
   * @brief Checks if the tensor is empty
   * @return true if tensor has no elements or unallocated memory, false otherwise
   */
  inline bool empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
  }

  /**
   * @brief Gets the device type where tensor data resides
   * @return Device type (CPU, CUDA, or Unknown if no buffer)
   */
  inline base::DeviceType deviceType() const {
    if (buffer_) {
      return buffer_->deviceType();
    } else {
      return base::DeviceType::kDeviceUnknown;
    }
  }

  /**
   * @brief Sets the device type of the tensor
   * @param deviceType Target device type (CPU or CUDA)
   */
  inline void setDeviceType(base::DeviceType deviceType) const {
    if (buffer_) {
      buffer_->setDeviceType(deviceType);
    }
  }

  /**
   * @brief Resets tensor metadata without reallocating memory
   * @param dataType New data type for the tensor
   * @param dims New dimension sizes
   */
  inline void reset(base::DataType dataType, const std::vector<int32_t>& dims) {
    dataType_ = dataType;
    dims_ = dims;
    size_ = 1;
    for (int32_t dim : dims) {
      size_ = reduceDimension(dims_.begin(), dims_.end(), 1);
    }
    buffer_.reset();
  }

 public:
  /**
   * @brief Transfers tensor data to CPU memory
   *
   * If the tensor is currently on CUDA, this copies the data to CPU memory.
   */
  void toCpu();

  /**
   * @brief Transfers tensor data to CUDA memory
   * @param stream CUDA stream for asynchronous transfer (optional)
   *
   * If the tensor is currently on CPU, this copies the data to CUDA memory.
   */
  void toCuda(cudaStream_t stream = nullptr);

  /**
   * @brief Initializes the tensor's memory buffer
   * @param alloc Device allocator for memory management
   * @param dataType Data type of tensor elements (unused parameter name)
   * @param needAlloc Whether to allocate memory immediately
   * @param ptr Existing memory pointer to use instead of allocating
   */
  void initBuffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType,
                  bool needAlloc = false, void* ptr = nullptr);

  /**
   * @brief Gets a typed pointer to the tensor data
   * @tparam T Element type to cast to
   * @return Pointer to the beginning of tensor data, or nullptr if empty
   */
  template <typename T>
  T* ptr();

  /**
   * @brief Gets a const typed pointer to the tensor data
   * @tparam T Element type to cast to
   * @return Const pointer to the beginning of tensor data, or nullptr if empty
   */
  template <typename T>
  const T* ptr() const;

  /**
   * @brief Gets a typed pointer to a specific element
   * @tparam T Element type to cast to
   * @param index Linear index of the element
   * @return Pointer to the element at the specified index
   */
  template <typename T>
  T* ptr(int64_t index);

  /**
   * @brief Gets a const typed pointer to a specific element
   * @tparam T Element type to cast to
   * @param index Linear index of the element
   * @return Const pointer to the element at the specified index
   */
  template <typename T>
  const T* ptr(int64_t index) const;

  /**
   * @brief Accesses a tensor element by linear index
   * @tparam T Element type
   * @param offset Linear offset into the tensor data
   * @return Reference to the element at the specified offset
   */
  template <typename T>
  T& index(int64_t offset);

  /**
   * @brief Accesses a tensor element by linear index (const version)
   * @tparam T Element type
   * @param offset Linear offset into the tensor data
   * @return Const reference to the element at the specified offset
   */
  template <typename T>
  const T& index(int64_t offset) const;

  /**
   * @brief Reshapes the tensor to new dimensions
   * @param dims New dimension sizes
   *
   * The total number of elements must remain the same.
   */
  void reshape(const std::vector<int32_t>& dims);

  /**
   * @brief Creates a deep copy of the tensor
   * @return New tensor with copied data
   */
  Tensor clone() const;

  /**
   * @brief Calculates the stride for each dimension
   * @return Vector of strides (elements to skip per dimension)
   *
   * Strides indicate how many elements to skip in memory to move one
   * position along each dimension.
   */
  std::vector<size_t> strides() const;

  /**
   * @brief Assigns an existing buffer to this tensor
   * @param buffer Shared pointer to the buffer to assign
   * @return true if assignment succeeded, false otherwise
   */
  bool assign(std::shared_ptr<base::Buffer> buffer);

  /**
   * @brief Allocates memory for the tensor
   * @param allocator Device allocator for memory management
   * @param needRealloc Whether to force reallocation if already allocated
   * @return true if allocation succeeded, false otherwise
   */
  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool needRealloc = false);

 private:
  /** @brief Total number of elements in the tensor (product of all dimensions) */
  size_t size_ = 0;

  /**
   * @brief Dimension sizes of the tensor
   *
   * Vector holding the size of each dimension. For example, a 3D tensor
   * with dimensions 2×3×4 would have dims_ = {2, 3, 4}.
   */
  std::vector<int32_t> dims_;

  /**
   * @brief Shared pointer to the buffer managing tensor data
   *
   * The Buffer class handles memory allocation, deallocation, and data
   * transfer between different devices (e.g., CPU and GPU).
   */
  std::shared_ptr<base::Buffer> buffer_;

  /** @brief Data type of tensor elements (e.g., float32, int32) */
  base::DataType dataType_ = base::DataType::kDataTypeUnknown;
};

// Template method implementations

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, size_);
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, size_);
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor
#endif  // INFER_INCLUDE_TENSOR_TENSOR_H_
