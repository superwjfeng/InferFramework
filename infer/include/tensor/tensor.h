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
namespace tensor {

// TODO: can we use std::reduce to replace this func?
// Helper function to calculate the total size of the tensor based on its
// dimensions
template <typename T, typename Tp>
static size_t reduceDimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }

  size_t size = std::accumulate(begin, end, init, std::multiplies<Tp>());
  return size;
}


class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(base::DataType dataType, int32_t dim0, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  int32_t dim2, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
                  int32_t dim2, int32_t dim3, bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType dataType, std::vector<int32_t> dims,
                  bool needAlloc = false,
                  std::shared_ptr<base::DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

 public:
  inline size_t size() const { return size_; }

  inline size_t byteSize() const {
    return size_ * dataTypeSize(dataType_);
  }

  inline const std::vector<int32_t>& dims() const { return dims_; }

  inline int32_t getDim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, dims_.size());
    return dims_.at(idx);
  }

  inline const std::shared_ptr<base::Buffer>& buffer() const { return buffer_; }

  inline base::DataType dataType() const { return dataType_; }

  inline bool empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
  }

  inline base::DeviceType deviceType() const {
    if (buffer_) {
      return buffer_->deviceType();
    } else {
      return base::DeviceType::kDeviceUnknown;
    }
  }

  inline void setDeviceType(base::DeviceType deviceType) const {
    if (buffer_) {
      buffer_->setDeviceType(deviceType);
    }
  }

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
  void toCpu();

  void toCuda(cudaStream_t stream = nullptr);

  void initBuffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType,
                  bool needAlloc = false, void* ptr = nullptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  void reshape(const std::vector<int32_t>& dims);

  Tensor clone() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool needRealloc = false);

 private:
  // size_ is the total number of elements in the tensor
  size_t size_ = 0;
  // dims_ is a vector that holds the size of each dimension of the tensor. For
  // example, if the tensor is a 3D tensor with dimensions 2x3x4, then dims_
  // would be {2, 3, 4}.
  std::vector<int32_t> dims_;
  // buffer_ is a shared pointer to a Buffer object that manages the memory for
  // the tensor's data. The Buffer class likely handles memory allocation,
  // deallocation, and possibly data transfer between different devices (e.g.,
  // CPU and GPU).
  std::shared_ptr<base::Buffer> buffer_;
  // dataType_ specifies the data type of the elements in the tensor
  base::DataType dataType_ = base::DataType::kDataTypeUnknown;
};
}  // namespace tensor
#endif  // INFER_INCLUDE_TENSOR_TENSOR_H_
