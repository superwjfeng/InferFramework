#include "tensor/tensor.h"

#include <memory>
#include <vector>

#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool needRealloc) {
  if (!allocator) {
    LOG(ERROR) << "Allocator is null.";
    return false;
  }

  size_t byteSize = this->byteSize();
  if (!byteSize) {
    LOG(ERROR) << "Tensor byte size is zero.";
    return false;
  }

  if (buffer_ && byteSize <= buffer_->byteSize() && !needRealloc) {
    return true;
  }

  // not external buffer, therefore 3rd parameter is nullptr
  buffer_ = std::make_shared<base::Buffer>(byteSize, allocator, nullptr);

  if (!buffer_->ptr()) {
    LOG(ERROR) << "Failed to allocate buffer for tensor.";
    return false;
  }

  return true;
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "Buffer is null.";
    return false;
  }

  if (buffer) {
    if (buffer_->deviceType() != buffer->deviceType()) {
      LOG(ERROR) << "Buffer device type mismatch.";
      return false;
    }
  }

  if (byteSize() > buffer->byteSize()) {
    LOG(ERROR) << "Buffer byte size is smaller than tensor byte size.";
    return false;
  }

  buffer_ = buffer;
  return true;
}

void Tensor::initBuffer(std::shared_ptr<base::DeviceAllocator> alloc,
                        base::DataType dataType, bool needAlloc, void* ptr) {
  if (!alloc && needAlloc) {
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
        dataTypeSize(dataType) * size_, nullptr, ptr, true);
    buffer_ = buffer;
  } else {
    allocate(alloc, true);
  }
}

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = reduceDimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

Tensor Tensor::clone() const {
  Tensor newTensor = *this;
  size_t byteSize = this->byteSize();

  auto allocator = buffer_->allocator();
  newTensor.buffer_ = std::make_shared<base::Buffer>(byteSize, allocator);
  newTensor.buffer_->copyFrom(buffer_.get());
  return newTensor;
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduceDimension(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    dims_ = dims;
    size_ = size;
    return;
  }

  if (size > size_) {
    auto newBuffer = std::make_shared<base::Buffer>(
        size * base::dataTypeSize(dataType_), buffer_->allocator());
    CHECK(newBuffer->allocate());
    newBuffer->copyFrom(buffer_.get());
    buffer_ = newBuffer;
  }
  dims_ = dims;
  size_ = size;
}

Tensor::Tensor(base::DataType dataType, int32_t dim0, bool needAlloc,
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr)
    : dataType_(dataType) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (needAlloc && allocator) {
    allocate(allocator);
  } else {
    if (!ptr) {
      CHECK(needAlloc == false) << "Pointer is null but needAlloc is true.";
      initBuffer(allocator, dataType_, needAlloc, ptr);
    }
  }
}

Tensor::Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
               bool needAlloc, std::shared_ptr<base::DeviceAllocator> allocator,
               void* ptr)
    : dataType_(dataType) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (needAlloc && allocator) {
    allocate(allocator);
  } else {
    initBuffer(allocator, dataType_, needAlloc, ptr);
  }
}

Tensor::Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
               int32_t dim2, bool needAlloc,
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr)
    : dataType_(dataType) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (needAlloc && allocator) {
    allocate(allocator);
  } else {
    initBuffer(allocator, dataType_, needAlloc, ptr);
  }
}

Tensor::Tensor(base::DataType dataType, int32_t dim0, int32_t dim1,
               int32_t dim2, int32_t dim3, bool needAlloc,
               std::shared_ptr<base::DeviceAllocator> allocator, void* ptr)
    : dataType_(dataType) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (needAlloc && allocator) {
    allocate(allocator);
  } else {
    initBuffer(allocator, dataType_, needAlloc, ptr);
  }
}

Tensor::Tensor(base::DataType dataType, std::vector<int32_t> dims,
               bool needAlloc, std::shared_ptr<base::DeviceAllocator> allocator,
               void* ptr)
    : dims_(std::move(dims)), dataType_(dataType) {
  size_ = reduceDimension(dims_.begin(), dims_.end(), 1);
  if (needAlloc && allocator) {
    allocate(allocator);
  } else {
    initBuffer(allocator, dataType_, needAlloc, ptr);
  }
}

void Tensor::toCuda(cudaStream_t stream) {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType deviceType = this->deviceType();
  if (deviceType == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (deviceType == base::DeviceType::kDeviceCPU) {
    size_t byteSize = this->byteSize();
    auto cuAlloc = base::CUDADeviceAllocatorFactory::getInstance();
    auto cuBuffer = std::make_shared<base::Buffer>(byteSize, cuAlloc);
    cuAlloc->memcpy(buffer_->ptr(), cuBuffer->ptr(), byteSize,
                    base::MemcpyKind::kMemcpyCPU2CUDA, stream);
    this->buffer_ = cuBuffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

void Tensor::toCpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType deviceType = this->deviceType();

  if (deviceType == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (deviceType == base::DeviceType::kDeviceCUDA) {
    size_t byteSize = this->byteSize();
    auto cpuAlloc = base::CPUDeviceAllocatorFactory::getInstance();
    auto cpuBuffer = std::make_shared<base::Buffer>(byteSize, cpuAlloc);
    cpuAlloc->memcpy(buffer_->ptr(), cpuBuffer->ptr(), byteSize,
                     base::MemcpyKind::kMemcpyCUDA2CPU);
    this->buffer_ = cpuBuffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}

}  // namespace tensor
