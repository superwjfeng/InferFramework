#ifndef INFER_INCLUDE_BASE_ALLOC_H
#define INFER_INCLUDE_BASE_ALLOC_H
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "base.h"

namespace base {

enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType deviceType) : deviceType_(deviceType) {}

  virtual DeviceType deviceType() const { return deviceType_; }

 public:
  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t) const = 0;

  virtual void memcpy(const void* srcPtr, void* destPtr, size_t byteSize,
                      MemcpyKind memcpyKind = MemcpyKind::kMemcpyCPU2CPU,
                      void* stream = nullptr, bool needSync = false) const;

  virtual void memsetZero(void* ptr, size_t byteSize, void* stream,
                          bool needSync = false);

 private:
  DeviceType deviceType_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void release(void* ptr) const override;

  void* allocate(size_t byteSize) const override;
};

struct CudaMemoryBuffer {
  void* data;
  size_t byteSize;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byteSize, bool busy)
      : data(data), byteSize(byteSize), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void release(void* ptr) const override;

  void* allocate(size_t byteSize) const override;

 private:
  mutable std::map<int, size_t> noBusyCnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap_;
};

// TODO: Meyer's Singleton pattern for allocator factory
class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> getInstance() {
    if (!instance_) {
      instance_ = std::make_shared<CPUDeviceAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance_;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> getInstance() {
    if (!instance_) {
      instance_ = std::make_shared<CUDADeviceAllocator>();
    }
    return instance_;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance_;
};

}  // namespace base

#endif  // INFER_INCLUDE_BASE_ALLOC_H
