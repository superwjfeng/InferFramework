#ifndef INFER_INCLUDE_BUFFER_H
#define INFER_INCLUDE_BUFFER_H

#include <cstddef>
#include <memory>

#include "alloc.h"
#include "base/base.h"

namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byteSize,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool useExternal = false);

  virtual ~Buffer();

  bool allocate();

  void copyFrom(const Buffer& buffer) const;

  void copyFrom(const Buffer* buffer) const;

 public:
  inline void* ptr() { return ptr_; }

  inline const void* ptr() const { return ptr_; }

  inline size_t byteSize() const { return byteSize_; }

  inline std::shared_ptr<DeviceAllocator> allocator() const {
    return allocator_;
  }

  inline DeviceType deviceType() const { return deviceType_; }

  inline void setDeviceType(DeviceType deviceType) { deviceType_ = deviceType; }

  inline std::shared_ptr<Buffer> get_shared_from_this() {
    return shared_from_this();
  }

  inline bool isExternal() const { return useExternal_; }

 private:
  // The size of the buffer in bytes
  size_t byteSize_ = 0;
  // Pointer to the buffer memory, two scenarios:
  // 1. If useExternal_ is true, ptr_ is provided by the user
  // 2. If useExternal_ is false, ptr_ is allocated by the allocator
  void* ptr_ = nullptr;
  // Flag indicating if the buffer uses external memory
  bool useExternal_ = false;
  // The type of device the buffer is associated with
  DeviceType deviceType_ = DeviceType::kDeviceUnknown;
  // Allocator used for managing the buffer's memory
  std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace base

#endif  // INFER_INCLUDE_BUFFER_H
