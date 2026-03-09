#include "base/buffer.h"

#include <cstddef>

#include "base/alloc.h"
#include "base/base.h"

namespace base {
Buffer::Buffer(size_t byteSize, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool useExternal)
    : byteSize_(byteSize),
      ptr_(ptr),
      useExternal_(useExternal),
      allocator_(allocator) {
  if (!ptr_ && allocator_) {
    deviceType_ = allocator_->deviceType();
    useExternal_ = false;
    ptr_ = allocator_->allocate(byteSize);
  }
}

Buffer::~Buffer() {
  if (!useExternal_ && ptr_ && allocator_) {
    allocator_->release(ptr_);
    ptr_ = nullptr;
  }
}

bool Buffer::allocate() {
  if (allocator_ && byteSize_ > 0) {
    useExternal_ = false;
    ptr_ = allocator_->allocate(byteSize_);
    if (!ptr_) {
      LOG(ERROR) << "Failed to allocate buffer of size " << byteSize_;
      return false;
    } else {
      return true;
    }
  }
  return false;
}

void Buffer::copyFrom(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr) << "Allocator is null.";
  CHECK(buffer.ptr_ != nullptr) << "Source buffer pointer is null.";

  size_t byteSize = std::min(byteSize_, buffer.byteSize_);
  const DeviceType& bufferDeviceType = buffer.deviceType();
  const DeviceType& currentDeviceType = deviceType_;
  CHECK(bufferDeviceType != DeviceType::kDeviceUnknown &&
        currentDeviceType != DeviceType::kDeviceUnknown)
      << "Unknown device type.";

  if (bufferDeviceType == DeviceType::kDeviceCPU &&
      currentDeviceType == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr_, ptr_, byteSize);
  } else if (bufferDeviceType == DeviceType::kDeviceCUDA &&
             currentDeviceType == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), ptr_, byteSize,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (bufferDeviceType == DeviceType::kDeviceCPU &&
             currentDeviceType == DeviceType::kDeviceCUDA) {
    allocator_->memcpy(buffer.ptr_, ptr_, byteSize,
                       MemcpyKind::kMemcpyCPU2CUDA);
  } else if (bufferDeviceType == DeviceType::kDeviceCUDA &&
             currentDeviceType == DeviceType::kDeviceCUDA) {
    allocator_->memcpy(buffer.ptr_, ptr_, byteSize,
                       MemcpyKind::kMemcpyCUDA2CUDA);
  } else {
    LOG(FATAL) << "Unsupported device type combination: "
               << static_cast<int>(bufferDeviceType) << " to "
               << static_cast<int>(currentDeviceType);
  }
}

void Buffer::copyFrom(const Buffer* buffer) const {
  CHECK(buffer != nullptr) << "Source buffer is null.";
  copyFrom(*buffer);
}

}  // namespace base
