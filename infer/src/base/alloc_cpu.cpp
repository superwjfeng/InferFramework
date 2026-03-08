#include "base/alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byteSize) const {
  void* data = std::malloc(byteSize);
  CHECK(data != nullptr) << "Failed to allocate " << byteSize
                         << " bytes of CPU memory.";
  return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance_ = nullptr;

}  // namespace base
