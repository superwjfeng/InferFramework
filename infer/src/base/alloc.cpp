#include "base/alloc.h"

#include <cuda_runtime_api.h>

namespace base {
void DeviceAllocator::memcpy(const void* srcPtr, void* destPtr, size_t byteSize,
                             MemcpyKind memcpyKind, void* stream,
                             bool needSync) const {
  CHECK_NE(srcPtr, nullptr);
  CHECK_NE(destPtr, nullptr);

  if (!byteSize) {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpyKind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(destPtr, srcPtr, byteSize);
  } else if (memcpyKind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(destPtr, srcPtr, byteSize, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(destPtr, srcPtr, byteSize, cudaMemcpyHostToDevice,
                      stream_);
    }
  } else if (memcpyKind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(destPtr, srcPtr, byteSize, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(destPtr, srcPtr, byteSize, cudaMemcpyDeviceToHost,
                      stream_);
    }
  } else if (memcpyKind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(destPtr, srcPtr, byteSize, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(destPtr, srcPtr, byteSize, cudaMemcpyDeviceToDevice,
                      stream_);
    }
  } else {
    LOG(FATAL) << "Unsupported memcpy kind: " << static_cast<int>(memcpyKind);
  }

  if (needSync) {
    cudaDeviceSynchronize();
  }
}

void DeviceAllocator::memsetZero(void* ptr, size_t byteSize, void* stream,
                                 bool needSync) {
  CHECK(deviceType_ != DeviceType::kDeviceUnkown);
  if (deviceType_ == DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byteSize);
  } else if (deviceType_ == DeviceType::kDeviceCUDA) {
    cudaStream_t stream_ = nullptr;
    if (stream) {
      stream_ = static_cast<CUstream_st*>(stream);
      cudaMemsetAsync(ptr, 0, byteSize, stream_);
    } else {
      cudaMemset(ptr, 0, byteSize);
    }
    if (needSync) {
      cudaDeviceSynchronize();
    }
  } else {
    LOG(FATAL) << "Unsupported device type: " << static_cast<int>(deviceType_);
  }
}

}  // namespace base
