#include <cuda_runtime_api.h>

#include <cstdio>

#include "base/alloc.h"
#include "driver_types.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byteSize) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  // TODO: make constant for big buffer threshold
  if (byteSize > 1024 * 1024) {
    auto& bigBuffers = bigBuffersMap_[id];
    int selId = -1;
    for (int i = 0; i < bigBuffers.size(); ++i) {
      if (bigBuffers.at(i).byteSize >= byteSize && !bigBuffers.at(i).busy &&
          bigBuffers.at(i).byteSize - byteSize < 1 * 1024 * 1024) {
        if (selId == -1 ||
            bigBuffers[selId].byteSize > bigBuffers.at(i).byteSize) {
          selId = i;
        }
      }
    }
    if (selId != -1) {
      bigBuffers[selId].busy = true;
      return bigBuffers[selId].data;
    }

    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byteSize);
    if (state != cudaSuccess) {
      char buf[256];
      snprintf(buf, sizeof(buf),
               "cudaMalloc failed with error code %d when trying to allocate "
               "%zu bytes\n",
               state, byteSize);
      LOG(ERROR) << buf;
      return nullptr;
    }
    bigBuffers.emplace_back(ptr, byteSize, true);
    return ptr;
  }

  auto& cudaBuffers = cudaBuffersMap_[id];
  for (int i = 0; i < cudaBuffers.size(); ++i) {
    if (cudaBuffers.at(i).byteSize >= byteSize && !cudaBuffers.at(i).busy) {
      cudaBuffers.at(i).busy = true;
      noBusyCnt_[id] -= cudaBuffers.at(i).byteSize;
      return cudaBuffers.at(i).data;
    }
  }

  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byteSize);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "cudaMalloc failed with error code %d when trying to allocate "
             "%zu bytes\n",
             state, byteSize);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cudaBuffers.emplace_back(ptr, byteSize, true);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (cudaBuffersMap_.empty()) {
    return;
  }

  cudaError_t state = cudaSuccess;
  for (auto& it : cudaBuffersMap_) {
    if (noBusyCnt_.at(it.first) > 1024 * 1024 * 1024) {
      auto& cudaBuffers = it.second;
      std::vector<CudaMemoryBuffer> tmp;
      for (int i = 0; i < cudaBuffers.size(); ++i) {
        if (!cudaBuffers.at(i).busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cudaBuffers.at(i).data);
          CHECK(state = cudaSuccess)
              << "cudaFree failed with error code " << state;
        } else {
          tmp.push_back(cudaBuffers.at(i));
        }
      }

      cudaBuffers.clear();
      it.second = tmp;
      noBusyCnt_[it.first] = 0;
    }
  }

  for (auto& it : cudaBuffersMap_) {
    auto& cudaBuffers = it.second;
    for (int i = 0; i < cudaBuffers.size(); ++i) {
      if (cudaBuffers.at(i).data == ptr) {
        noBusyCnt_.at(it.first) += cudaBuffers.at(i).byteSize;
        cudaBuffers.at(i).busy = false;
        return;
      }
    }

    auto& bigBuffers = bigBuffersMap_[it.first];
    for (int i = 0; i < bigBuffers.size(); ++i) {
      if (bigBuffers.at(i).data == ptr) {
        bigBuffers.at(i).busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "cudaFree failed with error code " << state;
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance_ =
    nullptr;

};  // namespace base
