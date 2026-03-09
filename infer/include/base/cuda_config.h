#ifndef INFER_INCLUDE_BASE_CUDA_CONFIG_H
#define INFER_INCLUDE_BASE_CUDA_CONFIG_H
#include "cuda_runtime_api.h"
#include "driver_types.h"

namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel

#endif  // INFER_INCLUDE_BASE_CUDA_CONFIG_H
