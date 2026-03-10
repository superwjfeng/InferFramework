/**
 * @file cuda_config.h
 * @brief CUDA configuration and resource management
 *
 * This file defines the configuration structure for CUDA runtime resources,
 * including stream management and automatic cleanup.
 */

#ifndef INFER_INCLUDE_BASE_CUDA_CONFIG_H
#define INFER_INCLUDE_BASE_CUDA_CONFIG_H
#include "cuda_runtime_api.h"
#include "driver_types.h"

/**
 * @namespace kernel
 * @brief Namespace for kernel execution and CUDA runtime operations
 */
namespace kernel {

/**
 * @struct CudaConfig
 * @brief Configuration structure for CUDA runtime resources
 *
 * This structure manages CUDA runtime resources with RAII semantics,
 * ensuring proper cleanup of allocated resources when the object is destroyed.
 */
struct CudaConfig {
  /**
   * @brief CUDA stream for asynchronous kernel execution
   *
   * The stream is used to manage the execution order of CUDA operations.
   * Initialized to nullptr by default. When set, it will be automatically
   * destroyed in the destructor.
   */
  cudaStream_t stream = nullptr;

  /**
   * @brief Destructor that cleans up CUDA resources
   *
   * Automatically destroys the CUDA stream if it has been allocated.
   * This ensures proper resource cleanup following RAII principles.
   */
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel

#endif  // INFER_INCLUDE_BASE_CUDA_CONFIG_H
