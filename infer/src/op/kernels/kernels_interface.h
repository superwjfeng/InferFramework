#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

typedef void (*AddKernel)(const tensor::Tensor& input1,
                          const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);

AddKernel getAddKernel(base::DeviceType device_type);

MatmulKernel getMatmulKernel(base::DeviceType device_type);
}  // namespace kernel

#endif  // KERNELS_INTERFACE_H
