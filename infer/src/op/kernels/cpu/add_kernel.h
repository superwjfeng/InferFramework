#ifndef INFER_SRC_OP_KERNELS_CPU_ADD_KERNEL_H_
#define INFER_SRC_OP_KERNELS_CPU_ADD_KERNEL_H_
#include "tensor/tensor.h"

namespace kernel {
void addKernelCpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel

#endif  // INFER_SRC_OP_KERNELS_CPU_ADD_KERNEL_H_
