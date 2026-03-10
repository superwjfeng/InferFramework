#ifndef INFER_SRC_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#define INFER_SRC_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH

namespace kernel {
size_t argmaxKernelCu(const float* inputPtr, int32_t size, void* stream);
}  // namespace kernel

#endif  // INFER_SRC_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
