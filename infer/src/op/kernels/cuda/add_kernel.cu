#include "add_kernel.cuh"

namespace kernel {
__global__ void addKernelCuFp32(int32_t size, const float* in1,
                                const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float inVal1 = in1[tid];
  float inVal2 = in2[tid];
  out[tid] = inVal1 + inVal2;
}

void addKernelCu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                 const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.empty(), false);
  CHECK_EQ(input2.empty(), false);
  CHECK_EQ(output.empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t threadNum = 512;
  int32_t blockNum = (size + threadNum - 1) / threadNum;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    addKernelCuFp32<<<blockNum, threadNum, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(),
        const_cast<float*>(output.ptr<float>()));
  } else {
    addKernelCuFp32<<<blockNum, threadNum>>>(
        size, input1.ptr<float>(), input2.ptr<float>(),
        const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel
