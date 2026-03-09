#include <gtest/gtest.h>

#include "../src/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/alloc.h"
#include "base/base.h"
#include "cuda_runtime_api.h"
#include "tensor/tensor.h"

TEST(testAddCu, add1NoStream) {
  auto allocator = base::CUDADeviceAllocatorFactory::getInstance();
  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFP32, size, true, allocator);
  tensor::Tensor t2(base::DataType::kDataTypeFP32, size, true, allocator);
  tensor::Tensor out(base::DataType::kDataTypeFP32, size, true, allocator);

  setValueCU(static_cast<float*>(t1.buffer()->ptr()), size, 2.0f);
  setValueCU(static_cast<float*>(t2.buffer()->ptr()), size, 3.0f);

  kernel::getAddKernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  delete[] output;
}
