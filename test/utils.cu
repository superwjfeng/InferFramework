#include <glog/logging.h>

#include "utils.cuh"

__global__ void testFunctionCu(float* cuArr, int32_t size, float value) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  cuArr[tid] = value;
}

void testFunction(float* arr, int32_t size, float value) {
  if (!arr) {
    return;
  }

  float* cuArr = nullptr;
  cudaMalloc(&cuArr, sizeof(float) * size);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  testFunctionCu<<<1, size>>>(cuArr, size, value);
  cudaDeviceSynchronize();
  const cudaError_t err2 = cudaGetLastError();
  CHECK_EQ(err2, cudaSuccess);

  cudaMemcpy(arr, cuArr, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaFree(cuArr);
}

void setValueCU(float* arrCu, int32_t size, float value) {
  int32_t threadsNum = 512;
  int32_t blockNum = (size + threadsNum - 1) / threadsNum;
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  testFunctionCu<<<blockNum, threadsNum>>>(arrCu, size, value);
  cudaDeviceSynchronize();
  const cudaError_t err2 = cudaGetLastError();
  CHECK_EQ(err2, cudaSuccess);
}
