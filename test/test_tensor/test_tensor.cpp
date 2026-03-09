#include <gtest/gtest.h>

#include "base/alloc.h"
#include "base/base.h"
#include "tensor/tensor.h"

TEST(testTensor, init1) {
  using namespace base;
  auto allocator = CPUDeviceAllocatorFactory::getInstance();

  int32_t size = 32 * 151;
  tensor::Tensor t1(base::DataType::kDataTypeFP32, size, true, allocator);
  ASSERT_EQ(t1.empty(), false);
}
