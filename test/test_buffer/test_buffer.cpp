#include <gtest/gtest.h>
#include "base/alloc.h"
#include "base/buffer.h"

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = CPUDeviceAllocatorFactory::getInstance();
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}
