#include <gtest/gtest.h>

#include "base/alloc.h"
#include "base/buffer.h"

TEST(testBuffer, allocate) {
  using namespace base;
  auto allocator = CPUDeviceAllocatorFactory::getInstance();
  {
    Buffer buffer(32, allocator);
    ASSERT_NE(buffer.ptr(), nullptr);
    LOG(INFO) << "HERE1";
  }
  LOG(INFO) << "HERE2";
}

TEST(testBuffer, allocate2) {
  using namespace base;
  auto allocator = CPUDeviceAllocatorFactory::getInstance();
  std::shared_ptr<Buffer> buffer;
  {
    buffer = std::make_shared<Buffer>(32, allocator);
  }
  LOG(INFO) << "HERE";
  ASSERT_NE(buffer->ptr(), nullptr);
}

TEST(testBuffer, useExternal) {
  using namespace base;
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.isExternal(), true);
  delete[] ptr;
}
