#include "model/raw_model_data.h"

#include <sys/mman.h>
#include <unistd.h>

namespace model {
RawModelData::~RawModelData() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, fileSize);
    data = nullptr;
  }

  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

const void* RawModelDataFp32::weight(size_t offset) const {
  return static_cast<float*>(weightData) + offset;
}

const void* RawModelDataInt8::weight(size_t offset) const {
  return static_cast<int8_t*>(weightData) + offset;
}

}  // namespace model
