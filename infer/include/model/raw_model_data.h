#ifndef INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#define INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_

#include <cstddef>
#include <cstdint>
namespace model {
struct RawModelData {
  ~RawModelData();

  int32_t fd = -1;
  size_t fileSize = 0;
  void* data = nullptr;
  void* weightData = nullptr;

  virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};

}  // namespace model
#endif  // INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_
