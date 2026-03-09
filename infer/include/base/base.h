#ifndef INFER_INCLUDE_BASE_BASE_H
#define INFER_INCLUDE_BASE_BASE_H

#include <glog/logging.h>

#include <cstdint>

namespace base {

// * k stands for Const, which is a common naming convention for enum values.
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
  // TODO: more device type like FPGA, DSA, etc.
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFP32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

inline size_t dataTypeSize(base::DataType dataType) {
  switch (dataType) {
    case base::DataType::kDataTypeFP32:
      return sizeof(float);
    case base::DataType::kDataTypeInt8:
      return sizeof(int8_t);
    case base::DataType::kDataTypeInt32:
      return sizeof(int32_t);
    default:
      LOG(ERROR) << "Unknown data type.";
      return 0;
  }
}

class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

}  // namespace base

#endif  // INFER_INCLUDE_BASE_BASE_H
