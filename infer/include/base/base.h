#ifndef INFER_INCLUDE_BASE_BASE_H
#define INFER_INCLUDE_BASE_BASE_H

#include <glog/logging.h>

#include <cstdint>
#include <ostream>

// TODO: what is this macro for?
#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

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

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnimplemented = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string errMsg = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t getErrCode() const;

  const std::string& getErrMsg() const;

  void setErrMsg(const std::string& errMsg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string msg_;
};

namespace error {
#define STATUS_CHECK(call)                                                  \
  do {                                                                      \
    const base::Status& status = call;                                      \
    if (!status) {                                                          \
      const size_t buf_size = 512;                                          \
      char buf[buf_size];                                                   \
      snprintf(                                                             \
          buf, buf_size - 1,                                                \
          "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
          __FILE__, __LINE__, int(status), status.getErrMsg().c_str());   \
      LOG(FATAL) << buf;                                                    \
    }                                                                       \
  } while (0)

Status Success(const std::string& errMsg = "");

Status FunctionNotImplement(const std::string& errMsg = "");

Status PathNotValid(const std::string& errMsg = "");

Status ModelParseError(const std::string& errMsg = "");

Status InternalError(const std::string& errMsg = "");

Status KeyHasExits(const std::string& errMsg = "");

Status InvalidArgument(const std::string& errMsg = "");

}  // namespace error

// TODO: how?
std::ostream& operator<<(std::ostream& os, const Status& status);

}  // namespace base

#endif  // INFER_INCLUDE_BASE_BASE_H
