#include "base/base.h"

#include <string>
namespace base {
Status::Status(int code, std::string errMsg)
    : code_(code), msg_(std::move(errMsg)) {}

Status& Status::operator=(int code) {
  code_ = code;
  return *this;
};

bool Status::operator==(int code) const {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int code) const {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kSuccess; }

int32_t Status::getErrCode() const { return code_; }

const std::string& Status::getErrMsg() const { return msg_; }

void Status::setErrMsg(const std::string& errMsg) { msg_ = errMsg; }

namespace error {
Status Success(const std::string& errMsg) { return Status{kSuccess, errMsg}; }

Status FunctionNotImplement(const std::string& errMsg) {
  return Status{kFunctionUnimplemented, errMsg};
}

Status PathNotValid(const std::string& errMsg) {
  return Status{kPathNotValid, errMsg};
}

Status ModelParseError(const std::string& errMsg) {
  return Status{kModelParseError, errMsg};
}

Status InternalError(const std::string& errMsg) {
  return Status{kInternalError, errMsg};
}

Status InvalidArgument(const std::string& errMsg) {
  return Status{kInvalidArgument, errMsg};
}

Status KeyHasExits(const std::string& errMsg) {
  return Status{kKeyValueHasExist, errMsg};
}
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.getErrMsg();
  return os;
}

}  // namespace base
