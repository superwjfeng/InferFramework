/**
 * @file base.h
 * @brief Core types and utilities for the inference framework
 *
 * This file defines fundamental types, enumerations, and utilities used
 * throughout the inference framework including device types, data types,
 * model types, status handling, and error reporting.
 */

#ifndef INFER_INCLUDE_BASE_BASE_H
#define INFER_INCLUDE_BASE_BASE_H

#include <glog/logging.h>

#include <cstdint>
#include <ostream>

/**
 * @def UNUSED
 * @brief Macro to suppress unused variable warnings
 *
 * This macro can be used to explicitly mark variables as intentionally unused,
 * preventing compiler warnings while making the intent clear in the code.
 *
 * Usage:
 * @code
 * void func(int needed, int optional) {
 *   UNUSED(optional);  // Suppress warning for optional parameter
 *   // ... use only 'needed' ...
 * }
 * @endcode
 */
#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

namespace model {
/**
 * @enum ModelBufferType
 * @brief Enumeration of buffer types used in neural network models
 *
 * Identifies different types of intermediate buffers and tensors used during
 * model inference. These buffers store various stages of computation including
 * inputs, embeddings, attention outputs, feed-forward outputs, and caches.
 */
enum class ModelBufferType {
  kInputTokens = 0,        ///< Input token IDs buffer
  kInputEmbeddings = 1,    ///< Token embeddings after embedding layer
  kOutputRMSNorm = 2,      ///< Output after RMSNorm normalization
  kKeyCache = 3,           ///< Key cache for attention mechanism (KV cache)
  kValueCache = 4,         ///< Value cache for attention mechanism (KV cache)
  kQuery = 5,              ///< Query tensor in attention computation
  kInputPos = 6,           ///< Position indices for tokens
  kScoreStorage = 7,       ///< Attention score storage
  kOutputMHA = 8,          ///< Multi-head attention output
  kAttnOutput = 9,         ///< Attention output after projection
  kW1Output = 10,          ///< Feed-forward W1 layer output
  kW2Output = 11,          ///< Feed-forward W2 layer output
  kW3Output = 12,          ///< Feed-forward W3 layer output
  kFFNRMSNorm = 13,        ///< Feed-forward network RMSNorm output
  kForwardOutput = 15,     ///< Final forward pass output
  kForwardOutputCPU = 16,  ///< Forward output copied to CPU

  kSinCache = 17,          ///< Sine values cache for RoPE
  kCosCache = 18,          ///< Cosine values cache for RoPE
};
}  // namespace model

namespace base {
/**
 * @enum DeviceType
 * @brief Enumeration of supported device types for computation
 *
 * Specifies where tensor operations and model inference should execute.
 * Different devices have different performance characteristics and memory
 * hierarchies.
 *
 * @note The 'k' prefix is a common naming convention for enum constants
 */
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,  ///< Unknown or uninitialized device type
  kDeviceCPU = 1,      ///< CPU (host) device
  kDeviceCUDA = 2,     ///< NVIDIA CUDA GPU device
  // TODO: more device type like FPGA, DSA, etc.
};

/**
 * @enum DataType
 * @brief Enumeration of supported data types for tensors
 *
 * Defines the numeric data types that can be used for tensor elements.
 * Different data types offer trade-offs between precision, memory usage,
 * and computational speed.
 */
enum class DataType : uint8_t {
  kDataTypeUnknown = 0,  ///< Unknown or uninitialized data type
  kDataTypeFP32 = 1,     ///< 32-bit floating point (single precision)
  kDataTypeInt8 = 2,     ///< 8-bit signed integer (often for quantization)
  kDataTypeInt32 = 3,    ///< 32-bit signed integer
};

/**
 * @brief Returns the size in bytes of a given data type
 *
 * This function provides the memory footprint of each data type,
 * which is essential for memory allocation and stride calculations.
 *
 * @param dataType The data type to query
 * @return Size in bytes of the data type (4 for FP32, 1 for Int8, 4 for Int32)
 * @return 0 if the data type is unknown or invalid
 *
 * @note Logs an error message if an unknown data type is provided
 */
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

/**
 * @enum ModelType
 * @brief Enumeration of supported model architectures
 *
 * Identifies the specific neural network architecture being used.
 * Different model types have different layer structures, attention
 * mechanisms, and parameter configurations.
 */
enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,  ///< Unknown or uninitialized model type
  kModelTypeLLaMA2 = 1,   ///< LLaMA 2 transformer language model
  kModelTypeQWen = 2      ///< QWen (Qwen) language model
};

/**
 * @enum TokenizerType
 * @brief Enumeration of supported tokenizer types
 *
 * Specifies the tokenization algorithm used to convert text to token IDs
 * and vice versa. Different tokenizers have different vocabularies and
 * encoding strategies.
 */
enum class TokenizerType {
  kEncodeUnknown = -1,  ///< Unknown or uninitialized tokenizer
  kEncodeSpe = 0,       ///< SentencePiece tokenizer (used by LLaMA)
  kEncodeBpe = 1,       ///< Byte-Pair Encoding (BPE) tokenizer
};

/**
 * @class NoCopyable
 * @brief Base class to disable copy operations
 *
 * Classes that inherit from NoCopyable cannot be copied or copy-assigned.
 * This is useful for classes that manage resources (like file handles,
 * GPU memory, or unique ownership semantics) where copying would be
 * dangerous or semantically incorrect.
 *
 * Usage:
 * @code
 * class MyResource : public NoCopyable {
 *   // This class cannot be copied
 * };
 * @endcode
 *
 * @note Move operations are still allowed unless explicitly deleted
 */
class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;             ///< Deleted copy constructor
  NoCopyable& operator=(const NoCopyable&) = delete;  ///< Deleted copy assignment operator
};

/**
 * @enum StatusCode
 * @brief Error codes for operation status reporting
 *
 * Defines standard error codes used throughout the framework to indicate
 * success or various failure conditions. These codes are wrapped in the
 * Status class for error handling.
 *
 * @see Status Status wrapper class for error handling
 */
enum StatusCode : uint8_t {
  kSuccess = 0,               ///< Operation completed successfully
  kFunctionUnimplemented = 1, ///< Function not yet implemented
  kPathNotValid = 2,          ///< File or directory path is invalid
  kModelParseError = 3,       ///< Error parsing model file
  kInternalError = 5,         ///< Internal framework error
  kKeyValueHasExist = 6,      ///< Key already exists in map/dictionary
  kInvalidArgument = 7,       ///< Invalid argument provided to function
};

/**
 * @class Status
 * @brief Status and error handling class
 *
 * This class encapsulates operation results, providing both an error code
 * and an optional error message. It supports implicit conversions and
 * comparison operators for convenient error checking.
 *
 * The Status class is used throughout the framework for error propagation
 * and handling, following a pattern similar to other modern C++ frameworks.
 *
 * Usage patterns:
 * @code
 * // Return status from function
 * Status loadModel() {
 *   if (error) return Status(StatusCode::kModelParseError, "Failed to load");
 *   return Status(StatusCode::kSuccess);
 * }
 *
 * // Check status
 * Status s = someOperation();
 * if (!s) {  // Implicit conversion to bool
 *   LOG(ERROR) << "Operation failed: " << s.getErrMsg();
 * }
 *
 * // Compare with status code
 * if (s == StatusCode::kSuccess) {
 *   // success
 * }
 * @endcode
 *
 * @see StatusCode Error code enumeration
 * @see error Namespace containing status factory functions
 */
class Status {
 public:
  /**
   * @brief Constructs a Status with error code and optional message
   *
   * @param code Error code (defaults to kSuccess)
   * @param errMsg Optional error message describing the error
   */
  Status(int code = StatusCode::kSuccess, std::string errMsg = "");

  /**
   * @brief Copy constructor (default)
   */
  Status(const Status& other) = default;

  /**
   * @brief Copy assignment operator (default)
   */
  Status& operator=(const Status& other) = default;

  /**
   * @brief Assigns a new error code
   *
   * @param code New error code
   * @return Reference to this Status
   */
  Status& operator=(int code);

  /**
   * @brief Compares status with an error code
   *
   * @param code Error code to compare against
   * @return true if status codes match, false otherwise
   */
  bool operator==(int code) const;

  /**
   * @brief Compares status with an error code (inequality)
   *
   * @param code Error code to compare against
   * @return true if status codes differ, false otherwise
   */
  bool operator!=(int code) const;

  /**
   * @brief Implicit conversion to error code integer
   *
   * @return Error code as integer
   */
  operator int() const;

  /**
   * @brief Implicit conversion to boolean for success checking
   *
   * @return true if status is kSuccess, false otherwise
   *
   * Usage:
   * @code
   * Status s = someOperation();
   * if (!s) {  // Check for error
   *   // handle error
   * }
   * @endcode
   */
  operator bool() const;

  /**
   * @brief Gets the error code
   *
   * @return Error code as int32_t
   */
  int32_t getErrCode() const;

  /**
   * @brief Gets the error message
   *
   * @return Const reference to error message string
   */
  const std::string& getErrMsg() const;

  /**
   * @brief Sets a new error message
   *
   * @param errMsg New error message
   */
  void setErrMsg(const std::string& errMsg);

 private:
  int code_ = StatusCode::kSuccess;  ///< Error code
  std::string msg_;                  ///< Error message
};

/**
 * @namespace error
 * @brief Namespace containing status factory functions and error checking macros
 *
 * This namespace provides convenient factory functions for creating Status
 * objects with specific error codes, as well as macros for status checking
 * and error reporting.
 */
namespace error {
/**
 * @def STATUS_CHECK
 * @brief Macro to check status and log fatal error on failure
 *
 * This macro checks if a Status-returning function call succeeded.
 * If the status indicates failure (non-success), it logs a fatal error
 * with file, line, error code, and error message information.
 *
 * Usage:
 * @code
 * STATUS_CHECK(model.init(DeviceType::kDeviceCUDA));
 * STATUS_CHECK(allocateBuffers());
 * @endcode
 *
 * @param call Function call that returns a base::Status
 *
 * @warning This macro calls LOG(FATAL) on failure, which terminates the program
 */
#define STATUS_CHECK(call)                                                  \
  do {                                                                      \
    const base::Status& status = call;                                      \
    if (!status) {                                                          \
      const size_t buf_size = 512;                                          \
      char buf[buf_size];                                                   \
      snprintf(                                                             \
          buf, buf_size - 1,                                                \
          "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
          __FILE__, __LINE__, int(status), status.getErrMsg().c_str());     \
      LOG(FATAL) << buf;                                                    \
    }                                                                       \
  } while (0)

/**
 * @brief Creates a success status
 *
 * @param errMsg Optional message (typically empty for success)
 * @return Status with kSuccess code
 */
Status Success(const std::string& errMsg = "");

/**
 * @brief Creates a "function not implemented" error status
 *
 * Used for functions that are declared but not yet implemented.
 *
 * @param errMsg Optional error message describing what's not implemented
 * @return Status with kFunctionUnimplemented code
 */
Status FunctionNotImplement(const std::string& errMsg = "");

/**
 * @brief Creates an "invalid path" error status
 *
 * Used when a file or directory path is invalid or doesn't exist.
 *
 * @param errMsg Optional error message with path details
 * @return Status with kPathNotValid code
 */
Status PathNotValid(const std::string& errMsg = "");

/**
 * @brief Creates a "model parse error" status
 *
 * Used when model file parsing or loading fails.
 *
 * @param errMsg Optional error message describing parse failure
 * @return Status with kModelParseError code
 */
Status ModelParseError(const std::string& errMsg = "");

/**
 * @brief Creates an "internal error" status
 *
 * Used for unexpected internal framework errors.
 *
 * @param errMsg Optional error message describing the internal error
 * @return Status with kInternalError code
 */
Status InternalError(const std::string& errMsg = "");

/**
 * @brief Creates a "key already exists" error status
 *
 * Used when attempting to insert a duplicate key in a map or dictionary.
 *
 * @param errMsg Optional error message with key information
 * @return Status with kKeyValueHasExist code
 */
Status KeyHasExits(const std::string& errMsg = "");

/**
 * @brief Creates an "invalid argument" error status
 *
 * Used when a function receives an invalid argument value.
 *
 * @param errMsg Optional error message describing the invalid argument
 * @return Status with kInvalidArgument code
 */
Status InvalidArgument(const std::string& errMsg = "");

}  // namespace error

/**
 * @brief Stream output operator for Status
 *
 * Allows Status objects to be printed to output streams.
 * Outputs the error code and message in a readable format.
 *
 * Usage:
 * @code
 * Status s = someOperation();
 * std::cout << "Operation result: " << s << std::endl;
 * @endcode
 *
 * @param os Output stream
 * @param status Status object to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const Status& status);

}  // namespace base

#endif  // INFER_INCLUDE_BASE_BASE_H
