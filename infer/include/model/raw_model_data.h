#ifndef INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#define INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_

#include <cstddef>
#include <cstdint>
namespace model {
/**
 * @struct RawModelData
 * @brief Abstract base structure for managing raw model weight data from files
 *
 * This structure provides an interface for loading and accessing model weights
 * stored in binary files. It uses memory mapping for efficient file access and
 * supports different weight formats through derived classes. The structure maintains
 * file descriptors, memory pointers, and provides virtual methods for type-specific
 * weight access.
 *
 * Memory layout:
 * - data: Points to the beginning of the memory-mapped file
 * - weightData: Points to the start of actual weight data (may skip header)
 *
 * @note This is an abstract structure. Use derived classes like RawModelDataFp32
 *       or RawModelDataInt8 for specific weight formats.
 * @see RawModelDataFp32 For 32-bit floating-point weights
 * @see RawModelDataInt8 For 8-bit quantized integer weights
 */
struct RawModelData {
  /**
   * @brief Destructor that cleans up file descriptors and memory mappings
   *
   * Releases system resources including closing file descriptors and
   * unmapping memory-mapped regions.
   */
  ~RawModelData();

  int32_t fd = -1;         ///< File descriptor for the model file (-1 if not opened)
  size_t fileSize = 0;     ///< Total size of the model file in bytes
  void* data = nullptr;    ///< Pointer to memory-mapped file data (includes header)
  void* weightData = nullptr;  ///< Pointer to the start of weight data in memory

  /**
   * @brief Retrieves a pointer to weight data at a specific offset
   *
   * This pure virtual method must be implemented by derived classes to provide
   * type-specific weight access. The method calculates the memory address of
   * weights at the given offset, handling type-specific size calculations.
   *
   * @param offset Offset in the weight array (in units of the weight type)
   * @return Const pointer to the weight data at the specified offset
   *
   * @note The offset is measured in units of the specific weight type
   *       (e.g., for FP32, offset 1 means 4 bytes ahead)
   */
  virtual const void* weight(size_t offset) const = 0;
};

/**
 * @struct RawModelDataFp32
 * @brief Concrete implementation for 32-bit floating-point model weights
 *
 * This structure handles model weights stored in 32-bit floating-point (FP32) format.
 * It provides the standard precision format commonly used in neural networks,
 * offering full precision without quantization.
 *
 * Memory layout:
 * - Each weight is stored as a 4-byte float (IEEE 754 single precision)
 * - Weight offset is calculated as: base_address + (offset * sizeof(float))
 *
 * Typical use case:
 * - Non-quantized models
 * - Models requiring high precision
 * - Models where inference speed is less critical than accuracy
 *
 * @see RawModelData Base structure for model data management
 */
struct RawModelDataFp32 : RawModelData {
  /**
   * @brief Retrieves a pointer to FP32 weight at the specified offset
   *
   * Calculates the memory address of a float weight at the given offset
   * and returns a pointer to it. The offset is in units of float elements.
   *
   * @param offset Offset in the weight array (number of float elements)
   * @return Const pointer to the float weight at the specified offset
   *
   * @note The returned pointer points to float data (4 bytes per element)
   */
  const void* weight(size_t offset) const override;
};

/**
 * @struct RawModelDataInt8
 * @brief Concrete implementation for 8-bit quantized integer model weights
 *
 * This structure handles model weights stored in 8-bit integer (INT8) format.
 * Quantization reduces memory footprint and can accelerate inference on
 * hardware with INT8 support, at the cost of slight precision loss.
 *
 * Memory layout:
 * - Each weight is stored as a 1-byte signed integer
 * - Weight offset is calculated as: base_address + (offset * sizeof(int8_t))
 * - May include additional quantization parameters (scale, zero-point)
 *
 * Typical use case:
 * - Memory-constrained environments
 * - Models optimized for edge devices
 * - Inference where speed is prioritized over precision
 * - Models with hardware INT8 acceleration support
 *
 * @see RawModelData Base structure for model data management
 */
struct RawModelDataInt8 : RawModelData {
  /**
   * @brief Retrieves a pointer to INT8 weight at the specified offset
   *
   * Calculates the memory address of an int8_t weight at the given offset
   * and returns a pointer to it. The offset is in units of int8_t elements.
   *
   * @param offset Offset in the weight array (number of int8_t elements)
   * @return Const pointer to the int8_t weight at the specified offset
   *
   * @note The returned pointer points to int8_t data (1 byte per element)
   * @note Actual floating-point value requires dequantization using scale/zero-point
   */
  const void* weight(size_t offset) const override;
};

}  // namespace model
#endif  // INFER_INCLUDE_MODEL_RAW_MODEL_DATA_H_
