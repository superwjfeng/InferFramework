#ifndef INFER_SRC_OP_LAYER_H_
#define INFER_SRC_OP_LAYER_H_
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace op {
class Layer;

/**
 * @enum LayerType
 * @brief Enumeration of neural network layer types
 *
 * This enumeration defines all supported layer types in the inference framework.
 * Each layer type corresponds to a specific neural network operation.
 */
enum class LayerType : uint8_t {
  kLayerUnknown = 0,    ///< Unknown or uninitialized layer type
  kLayerLinear = 1,     ///< Linear/fully-connected layer (matrix multiplication + bias)
  kLayerEncode = 2,     ///< Encoding layer for tokenization
  kLayerEmbedding = 3,  ///< Token embedding layer
  kLayerRMSNorm = 4,    ///< Root Mean Square Layer Normalization
  kLayerMatmul = 5,     ///< Matrix multiplication layer (without bias)
  kLayerRoPe = 6,       ///< Rotary Position Embedding layer
  kLayerMHA = 7,        ///< Multi-Head Attention layer
  kLayerSoftmax = 8,    ///< Softmax activation layer
  kLayerAdd = 9,        ///< Element-wise addition layer
  kLayerSwiGLU = 10,    ///< SwiGLU activation function layer
};

/**
 * @class BaseLayer
 * @brief Abstract base class for all neural network layers
 *
 * This class provides the fundamental interface that all layer implementations
 * must follow. It defines the core operations including initialization, forward
 * propagation, and tensor management. The class supports:
 * - Multiple input/output configurations (1-5 inputs)
 * - Device-agnostic operations (CPU, CUDA)
 * - Various data types (FP32, INT8, etc.)
 * - Optional weight management for parameterized layers
 *
 * @note This is an abstract class and cannot be instantiated directly.
 *       Use derived classes like Layer or LayerParam for concrete implementations.
 * @see Layer Non-parameterized layer implementation
 * @see LayerParam Parameterized layer implementation with weights
 */
class BaseLayer {
 public:
  /**
   * @brief Constructs a BaseLayer with specified configuration
   *
   * @param deviceType Device to run the layer on (CPU, CUDA, etc.)
   * @param layerType Type of layer operation (Linear, RMSNorm, etc.)
   * @param dataType Data type for computations (FP32, INT8, etc.)
   * @param layerName Optional human-readable name for debugging
   */
  explicit BaseLayer(base::DeviceType deviceType, LayerType layerType,
                     base::DataType dataType, std::string layerName = "")
      : deviceType_(deviceType),
        layerType_(layerType),
        dataType_(dataType),
        layerName_(std::move(layerName)) {}

 public:
  /**
   * @brief Gets the data type used by this layer
   * @return Data type (FP32, INT8, etc.)
   */
  inline base::DataType dataType() const { return dataType_; }

  /**
   * @brief Gets the type of this layer
   * @return Layer type enumeration value
   */
  inline LayerType layerType() const { return layerType_; }

  /**
   * @brief Gets the name of this layer
   * @return Layer name string
   */
  inline const std::string& getLayerName() const { return layerName_; }

  /**
   * @brief Sets the name of this layer
   * @param layerName New name for the layer
   */
  inline void setLayerName(const std::string& layerName) {
    layerName_ = layerName;
  }

  /**
   * @brief Gets the device type where this layer executes
   * @return Device type (CPU, CUDA, etc.)
   */
  inline base::DeviceType deviceType() const { return deviceType_; }

  /**
   * @brief Sets the device type for this layer
   * @param deviceType New device type
   */
  inline void setDeviceType(base::DeviceType deviceType) {
    deviceType_ = deviceType;
  }

 public:
  /**
   * @brief Initializes the layer
   *
   * Performs layer-specific initialization such as allocating buffers,
   * preparing kernels, or validating configuration.
   *
   * @return Status indicating success or failure
   */
  virtual base::Status init() = 0;

  /**
   * @brief Performs forward propagation with no explicit inputs/outputs
   *
   * This variant uses inputs/outputs set via setInput/setOutput methods.
   * Useful for layers with complex or variable numbers of inputs.
   *
   * @return Status indicating success or failure
   */
  virtual base::Status forward() = 0;

  /**
   * @brief Performs forward propagation with one input and one output
   *
   * @param input1 Input tensor
   * @param output1 Output tensor where result is written
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& output1) = 0;

  /**
   * @brief Performs forward propagation with two inputs and one output
   *
   * Used by layers like Add, Matmul, etc. that require two input tensors.
   *
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param output1 Output tensor where result is written
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  /**
   * @brief Performs forward propagation with three inputs and one output
   *
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param output1 Output tensor where result is written
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& output1) = 0;

  /**
   * @brief Performs forward propagation with four inputs and one output
   *
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param input4 Fourth input tensor
   * @param output1 Output tensor where result is written
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  /**
   * @brief Performs forward propagation with five inputs and one output
   *
   * Used by complex layers like Multi-Head Attention that require
   * multiple input tensors (Q, K, V, cache, mask, etc.).
   *
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param input4 Fourth input tensor
   * @param input5 Fifth input tensor
   * @param output1 Output tensor where result is written
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& input4,
                               const tensor::Tensor& input5,
                               const tensor::Tensor& output1) = 0;

  /**
   * @brief Sets an input tensor at the specified index
   *
   * @param idx Index of the input slot (0-based)
   * @param input Tensor to set as input
   */
  virtual void setInput(int32_t idx, const tensor::Tensor& input) = 0;

  /**
   * @brief Sets an output tensor at the specified index
   *
   * @param idx Index of the output slot (0-based)
   * @param output Tensor to set as output
   */
  virtual void setOutput(int32_t idx, const tensor::Tensor& output) = 0;

  /**
   * @brief Gets the number of input tensors
   * @return Number of inputs
   */
  virtual size_t inputSize() const = 0;

  /**
   * @brief Gets the number of output tensors
   * @return Number of outputs
   */
  virtual size_t outputSize() const = 0;

  /**
   * @brief Validates layer configuration and tensor dimensions
   *
   * Checks that all inputs, outputs, and weights have compatible shapes
   * and are properly initialized.
   *
   * @return Status indicating validity
   */
  virtual base::Status check() const = 0;

  /**
   * @brief Retrieves a mutable input tensor by index
   *
   * @param idx Index of the input tensor
   * @return Reference to the input tensor
   */
  virtual tensor::Tensor& getInput(int32_t idx) = 0;

  /**
   * @brief Retrieves a mutable output tensor by index
   *
   * @param idx Index of the output tensor
   * @return Reference to the output tensor
   */
  virtual tensor::Tensor& getOutput(int32_t idx) = 0;

  /**
   * @brief Retrieves a const input tensor by index
   *
   * @param idx Index of the input tensor
   * @return Const reference to the input tensor
   */
  virtual const tensor::Tensor& getInput(int32_t idx) const = 0;

  /**
   * @brief Retrieves a const output tensor by index
   *
   * @param idx Index of the output tensor
   * @return Const reference to the output tensor
   */
  virtual const tensor::Tensor& getOutput(int32_t idx) const = 0;

  /**
   * @brief Sets a weight tensor for parameterized layers
   *
   * This method is only applicable to layers with learnable parameters.
   * Returns FunctionNotImplement for non-parameterized layers.
   *
   * @param idx Index of the weight tensor
   * @param weight Weight tensor to set
   * @return Status indicating success or FunctionNotImplement
   */
  virtual base::Status setWeight(int32_t idx, const tensor::Tensor& weight) {
    return base::error::FunctionNotImplement();
  }

  /**
   * @brief Sets a weight tensor from raw pointer
   *
   * Creates a weight tensor from raw memory and sets it at the specified index.
   * Only applicable to parameterized layers.
   *
   * @param idx Index of the weight tensor
   * @param dims Dimensions of the weight tensor
   * @param weight_ptr Pointer to raw weight data
   * @param device_type Device where weight resides (CPU, CUDA, etc.)
   * @return Status indicating success or FunctionNotImplement
   */
  virtual base::Status setWeight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::kDeviceUnknown) {
    return base::error::FunctionNotImplement();
  }

 protected:
  std::string layerName_;       ///< Human-readable name for debugging
  LayerType layerType_ = LayerType::kLayerUnknown;  ///< Type of layer operation
  base::DataType dataType_ = base::DataType::kDataTypeUnknown;  ///< Data type for computations
  base::DeviceType deviceType_ = base::DeviceType::kDeviceUnknown;  ///< Device where layer executes
};

/**
 * @class Layer
 * @brief Concrete implementation of non-parameterized neural network layers
 *
 * This class provides a complete implementation of the BaseLayer interface for
 * layers that do not have learnable weights. Examples include activation functions,
 * normalization operations, and other stateless transformations. The class:
 * - Manages input and output tensor collections
 * - Provides default implementations for common operations
 * - Supports CUDA acceleration through CudaConfig
 * - Includes tensor validation utilities
 *
 * Use this class for layers like ReLU, Softmax, Add, etc. For layers with
 * learnable parameters, use LayerParam instead.
 *
 * @see BaseLayer Base interface for all layers
 * @see LayerParam For layers with learnable weights
 */
class Layer : public BaseLayer {
 public:
  /**
   * @brief Constructs a Layer instance
   *
   * @param deviceType Device to run the layer on (CPU, CUDA, etc.)
   * @param layerType Type of layer operation
   * @param layerName Optional name for debugging
   */
  explicit Layer(base::DeviceType deviceType, LayerType layerType,
                 std::string layerName = "")
      : BaseLayer(deviceType, layerType, base::DataType::kDataTypeUnknown,
                  std::move(layerName)) {}

  /**
   * @brief Default initialization (no-op for non-parameterized layers)
   * @return Success status
   */
  inline base::Status init() override { return base::error::Success(); }

  /**
   * @brief Default check implementation
   * @return FunctionNotImplement status (override in derived classes)
   */
  inline base::Status check() const override {
    return base::error::FunctionNotImplement(
        "The check function is not implement yet");
  }

  /**
   * @brief Validates a tensor's device type and data type
   *
   * @param tensor Tensor to validate
   * @param deviceType Expected device type
   * @param dataType Expected data type
   * @return Status indicating validity
   */
  base::Status checkTensor(const tensor::Tensor& tensor,
                           base::DeviceType deviceType,
                           base::DataType dataType) const;

  /**
   * @brief Validates a tensor's device, data type, and dimensions
   *
   * Uses variadic arguments to specify expected dimension values.
   *
   * @param tensor Tensor to validate
   * @param deviceType Expected device type
   * @param dataType Expected data type
   * @param ... Expected dimension values (variable number of arguments)
   * @return Status indicating validity
   */
  base::Status checkTensorWithDim(const tensor::Tensor& tensor,
                                  base::DeviceType deviceType,
                                  base::DataType dataType, ...) const;

  /**
   * @brief Default forward with no arguments (not implemented)
   * @return FunctionNotImplement status
   */
  inline base::Status forward() override {
    return base::error::FunctionNotImplement("");
  }

  /**
   * @brief Forward propagation with one input and one output
   * @param input1 Input tensor
   * @param output1 Output tensor
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& output1) override;

  /**
   * @brief Forward propagation with two inputs and one output
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param output1 Output tensor
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

  /**
   * @brief Forward propagation with three inputs and one output
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param output1 Output tensor
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& output1) override;

  /**
   * @brief Forward propagation with four inputs and one output
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param input4 Fourth input tensor
   * @param output1 Output tensor
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

  /**
   * @brief Forward propagation with five inputs and one output
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param input3 Third input tensor
   * @param input4 Fourth input tensor
   * @param input5 Fifth input tensor
   * @param output1 Output tensor
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& input4,
                       const tensor::Tensor& input5,
                       const tensor::Tensor& output1) override;

  /**
   * @brief Sets an input tensor at the specified index
   * @param idx Input slot index (must be within range)
   * @param input Tensor to set
   * @throws CHECK failure if index is out of range
   */
  inline void setInput(int32_t idx, const tensor::Tensor& input) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
  }

  /**
   * @brief Sets an output tensor at the specified index
   * @param idx Output slot index (must be within range)
   * @param output Tensor to set
   * @throws CHECK failure if index is out of range
   */
  inline void setOutput(int32_t idx, const tensor::Tensor& output) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
  }

  /**
   * @brief Gets a const input tensor by index
   * @param idx Input tensor index
   * @return Const reference to input tensor
   * @throws CHECK failure if index is out of range
   */
  inline const tensor::Tensor& getInput(int32_t idx) const override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
  }

  /**
   * @brief Gets a const output tensor by index
   * @param idx Output tensor index
   * @return Const reference to output tensor
   * @throws CHECK failure if index is out of range
   */
  inline const tensor::Tensor& getOutput(int32_t idx) const override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
  }

  /**
   * @brief Gets a mutable input tensor by index
   * @param idx Input tensor index
   * @return Reference to input tensor
   * @throws CHECK failure if index is out of range
   */
  inline tensor::Tensor& getInput(int32_t idx) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
  }

  /**
   * @brief Gets a mutable output tensor by index
   * @param idx Output tensor index
   * @return Reference to output tensor
   * @throws CHECK failure if index is out of range
   */
  inline tensor::Tensor& getOutput(int32_t idx) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
  }

  /**
   * @brief Gets the number of input tensors
   * @return Number of inputs
   */
  inline size_t inputSize() const override { return inputs_.size(); }

  /**
   * @brief Gets the number of output tensors
   * @return Number of outputs
   */
  inline size_t outputSize() const override { return outputs_.size(); }

  /**
   * @brief Resizes the input tensor collection
   * @param size New number of input slots
   */
  inline void resetInputSize(size_t size) { inputs_.resize(size); }

  /**
   * @brief Resizes the output tensor collection
   * @param size New number of output slots
   */
  inline void resetOutputSize(size_t size) { outputs_.resize(size); }

  /**
   * @brief Transfers layer data to CUDA device
   *
   * Moves all input and output tensors to GPU memory for CUDA execution.
   * Derived classes may override to transfer additional resources.
   */
  virtual void toCuda();

  /**
   * @brief Sets the CUDA configuration for GPU execution
   *
   * @param cudaConfig Shared pointer to CUDA configuration object
   */
  inline void setCudaConfig(std::shared_ptr<kernel::CudaConfig> cudaConfig) {
    if (!cudaConfig) {
      LOG(ERROR) << "CudaConfig is null.";
      return;
    }
    cudaConfig_ = cudaConfig;
  }

  /**
   * @brief Gets the CUDA configuration
   * @return Shared pointer to CUDA configuration, or nullptr if not set
   */
  inline std::shared_ptr<kernel::CudaConfig> cudaConfig() const {
    return cudaConfig_;
  }

 private:
  std::vector<tensor::Tensor> inputs_;   ///< Collection of input tensors
  std::vector<tensor::Tensor> outputs_;  ///< Collection of output tensors
  std::shared_ptr<kernel::CudaConfig> cudaConfig_;  ///< CUDA configuration for GPU execution
};

/**
 * @class LayerParam
 * @brief Layer implementation with learnable weights and optional quantization
 *
 * This class extends Layer to support parameterized layers that have learnable
 * weights. It provides:
 * - Weight tensor management (storage and access)
 * - Quantization support with scale factors and group size
 * - CUDA weight transfer capabilities
 * - Weight initialization from tensors or raw pointers
 *
 * Typical use cases include Linear layers, Embedding layers, and Convolutional
 * layers. The class supports both full-precision (FP32) and quantized (INT8)
 * weights with per-group scaling.
 *
 * @see Layer Base class for non-parameterized layers
 * @see BaseLayer Root interface for all layers
 */
class LayerParam : public Layer {
 public:
  /**
   * @brief Constructs a LayerParam instance
   *
   * @param deviceType Device to run the layer on (CPU, CUDA, etc.)
   * @param layerType Type of layer operation (Linear, Embedding, etc.)
   * @param isQuantLayer Flag indicating if this layer uses quantized weights
   * @param layerName Optional name for debugging
   */
  explicit LayerParam(base::DeviceType deviceType, LayerType layerType,
                      bool isQuantLayer = false, std::string layerName = "")
      : Layer(deviceType, layerType, std::move(layerName)),
        isQuantLayer_(isQuantLayer) {}

  /**
   * @brief Gets the number of weight tensors
   * @return Number of weights
   */
  inline size_t weightSize() const { return weights_.size(); }

  /**
   * @brief Resizes the weight tensor collection
   * @param size New number of weight slots
   */
  inline void resetWeightSize(size_t size) { return weights_.resize(size); }

  /**
   * @brief Gets a mutable weight tensor by index
   *
   * @param idx Weight tensor index
   * @return Reference to weight tensor
   * @throws CHECK failure if index is out of range
   */
  inline tensor::Tensor& getWeight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
  }

  /**
   * @brief Gets a const weight tensor by index
   *
   * @param idx Weight tensor index
   * @return Const reference to weight tensor
   * @throws CHECK failure if index is out of range
   */
  inline const tensor::Tensor& getWeight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
  }

  /**
   * @brief Transfers layer weights to CUDA device
   *
   * Moves all weight tensors, scale factors, and layer data to GPU memory
   * for CUDA execution. Overrides the base Layer::toCuda() method.
   */
  void toCuda() override;

  /**
   * @brief Sets a weight tensor at the specified index
   *
   * @param idx Weight slot index
   * @param weight Weight tensor to set
   * @return Status indicating success or failure
   */
  base::Status setWeight(int32_t idx, const tensor::Tensor& weight) override;

  /**
   * @brief Sets a weight tensor from raw memory
   *
   * Creates a weight tensor from raw pointer and assigns it to the specified slot.
   * Useful for loading weights from model files.
   *
   * @param idx Weight slot index
   * @param dims Dimensions of the weight tensor
   * @param weightPtr Pointer to raw weight data
   * @param deviceType Device where weight data resides
   * @return Status indicating success or failure
   */
  base::Status setWeight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weightPtr,
      base::DeviceType deviceType = base::DeviceType::kDeviceUnknown) override;

  /**
   * @brief Sets the scale factors for quantized weights
   *
   * For quantized layers, scale factors are used to dequantize INT8 weights
   * back to floating-point during computation. One scale per group.
   *
   * @param scales Tensor containing scale factors
   */
  inline void setScales(const tensor::Tensor& scales) {
    CHECK(!scales.empty());
    scales_ = scales;
  }

  /**
   * @brief Sets the group size for quantization
   *
   * Weight quantization is performed in groups. Each group has one scale factor.
   * Typical group sizes are 32, 64, or 128.
   *
   * @param groupSize Number of weights per quantization group
   */
  inline void setGroupSize(int32_t groupSize) { groupSize_ = groupSize; }

  /**
   * @brief Gets the number of scale factors
   *
   * @return Number of scale factors (equal to number of quantization groups)
   */
  inline int32_t getScaleNum() const { return scales_.size(); }

 protected:
  int32_t groupSize_ = 0;         ///< Number of weights per quantization group
  bool isQuantLayer_ = false;     ///< Flag indicating if weights are quantized
  tensor::Tensor scales_;         ///< Scale factors for dequantizing weights
  std::vector<tensor::Tensor> weights_;  ///< Collection of weight tensors
};

}  // namespace op

#endif  // INFER_SRC_OP_LAYER_H_
