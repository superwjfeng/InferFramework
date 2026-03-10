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

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

class BaseLayer {
 public:
  explicit BaseLayer(base::DeviceType deviceType, LayerType layerType,
                     base::DataType dataType, std::string layerName = "")
      : deviceType_(deviceType),
        layerType_(layerType),
        dataType_(dataType),
        layerName_(std::move(layerName)) {}

 public:
  inline base::DataType dataType() const { return dataType_; }

  inline LayerType layerType() const { return layerType_; }

  inline const std::string& getLayerName() const { return layerName_; }

  inline void setLayerName(const std::string& layerName) {
    layerName_ = layerName;
  }

  inline base::DeviceType deviceType() const { return deviceType_; }

  inline void setDeviceType(base::DeviceType deviceType) {
    deviceType_ = deviceType;
  }

 public:
  virtual base::Status init() = 0;

  virtual base::Status forward() = 0;

  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1,
                               const tensor::Tensor& input2,
                               const tensor::Tensor& input3,
                               const tensor::Tensor& input4,
                               const tensor::Tensor& input5,
                               const tensor::Tensor& output1) = 0;

  virtual void setInput(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void setOutput(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t inputSize() const = 0;

  virtual size_t outputSize() const = 0;

  virtual base::Status check() const = 0;

  virtual tensor::Tensor& getInput(int32_t idx) = 0;

  virtual tensor::Tensor& getOutput(int32_t idx) = 0;

  virtual const tensor::Tensor& getInput(int32_t idx) const = 0;

  virtual const tensor::Tensor& getOutput(int32_t idx) const = 0;

  virtual base::Status setWeight(int32_t idx, const tensor::Tensor& weight) {
    return base::error::FunctionNotImplement();
  }

  virtual base::Status setWeight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::kDeviceUnknown) {
    return base::error::FunctionNotImplement();
  }

 protected:
  std::string layerName_;
  LayerType layerType_ = LayerType::kLayerUnknown;
  base::DataType dataType_ = base::DataType::kDataTypeUnknown;
  base::DeviceType deviceType_ = base::DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType deviceType, LayerType layerType,
                 std::string layerName = "")
      : BaseLayer(deviceType, layerType, base::DataType::kDataTypeUnknown,
                  std::move(layerName)) {}

  inline base::Status init() override { return base::error::Success(); }

  inline base::Status check() const override {
    return base::error::FunctionNotImplement(
        "The check function is not implement yet");
  }

  base::Status checkTensor(const tensor::Tensor& tensor,
                           base::DeviceType deviceType,
                           base::DataType dataType) const;

  base::Status checkTensorWithDim(const tensor::Tensor& tensor,
                                  base::DeviceType deviceType,
                                  base::DataType dataType, ...) const;

  inline base::Status forward() override {
    return base::error::FunctionNotImplement("");
  }

  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       const tensor::Tensor& input3,
                       const tensor::Tensor& input4,
                       const tensor::Tensor& input5,
                       const tensor::Tensor& output1) override;

  inline void setInput(int32_t idx, const tensor::Tensor& input) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
  }

  inline void setOutput(int32_t idx, const tensor::Tensor& output) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
  }

  inline const tensor::Tensor& getInput(int32_t idx) const override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
  }

  inline const tensor::Tensor& getOutput(int32_t idx) const override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
  }

  inline tensor::Tensor& getInput(int32_t idx) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
  }

  inline tensor::Tensor& getOutput(int32_t idx) override {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
  }

  inline size_t inputSize() const override { return inputs_.size(); }

  inline size_t outputSize() const override { return outputs_.size(); }

  inline void resetInputSize(size_t size) { inputs_.resize(size); }

  inline void resetOutputSize(size_t size) { outputs_.resize(size); }

  virtual void toCuda();

  inline void setCudaConfig(std::shared_ptr<kernel::CudaConfig> cudaConfig) {
    if (!cudaConfig) {
      LOG(ERROR) << "CudaConfig is null.";
      return;
    }
    cudaConfig_ = cudaConfig;
  }

  inline std::shared_ptr<kernel::CudaConfig> cudaConfig() const {
    return cudaConfig_;
  }

 private:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  std::shared_ptr<kernel::CudaConfig> cudaConfig_;
};

class LayerParam : public Layer {
 public:
  explicit LayerParam(base::DeviceType deviceType, LayerType layerType,
                      bool isQuantLayer = false, std::string layerName = "")
      : Layer(deviceType, layerType, std::move(layerName)),
        isQuantLayer_(isQuantLayer) {}

  inline size_t weightSize() const { return weights_.size(); }

  inline void resetWeightSize(size_t size) { return weights_.resize(size); }

  inline tensor::Tensor& getWeight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
  }

  inline const tensor::Tensor& getWeight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
  }

  void toCuda() override;

  base::Status setWeight(int32_t idx, const tensor::Tensor& weight) override;

  base::Status setWeight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weightPtr,
      base::DeviceType deviceType = base::DeviceType::kDeviceUnknown) override;

  inline void setScales(const tensor::Tensor& scales) {
    CHECK(!scales.empty());
    scales_ = scales;
  }

  inline void setGroupSize(int32_t groupSize) { groupSize_ = groupSize; }

  inline int32_t getScaleNum() const { return scales_.size(); }

 protected:
  int32_t groupSize_ = 0;
  bool isQuantLayer_ = false;
  tensor::Tensor scales_;
  std::vector<tensor::Tensor> weights_;
};

}  // namespace op

#endif  // INFER_SRC_OP_LAYER_H_
