#ifndef INFER_INCLUDE_MODEL_MODEL_H_
#define INFER_INCLUDE_MODEL_MODEL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "sampler/sampler.h"

namespace model {

class Model {
 public:
  explicit Model(base::TokenizerType tokenizerType, base::ModelType modelType,
                 std::string tokenPath, std::string modelPath,
                 bool isQuantModel);

 public:
  virtual base::Status init(base::DeviceType deviceType) = 0;

  virtual base::Status predict(const tensor::Tensor& input,
                               const tensor::Tensor& posTensor, bool isPrompt,
                               int& next) const = 0;

  virtual base::Status forward(const tensor::Tensor& input,
                               const tensor::Tensor& posTensor,
                               int& next) const = 0;

  virtual op::EmbeddingOutput embedding(
      const std::vector<int>& tokens) const = 0;

 private:
  virtual void initMem() = 0;

  virtual base::Status createLayers() = 0;

  virtual void createParamLayers() = 0;

  virtual void createNonParamLayers() = 0;

  virtual void createParamQuantLayers() = 0;

 public:
  inline base::ModelType modelType() const { return modelType_; }

  inline const std::string& tokenPath() const { return tokenPath_; }

  inline const std::string& modelPath() const { return modelPath_; }

  inline virtual tensor::Tensor& getBuffer(ModelBufferType bufferIdx) {
    CHECK_GT(buffers_.count(bufferIdx), 0) << int(bufferIdx);
    return buffers_.at(bufferIdx);
  }

  inline virtual const tensor::Tensor& getBuffer(
      ModelBufferType bufferIdx) const {
    CHECK_GT(buffers_.count(bufferIdx), 0) << int(bufferIdx);
    return buffers_.at(bufferIdx);
  }

  inline virtual bool isSentenceEnding(int32_t token_idx) const {
    CHECK(encodeLayer_ != nullptr);
    return encodeLayer_->isSentenceEnding(token_idx);
  }

  inline virtual std::vector<int32_t> encode(
      const std::string& sentence) const {
    CHECK(encodeLayer_ != nullptr);
    return encodeLayer_->encode(sentence);
  }

  inline virtual std::string decode(int32_t tokenIdx) const {
    CHECK(this->encodeLayer_ != nullptr);
    return this->encodeLayer_->decode(tokenIdx);
  }

  inline virtual std::string decode(std::vector<int32_t> tokenIdxs) const {
    CHECK(this->encodeLayer_ != nullptr);
    return this->encodeLayer_->decode(tokenIdxs);
  }

 public:
  virtual std::pair<tensor::Tensor, tensor::Tensor> sliceKvCache(
      int32_t layerIdx, int32_t tokenPos) const;

  virtual tensor::Tensor fillInput(const tensor::Tensor& posTensor,
                                   const op::EmbeddingOutput& embeddingOutput,
                                   bool isPrompt) const;

 protected:
  virtual base::Status insertBuffer(ModelBufferType bufferIdx,
                                    const tensor::Tensor& tensor);

  virtual base::Status readModelFile();

  virtual base::Status createEncodeLayer();

  virtual base::Status genModelFromFile();

  virtual base::Status generateModelInfos(const ModelConfig& config) const;

  virtual int32_t postProcessing(const tensor::Tensor& pos,
                                 bool isPrompt) const = 0;

 protected:
  int32_t groupSize_ = 1;
  bool isQuantModel_ = false;
  std::unique_ptr<TransformerConfig> transformerConfig_;

  std::string tokenPath_;
  std::string modelPath_;

  std::unique_ptr<op::EncodeLayerBase> encodeLayer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> rawModelData_;
  base::DeviceType deviceType_ = base::DeviceType::kDeviceUnknown;
  base::ModelType modelType_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizerType_ = base::TokenizerType::kEncodeUnknown;
};

}  // namespace model

#endif  // INFER_INCLUDE_MODEL_MODEL_H_
