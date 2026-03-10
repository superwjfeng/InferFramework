#ifndef INFER_INCLUDE_MODEL_LLAMA3_H_
#define INFER_INCLUDE_MODEL_LLAMA3_H_

#include <memory>

#include "base/cuda_config.h"
#include "model/model.h"
#include "op/layer.h"
namespace model {
struct LLama2Layers {
  std::shared_ptr<op::Layer> addLayer;
  std::shared_ptr<op::Layer> ropeLayer;
  std::shared_ptr<op::Layer> swigluLayer;
  std::shared_ptr<op::Layer> mhaLayer;

  std::vector<std::shared_ptr<op::Layer>> wqLayers;
  std::vector<std::shared_ptr<op::Layer>> wkLayers;
  std::vector<std::shared_ptr<op::Layer>> wvLayers;
  std::vector<std::shared_ptr<op::Layer>> woLayers;

  std::vector<std::shared_ptr<op::Layer>> w1Layers;
  std::vector<std::shared_ptr<op::Layer>> w2Layers;
  std::vector<std::shared_ptr<op::Layer>> rmsnormLayers;
  std::vector<std::shared_ptr<op::Layer>> w3Layers;
  std::shared_ptr<op::Layer> clsLayer;

  std::shared_ptr<op::Layer> embeddingLayer;

  void toCuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizerType, std::string tokenPath,
                       std::string modelPath, bool isQuantModel);

  base::Status init(base::DeviceType deviceType) override;

  base::Status predict(const tensor::Tensor& input,
                       const tensor::Tensor& posTensor, bool isPrompt,
                       int& next) const override;

  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& posTensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  void initMem() override;

  base::Status createLayers() override;

  void createParamLayers() override;

  void createNonParamLayers() override;

  void createParamQuantLayers() override;

  void attentionMha(int32_t layer_idx, const tensor::Tensor& posTensor) const;

  void attentionRms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feedForward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attentionQkv(int32_t layer_idx, const tensor::Tensor& posTensor) const;

  void clsLogits(const tensor::Tensor& input) const;

  int32_t postProcessing(const tensor::Tensor& pos,
                         bool isPrompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cudaConfig_;
  std::unique_ptr<LLama2Layers> llamaLayers_;
};
}  // namespace model

#endif  // INFER_INCLUDE_MODEL_LLAMA3_H_
