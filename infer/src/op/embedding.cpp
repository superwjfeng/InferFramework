#include "op/embedding.h"

#include "base/base.h"
#include "op/layer.h"

namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType deviceType, int32_t dim,
                               int32_t seqLen, int32_t vocabSize)
    : dim_(dim),
      seqLen_(seqLen),
      vocabSize_(vocabSize),
      LayerParam(deviceType, LayerType::kLayerEmbedding, false,
                 "Embedding Layer") {
  resetWeightSize(1);
  resetInputSize(2);
  resetOutputSize(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& inputTensor = getInput(0);
  const auto& tokenSize = getInput(1).size();

  return base::error::Success();
}

base::Status EmbeddingLayer::forward() { return base::error::Success(); }
}  // namespace op
