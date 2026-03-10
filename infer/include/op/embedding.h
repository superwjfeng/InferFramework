#ifndef INFER_INCLUDE_OP_EMBEDDING_H_
#define INFER_INCLUDE_OP_EMBEDDING_H_

#include <cstdint>
#include <vector>

#include "base/base.h"
#include "op/layer.h"
#include "tensor/tensor.h"

namespace op {

struct EmbeddingOutput {
  tensor::Tensor inputTokens;
  tensor::Tensor inputEmbeddings;
  tensor::Tensor inputTokenNum;

  explicit EmbeddingOutput(tensor::Tensor inputTokens,
                           tensor::Tensor inputEmbeddings,
                           tensor::Tensor inputTokenNum)
      : inputTokens(std::move(inputTokens)),
        inputEmbeddings(std::move(inputEmbeddings)),
        inputTokenNum(std::move(inputTokenNum)) {}
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType deviceType, int32_t dim,
                          int32_t seqLen, int32_t vocabSize);
  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seqLen_ = 0;
  int32_t vocabSize_ = 0;
};

}  // namespace op

#endif  // INFER_INCLUDE_OP_EMBEDDING_H_
