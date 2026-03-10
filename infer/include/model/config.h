#ifndef INFER_INCLUDE_MODEL_CONFIG_H_
#define INFER_INCLUDE_MODEL_CONFIG_H_

#include <cstdint>
namespace model {
struct ModelConfig {
  int32_t dim = 0;
  int32_t hiddenDim = 0;
  int32_t layerNum = 0;
  int32_t headNum = 0;
  int32_t kvHeadNum = 0;
  int32_t vocabSize = 0;
  int32_t seqLen = 0;
};

struct TransformerConfig {
  int32_t kvDim = 0;
  int32_t kvMul = 0;
  int32_t headSize = 0;
  int32_t vocabSize = 0;

  int32_t dim = 0;
  int32_t hiddenDim = 0;
  int32_t layerNum = 0;
  int32_t headNum = 0;
  int32_t kvHeadNum = 0;
  int32_t seqLen = 0;
  bool isSharedWeight = false;
};
}  // namespace model
#endif  // INFER_INCLUDE_MODEL_CONFIG_H_
