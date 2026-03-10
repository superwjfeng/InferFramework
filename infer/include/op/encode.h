#ifndef INFER_INCLUDE_OP_ENCODE_H
#define INFER_INCLUDE_OP_ENCODE_H
#include <cstdint>
#include <string>
#include <vector>

#include <sentencepiece_processor.h>
#include "layer.h"

namespace op {
class EncodeLayerBase : public Layer {
 public:
  explicit EncodeLayerBase(std::string tokenModelPath, bool hasBos, bool hasEos)
      : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
        hasBos_(hasBos),
        hasEos_(hasEos),
        tokenModelPath_(std::move(tokenModelPath)) {}

 public:
  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  virtual std::string decode(int32_t tokenId) const = 0;

  virtual std::string decode(const std::vector<int32_t>& tokenIds) const = 0;

  virtual bool isSentenceEnding(int32_t tokenId) const = 0;

  virtual int32_t vocabSize() const = 0;

 protected:
  bool hasBos_ = true;
  bool hasEos_ = false;
  std::string tokenModelPath_;
};

class SpeEncodeLayer : public EncodeLayerBase {
 public:
  explicit SpeEncodeLayer(std::string tokenModelPath, bool hasBos, bool hasEos);

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::string decode(int32_t tokenId) const override;

  std::string decode(const std::vector<int32_t>& tokenIds) const override;

  bool isSentenceEnding(int32_t tokenId) const override;

  int32_t vocabSize() const override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};


}  // namespace op

#endif  // INFER_INCLUDE_OP_ENCODE_H
