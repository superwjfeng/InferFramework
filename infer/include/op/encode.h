/**
 * @file encode.h
 * @brief Text encoding and decoding layer implementations
 *
 * This file defines the encoding layer interface and implementations for
 * converting text to token IDs (encoding) and token IDs back to text (decoding).
 * Used for tokenization in natural language processing pipelines.
 */

#ifndef INFER_INCLUDE_OP_ENCODE_H
#define INFER_INCLUDE_OP_ENCODE_H
#include <cstdint>
#include <string>
#include <vector>

#include <sentencepiece_processor.h>
#include "layer.h"

/**
 * @namespace op
 * @brief Namespace containing neural network operation implementations
 */
namespace op {

/**
 * @class EncodeLayerBase
 * @brief Abstract base class for text encoding and decoding operations
 *
 * Provides the interface for tokenization (text to token IDs) and
 * detokenization (token IDs to text). Subclasses implement specific
 * tokenization algorithms.
 */
class EncodeLayerBase : public Layer {
 public:
  /**
   * @brief Constructs an encoding layer
   * @param tokenModelPath Path to the tokenizer model file
   * @param hasBos Whether to add Beginning-of-Sequence token
   * @param hasEos Whether to add End-of-Sequence token
   */
  explicit EncodeLayerBase(std::string tokenModelPath, bool hasBos, bool hasEos)
      : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
        hasBos_(hasBos),
        hasEos_(hasEos),
        tokenModelPath_(std::move(tokenModelPath)) {}

 public:
  /**
   * @brief Encodes a text sentence into token IDs
   * @param sentence Input text to encode
   * @return Vector of token IDs representing the sentence
   */
  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  /**
   * @brief Decodes a single token ID to its text representation
   * @param tokenId Token ID to decode
   * @return Text representation of the token
   */
  virtual std::string decode(int32_t tokenId) const = 0;

  /**
   * @brief Decodes a sequence of token IDs to text
   * @param tokenIds Vector of token IDs to decode
   * @return Decoded text string
   */
  virtual std::string decode(const std::vector<int32_t>& tokenIds) const = 0;

  /**
   * @brief Checks if a token ID represents a sentence ending
   * @param tokenId Token ID to check
   * @return true if the token ends a sentence, false otherwise
   */
  virtual bool isSentenceEnding(int32_t tokenId) const = 0;

  /**
   * @brief Gets the vocabulary size
   * @return Number of tokens in the vocabulary
   */
  virtual int32_t vocabSize() const = 0;

 protected:
  /** @brief Whether to add Beginning-of-Sequence token */
  bool hasBos_ = true;

  /** @brief Whether to add End-of-Sequence token */
  bool hasEos_ = false;

  /** @brief Path to the tokenizer model file */
  std::string tokenModelPath_;
};

/**
 * @class SpeEncodeLayer
 * @brief SentencePiece-based encoding layer implementation
 *
 * Implements text encoding and decoding using the SentencePiece library,
 * which provides efficient subword tokenization for neural language models.
 */
class SpeEncodeLayer : public EncodeLayerBase {
 public:
  /**
   * @brief Constructs a SentencePiece encoding layer
   * @param tokenModelPath Path to the SentencePiece model file
   * @param hasBos Whether to add Beginning-of-Sequence token
   * @param hasEos Whether to add End-of-Sequence token
   */
  explicit SpeEncodeLayer(std::string tokenModelPath, bool hasBos, bool hasEos);

  /**
   * @brief Encodes text using SentencePiece tokenization
   * @param sentence Input text to encode
   * @return Vector of token IDs
   */
  std::vector<int32_t> encode(const std::string& sentence) const override;

  /**
   * @brief Decodes a single token ID using SentencePiece
   * @param tokenId Token ID to decode
   * @return Text representation of the token
   */
  std::string decode(int32_t tokenId) const override;

  /**
   * @brief Decodes token IDs to text using SentencePiece
   * @param tokenIds Vector of token IDs to decode
   * @return Decoded text string
   */
  std::string decode(const std::vector<int32_t>& tokenIds) const override;

  /**
   * @brief Checks if a token represents sentence ending
   * @param tokenId Token ID to check
   * @return true if the token is an end-of-sentence marker
   */
  bool isSentenceEnding(int32_t tokenId) const override;

  /**
   * @brief Gets the SentencePiece vocabulary size
   * @return Number of tokens in the vocabulary
   */
  int32_t vocabSize() const override;

 private:
  /** @brief SentencePiece processor for tokenization */
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};


}  // namespace op

#endif  // INFER_INCLUDE_OP_ENCODE_H
