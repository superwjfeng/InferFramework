/**
 * @file embedding.h
 * @brief Token embedding layer implementation
 *
 * This file defines the embedding layer that converts discrete token IDs
 * into continuous vector representations (embeddings) for neural network
 * processing.
 */

#ifndef INFER_INCLUDE_OP_EMBEDDING_H_
#define INFER_INCLUDE_OP_EMBEDDING_H_

#include <cstdint>
#include <vector>

#include "base/base.h"
#include "op/layer.h"
#include "tensor/tensor.h"

/**
 * @namespace op
 * @brief Namespace containing neural network operation implementations
 */
namespace op {

/**
 * @struct EmbeddingOutput
 * @brief Output structure for the embedding layer
 *
 * Contains the processed token information including the original token IDs,
 * their corresponding embeddings, and the number of tokens in the batch.
 */
struct EmbeddingOutput {
  /** @brief Input token IDs from the vocabulary */
  tensor::Tensor inputTokens;

  /** @brief Embedded representations of the input tokens */
  tensor::Tensor inputEmbeddings;

  /** @brief Number of tokens in the current batch */
  tensor::Tensor inputTokenNum;

  /**
   * @brief Constructs an EmbeddingOutput with the given tensors
   * @param inputTokens Token IDs from the vocabulary
   * @param inputEmbeddings Embedded vector representations
   * @param inputTokenNum Number of tokens in the batch
   */
  explicit EmbeddingOutput(tensor::Tensor inputTokens,
                           tensor::Tensor inputEmbeddings,
                           tensor::Tensor inputTokenNum)
      : inputTokens(std::move(inputTokens)),
        inputEmbeddings(std::move(inputEmbeddings)),
        inputTokenNum(std::move(inputTokenNum)) {}
};

/**
 * @class EmbeddingLayer
 * @brief Neural network embedding layer
 *
 * Converts discrete token IDs into continuous vector representations.
 * This is typically the first layer in language models, mapping vocabulary
 * indices to learned embeddings.
 */
class EmbeddingLayer : public LayerParam {
 public:
  /**
   * @brief Constructs an embedding layer
   * @param deviceType Device type for computation (CPU/CUDA)
   * @param dim Embedding dimension (size of each embedding vector)
   * @param seqLen Maximum sequence length
   * @param vocabSize Size of the vocabulary (number of unique tokens)
   */
  explicit EmbeddingLayer(base::DeviceType deviceType, int32_t dim,
                          int32_t seqLen, int32_t vocabSize);

  /**
   * @brief Validates the layer configuration and parameters
   * @return Status indicating success or specific error
   */
  base::Status check() const override;

  /**
   * @brief Performs the forward pass of the embedding layer
   *
   * Converts input token IDs to their corresponding embedding vectors
   * by looking up the embedding table.
   *
   * @return Status indicating success or specific error
   */
  base::Status forward() override;

 private:
  /** @brief Embedding dimension (size of each embedding vector) */
  int32_t dim_ = 0;

  /** @brief Maximum sequence length supported */
  int32_t seqLen_ = 0;

  /** @brief Size of the vocabulary */
  int32_t vocabSize_ = 0;
};

}  // namespace op

#endif  // INFER_INCLUDE_OP_EMBEDDING_H_
