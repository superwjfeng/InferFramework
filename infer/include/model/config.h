/**
 * @file config.h
 * @brief Model configuration structures for neural network architectures
 *
 * This file defines configuration structures that hold hyperparameters
 * and architectural settings for various model types, particularly
 * transformer-based language models.
 */

#ifndef INFER_INCLUDE_MODEL_CONFIG_H_
#define INFER_INCLUDE_MODEL_CONFIG_H_

#include <cstdint>

/**
 * @namespace model
 * @brief Namespace containing model-related structures and utilities
 */
namespace model {

/**
 * @struct ModelConfig
 * @brief Basic configuration structure for language models
 *
 * Contains fundamental hyperparameters that define the architecture
 * of a neural network model, including dimensions, layer counts,
 * and attention head configurations.
 */
struct ModelConfig {
  /** @brief Model embedding dimension */
  int32_t dim = 0;

  /** @brief Hidden layer dimension (typically in feed-forward networks) */
  int32_t hiddenDim = 0;

  /** @brief Number of transformer layers */
  int32_t layerNum = 0;

  /** @brief Number of attention heads */
  int32_t headNum = 0;

  /** @brief Number of key-value heads (for grouped-query attention) */
  int32_t kvHeadNum = 0;

  /** @brief Vocabulary size of the tokenizer */
  int32_t vocabSize = 0;

  /** @brief Maximum sequence length supported by the model */
  int32_t seqLen = 0;
};

/**
 * @struct TransformerConfig
 * @brief Extended configuration structure for transformer models
 *
 * Contains all hyperparameters needed for transformer-based models,
 * including derived dimensions for key-value attention and configuration
 * flags for weight sharing strategies.
 */
struct TransformerConfig {
  /** @brief Dimension of key-value projections */
  int32_t kvDim = 0;

  /** @brief Multiplier for key-value dimensions (headNum / kvHeadNum) */
  int32_t kvMul = 0;

  /** @brief Size of each attention head (dim / headNum) */
  int32_t headSize = 0;

  /** @brief Vocabulary size of the tokenizer */
  int32_t vocabSize = 0;

  /** @brief Model embedding dimension */
  int32_t dim = 0;

  /** @brief Hidden layer dimension in feed-forward networks */
  int32_t hiddenDim = 0;

  /** @brief Number of transformer layers */
  int32_t layerNum = 0;

  /** @brief Number of query attention heads */
  int32_t headNum = 0;

  /** @brief Number of key-value heads (for grouped-query attention) */
  int32_t kvHeadNum = 0;

  /** @brief Maximum sequence length supported by the model */
  int32_t seqLen = 0;

  /** @brief Whether input and output embeddings share weights */
  bool isSharedWeight = false;
};
}  // namespace model
#endif  // INFER_INCLUDE_MODEL_CONFIG_H_
