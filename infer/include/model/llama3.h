#ifndef INFER_INCLUDE_MODEL_LLAMA3_H_
#define INFER_INCLUDE_MODEL_LLAMA3_H_

#include <memory>

#include "base/cuda_config.h"
#include "model/model.h"
#include "op/layer.h"
namespace model {
/**
 * @struct LLama2Layers
 * @brief Container for all neural network layers used in LLaMA 2 model
 *
 * This structure holds shared pointers to all the layers that comprise a LLaMA 2
 * transformer model, including attention layers, feed-forward layers, normalization
 * layers, and utility layers. The layers are organized to support multi-layer
 * transformer architecture with grouped query attention.
 */
struct LLama2Layers {
  std::shared_ptr<op::Layer> addLayer;       ///< Layer for element-wise addition operations
  std::shared_ptr<op::Layer> ropeLayer;      ///< Rotary Position Embedding (RoPE) layer
  std::shared_ptr<op::Layer> swigluLayer;    ///< SwiGLU activation function layer
  std::shared_ptr<op::Layer> mhaLayer;       ///< Multi-Head Attention (MHA) layer

  std::vector<std::shared_ptr<op::Layer>> wqLayers;  ///< Query projection layers for each transformer block
  std::vector<std::shared_ptr<op::Layer>> wkLayers;  ///< Key projection layers for each transformer block
  std::vector<std::shared_ptr<op::Layer>> wvLayers;  ///< Value projection layers for each transformer block
  std::vector<std::shared_ptr<op::Layer>> woLayers;  ///< Output projection layers for each transformer block

  std::vector<std::shared_ptr<op::Layer>> w1Layers;       ///< First feed-forward projection layers (gate)
  std::vector<std::shared_ptr<op::Layer>> w2Layers;       ///< Second feed-forward projection layers (down)
  std::vector<std::shared_ptr<op::Layer>> rmsnormLayers;  ///< RMSNorm normalization layers
  std::vector<std::shared_ptr<op::Layer>> w3Layers;       ///< Third feed-forward projection layers (up)
  std::shared_ptr<op::Layer> clsLayer;                    ///< Classification/output layer for final logits

  std::shared_ptr<op::Layer> embeddingLayer;  ///< Token embedding layer

  /**
   * @brief Transfers all layers to CUDA device
   *
   * This method moves all layer parameters and operations to GPU memory
   * for accelerated inference using CUDA.
   *
   * @param config CUDA configuration object containing device settings
   */
  void toCuda(std::shared_ptr<kernel::CudaConfig> config);
};

/**
 * @class LLama2Model
 * @brief Concrete implementation of LLaMA 2 transformer language model
 *
 * This class implements the LLaMA 2 architecture, a decoder-only transformer model
 * developed by Meta AI. The implementation includes:
 * - Multi-head attention with grouped query attention
 * - RoPE (Rotary Position Embedding) for positional encoding
 * - SwiGLU activation function in feed-forward networks
 * - RMSNorm for layer normalization
 * - Support for both quantized and non-quantized models
 * - CUDA acceleration support
 *
 * The model follows the transformer decoder architecture with multiple identical
 * layers, each containing self-attention and feed-forward sub-layers.
 *
 * @see Model Base class providing the model interface
 * @see LLama2Layers Structure containing all model layers
 */
class LLama2Model : public Model {
 public:
  /**
   * @brief Constructs a LLaMA 2 model instance
   *
   * @param tokenizerType Type of tokenizer (e.g., SentencePiece, BPE)
   * @param tokenPath Path to the tokenizer model file
   * @param modelPath Path to the model weights file
   * @param isQuantModel Flag indicating whether to use quantized weights
   */
  explicit LLama2Model(base::TokenizerType tokenizerType, std::string tokenPath,
                       std::string modelPath, bool isQuantModel);

  /**
   * @brief Initializes the LLaMA 2 model on the specified device
   *
   * Performs model initialization including loading weights, creating layers,
   * and allocating memory buffers on the target device.
   *
   * @param deviceType The device type to run the model on (CPU or CUDA)
   * @return Status indicating success or failure of initialization
   */
  base::Status init(base::DeviceType deviceType) override;

  /**
   * @brief Predicts the next token using LLaMA 2 inference
   *
   * Executes a complete forward pass through the model and predicts the next
   * token in the sequence. Handles both prompt processing and token generation.
   *
   * @param input Input tensor containing token embeddings
   * @param posTensor Position tensor for RoPE positional encoding
   * @param isPrompt Flag indicating if this is prompt phase (true) or generation (false)
   * @param[out] next The predicted next token ID
   * @return Status indicating success or failure
   */
  base::Status predict(const tensor::Tensor& input,
                       const tensor::Tensor& posTensor, bool isPrompt,
                       int& next) const override;

  /**
   * @brief Performs a forward pass through the LLaMA 2 model
   *
   * @param input Input tensor containing token embeddings
   * @param posTensor Position tensor for RoPE positional encoding
   * @param[out] next The output token ID after forward pass
   * @return Status indicating success or failure
   */
  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& posTensor,
                       int& next) const override;

  /**
   * @brief Generates embeddings for input tokens
   *
   * Converts a sequence of token IDs to their corresponding embeddings
   * using the model's embedding layer.
   *
   * @param tokens Vector of token IDs to embed
   * @return EmbeddingOutput containing the embedded representations
   */
  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  /**
   * @brief Initializes memory buffers for LLaMA 2 model
   *
   * Allocates all necessary memory buffers including KV cache, attention masks,
   * and intermediate computation buffers.
   */
  void initMem() override;

  /**
   * @brief Creates all layers for the LLaMA 2 model
   *
   * Instantiates all transformer layers based on model configuration,
   * including attention, feed-forward, and normalization layers.
   *
   * @return Status indicating success or failure of layer creation
   */
  base::Status createLayers() override;

  /**
   * @brief Creates parameterized layers with learnable weights
   *
   * Instantiates layers that contain trainable parameters such as
   * attention projection layers and feed-forward layers.
   */
  void createParamLayers() override;

  /**
   * @brief Creates non-parameterized layers
   *
   * Instantiates layers without trainable parameters such as
   * activation functions and utility layers.
   */
  void createNonParamLayers() override;

  /**
   * @brief Creates quantized versions of parameterized layers
   *
   * Instantiates quantized layers for reduced memory footprint
   * and faster inference when quantization is enabled.
   */
  void createParamQuantLayers() override;

  /**
   * @brief Performs multi-head attention computation
   *
   * Executes the multi-head attention mechanism with grouped query attention,
   * including Q-K-V projection, attention score computation, and output projection.
   *
   * @param layer_idx Index of the transformer layer
   * @param posTensor Position tensor for RoPE encoding
   */
  void attentionMha(int32_t layer_idx, const tensor::Tensor& posTensor) const;

  /**
   * @brief Applies RMSNorm normalization before attention
   *
   * Performs Root Mean Square Layer Normalization on the input tensor
   * before the attention sub-layer.
   *
   * @param layer_idx Index of the transformer layer
   * @param input Input tensor to normalize
   */
  void attentionRms(int32_t layer_idx, const tensor::Tensor& input) const;

  /**
   * @brief Executes the feed-forward network
   *
   * Performs the feed-forward transformation using SwiGLU activation:
   * FFN(x) = (W2 * SwiGLU(W1 * x, W3 * x))
   *
   * @param layer_idx Index of the transformer layer
   * @param input Input tensor to the feed-forward network
   */
  void feedForward(int32_t layer_idx, const tensor::Tensor& input) const;

  /**
   * @brief Computes query, key, and value projections
   *
   * Projects the input tensor to Q, K, V representations and applies
   * RoPE positional encoding to queries and keys.
   *
   * @param layer_idx Index of the transformer layer
   * @param posTensor Position tensor for RoPE encoding
   */
  void attentionQkv(int32_t layer_idx, const tensor::Tensor& posTensor) const;

  /**
   * @brief Computes final output logits over vocabulary
   *
   * Projects the final hidden state to vocabulary size to produce
   * logits for next token prediction.
   *
   * @param input Final hidden state tensor from last transformer layer
   */
  void clsLogits(const tensor::Tensor& input) const;

  /**
   * @brief Performs post-processing and sampling
   *
   * Applies sampling strategy (greedy, top-k, top-p, etc.) to select
   * the next token from the output logits.
   *
   * @param pos Position tensor
   * @param isPrompt Flag indicating if this is prompt phase
   * @return Sampled token ID
   */
  int32_t postProcessing(const tensor::Tensor& pos,
                         bool isPrompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cudaConfig_;  ///< CUDA configuration for GPU acceleration
  std::unique_ptr<LLama2Layers> llamaLayers_;      ///< Container holding all model layers
};
}  // namespace model

#endif  // INFER_INCLUDE_MODEL_LLAMA3_H_
