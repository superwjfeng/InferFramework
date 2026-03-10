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

/**
 * @class Model
 * @brief Abstract base class for neural network models in the inference framework
 *
 * This class provides the interface for model operations including initialization,
 * inference, encoding/decoding, and buffer management. It serves as the base class
 * for specific model implementations (e.g., LLaMA, GPT, etc.).
 *
 * The Model class supports:
 * - Multiple device types (CPU, CUDA, etc.)
 * - Quantized and non-quantized models
 * - Token encoding and decoding
 * - KV cache management for efficient inference
 * - Embedding generation and forward pass operations
 *
 * @note This is an abstract class and cannot be instantiated directly.
 *       Concrete implementations must override all pure virtual methods.
 */
class Model {
 public:
  /**
   * @brief Constructs a Model instance with specified configuration
   *
   * @param tokenizerType Type of tokenizer to use for text encoding/decoding
   * @param modelType Type of model architecture (e.g., LLaMA, GPT)
   * @param tokenPath Path to the tokenizer model file
   * @param modelPath Path to the model weights file
   * @param isQuantModel Flag indicating whether the model is quantized
   */
  explicit Model(base::TokenizerType tokenizerType, base::ModelType modelType,
                 std::string tokenPath, std::string modelPath,
                 bool isQuantModel);

 public:
  /**
   * @brief Initializes the model on the specified device
   *
   * This method performs model initialization including loading weights,
   * creating layers, and allocating buffers on the target device.
   *
   * @param deviceType The device type to run the model on (CPU, CUDA, etc.)
   * @return Status indicating success or failure of initialization
   */
  virtual base::Status init(base::DeviceType deviceType) = 0;

  /**
   * @brief Predicts the next token given input and position tensors
   *
   * This method performs a forward pass through the model and predicts
   * the next token in the sequence. It handles both prompt and generation phases.
   *
   * @param input Input tensor containing token embeddings
   * @param posTensor Position tensor for positional encoding
   * @param isPrompt Flag indicating if this is the prompt phase (true) or generation phase (false)
   * @param[out] next The predicted next token ID
   * @return Status indicating success or failure of prediction
   */
  virtual base::Status predict(const tensor::Tensor& input,
                               const tensor::Tensor& posTensor, bool isPrompt,
                               int& next) const = 0;

  /**
   * @brief Performs a forward pass through the model
   *
   * @param input Input tensor containing token embeddings
   * @param posTensor Position tensor for positional encoding
   * @param[out] next The output token ID after forward pass
   * @return Status indicating success or failure
   */
  virtual base::Status forward(const tensor::Tensor& input,
                               const tensor::Tensor& posTensor,
                               int& next) const = 0;

  /**
   * @brief Generates embeddings for a sequence of tokens
   *
   * @param tokens Vector of token IDs to embed
   * @return EmbeddingOutput containing the embedded representations
   */
  virtual op::EmbeddingOutput embedding(
      const std::vector<int>& tokens) const = 0;

 private:
  /**
   * @brief Initializes memory buffers for the model
   *
   * Pure virtual method to be implemented by derived classes for allocating
   * memory buffers required by the model.
   */
  virtual void initMem() = 0;

  /**
   * @brief Creates all model layers
   *
   * @return Status indicating success or failure of layer creation
   */
  virtual base::Status createLayers() = 0;

  /**
   * @brief Creates parameterized layers (layers with learnable weights)
   */
  virtual void createParamLayers() = 0;

  /**
   * @brief Creates non-parameterized layers (layers without learnable weights)
   */
  virtual void createNonParamLayers() = 0;

  /**
   * @brief Creates quantized parameterized layers for quantized models
   */
  virtual void createParamQuantLayers() = 0;

 public:
  /**
   * @brief Gets the model type
   *
   * @return The type of model architecture (e.g., LLaMA, GPT)
   */
  inline base::ModelType modelType() const { return modelType_; }

  /**
   * @brief Gets the path to the tokenizer model file
   *
   * @return Path to the tokenizer file
   */
  inline const std::string& tokenPath() const { return tokenPath_; }

  /**
   * @brief Gets the path to the model weights file
   *
   * @return Path to the model file
   */
  inline const std::string& modelPath() const { return modelPath_; }

  /**
   * @brief Retrieves a mutable buffer by its type
   *
   * @param bufferIdx Index identifying the buffer type
   * @return Reference to the requested buffer tensor
   * @throws CHECK failure if buffer does not exist
   */
  inline virtual tensor::Tensor& getBuffer(ModelBufferType bufferIdx) {
    CHECK_GT(buffers_.count(bufferIdx), 0) << int(bufferIdx);
    return buffers_.at(bufferIdx);
  }

  /**
   * @brief Retrieves a const buffer by its type
   *
   * @param bufferIdx Index identifying the buffer type
   * @return Const reference to the requested buffer tensor
   * @throws CHECK failure if buffer does not exist
   */
  inline virtual const tensor::Tensor& getBuffer(
      ModelBufferType bufferIdx) const {
    CHECK_GT(buffers_.count(bufferIdx), 0) << int(bufferIdx);
    return buffers_.at(bufferIdx);
  }

  /**
   * @brief Checks if a token marks the end of a sentence
   *
   * @param token_idx Token ID to check
   * @return true if the token is an end-of-sentence marker, false otherwise
   */
  inline virtual bool isSentenceEnding(int32_t token_idx) const {
    CHECK(encodeLayer_ != nullptr);
    return encodeLayer_->isSentenceEnding(token_idx);
  }

  /**
   * @brief Encodes a text sentence into a sequence of token IDs
   *
   * @param sentence Input text string to encode
   * @return Vector of token IDs representing the encoded sentence
   */
  inline virtual std::vector<int32_t> encode(
      const std::string& sentence) const {
    CHECK(encodeLayer_ != nullptr);
    return encodeLayer_->encode(sentence);
  }

  /**
   * @brief Decodes a single token ID back to text
   *
   * @param tokenIdx Token ID to decode
   * @return Decoded text string
   */
  inline virtual std::string decode(int32_t tokenIdx) const {
    CHECK(this->encodeLayer_ != nullptr);
    return this->encodeLayer_->decode(tokenIdx);
  }

  /**
   * @brief Decodes a sequence of token IDs back to text
   *
   * @param tokenIdxs Vector of token IDs to decode
   * @return Decoded text string
   */
  inline virtual std::string decode(std::vector<int32_t> tokenIdxs) const {
    CHECK(this->encodeLayer_ != nullptr);
    return this->encodeLayer_->decode(tokenIdxs);
  }

 public:
  /**
   * @brief Extracts a slice of the KV cache for a specific layer and position
   *
   * This method retrieves the key and value cache tensors for efficient
   * attention computation during autoregressive generation.
   *
   * @param layerIdx Index of the transformer layer
   * @param tokenPos Position of the current token
   * @return Pair of tensors containing (key_cache, value_cache)
   */
  virtual std::pair<tensor::Tensor, tensor::Tensor> sliceKvCache(
      int32_t layerIdx, int32_t tokenPos) const;

  /**
   * @brief Prepares the input tensor for model forward pass
   *
   * Combines position information with embeddings to create the final
   * input tensor for the model.
   *
   * @param posTensor Position tensor for positional encoding
   * @param embeddingOutput Output from the embedding layer
   * @param isPrompt Flag indicating if this is the prompt phase
   * @return Filled input tensor ready for forward pass
   */
  virtual tensor::Tensor fillInput(const tensor::Tensor& posTensor,
                                   const op::EmbeddingOutput& embeddingOutput,
                                   bool isPrompt) const;

 protected:
  /**
   * @brief Inserts a tensor into the buffer map
   *
   * @param bufferIdx Type identifier for the buffer
   * @param tensor Tensor to insert into the buffer map
   * @return Status indicating success or failure
   */
  virtual base::Status insertBuffer(ModelBufferType bufferIdx,
                                    const tensor::Tensor& tensor);

  /**
   * @brief Reads and parses the model file from disk
   *
   * @return Status indicating success or failure of file reading
   */
  virtual base::Status readModelFile();

  /**
   * @brief Creates the encoding/tokenization layer
   *
   * @return Status indicating success or failure
   */
  virtual base::Status createEncodeLayer();

  /**
   * @brief Generates the model structure from the loaded file
   *
   * @return Status indicating success or failure
   */
  virtual base::Status genModelFromFile();

  /**
   * @brief Generates and validates model information from configuration
   *
   * @param config Model configuration object
   * @return Status indicating success or failure
   */
  virtual base::Status generateModelInfos(const ModelConfig& config) const;

  /**
   * @brief Performs post-processing after model inference
   *
   * Pure virtual method for implementing model-specific post-processing
   * operations such as sampling or output transformation.
   *
   * @param pos Position tensor
   * @param isPrompt Flag indicating if this is the prompt phase
   * @return Processed token ID
   */
  virtual int32_t postProcessing(const tensor::Tensor& pos,
                                 bool isPrompt) const = 0;

 protected:
  int32_t groupSize_ = 1;  ///< Group size for quantization (default: 1)
  bool isQuantModel_ = false;  ///< Flag indicating if model is quantized
  std::unique_ptr<TransformerConfig> transformerConfig_;  ///< Transformer architecture configuration

  std::string tokenPath_;  ///< Path to tokenizer model file
  std::string modelPath_;  ///< Path to model weights file

  std::unique_ptr<op::EncodeLayerBase> encodeLayer_;  ///< Tokenizer/encoder layer
  std::map<ModelBufferType, tensor::Tensor> buffers_;  ///< Map of model buffers (KV cache, etc.)
  std::unique_ptr<sampler::Sampler> sampler_;  ///< Sampling strategy for token generation
  std::shared_ptr<RawModelData> rawModelData_;  ///< Raw model data loaded from file
  base::DeviceType deviceType_ = base::DeviceType::kDeviceUnknown;  ///< Device type (CPU, CUDA, etc.)
  base::ModelType modelType_ = base::ModelType::kModelTypeUnknown;  ///< Model architecture type
  base::TokenizerType tokenizerType_ = base::TokenizerType::kEncodeUnknown;  ///< Tokenizer type
};

}  // namespace model

#endif  // INFER_INCLUDE_MODEL_MODEL_H_
