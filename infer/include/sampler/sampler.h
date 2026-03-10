/**
 * @file sampler.h
 * @brief Token sampling strategies for language model generation
 *
 * This file defines the abstract base class for implementing various sampling
 * strategies used in language model inference to select the next token from
 * probability distributions (logits).
 */

#ifndef INFER_INCLUDE_SAMPLER_SAMPLER_H_
#define INFER_INCLUDE_SAMPLER_SAMPLER_H_

#include "base/base.h"

/**
 * @namespace sampler
 * @brief Namespace for token sampling implementations
 */
namespace sampler {

/**
 * @class Sampler
 * @brief Abstract base class for token sampling strategies
 *
 * Provides the interface for implementing various sampling algorithms
 * (e.g., greedy, top-k, top-p/nucleus, temperature sampling) that select
 * the next token from model logits during text generation.
 */
class Sampler {
 public:
  /**
   * @brief Constructs a sampler with specified device type
   * @param deviceType Device type for computation (CPU/CUDA)
   */
  explicit Sampler(base::DeviceType deviceType) : deviceType_(deviceType) {}

  /**
   * @brief Samples a token index from the logit distribution
   *
   * Pure virtual function that implements the sampling strategy.
   * Subclasses define specific algorithms (greedy, top-k, top-p, etc.)
   * to select a token index from the probability distribution.
   *
   * @param logits Pointer to array of logit values (unnormalized probabilities)
   * @param size Number of elements in the logits array (vocabulary size)
   * @param stream Optional CUDA stream for asynchronous operations (default: nullptr)
   * @return Selected token index from the vocabulary
   */
  virtual size_t sample(const float* logits, size_t size,
                        void* stream = nullptr) = 0;

 protected:
  /** @brief Device type for computation (CPU or CUDA) */
  base::DeviceType deviceType_;
};
}  // namespace sampler

#endif  // INFER_INCLUDE_SAMPLER_SAMPLER_H_
