/**
 * @file argmax_sampler.h
 * @brief Argmax (greedy) sampling strategy implementation
 *
 * This file defines the argmax sampling strategy that always selects
 * the token with the highest probability (greedy decoding) during
 * text generation.
 */

#ifndef INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_
#define INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_

#include "base/base.h"
#include "sampler.h"

/**
 * @namespace sampler
 * @brief Namespace for token sampling implementations
 */
namespace sampler {

/**
 * @class ArgmaxSampler
 * @brief Greedy sampling strategy that selects the most probable token
 *
 * Implements deterministic argmax (greedy) sampling by always choosing
 * the token with the highest logit value. This produces deterministic,
 * focused output but may lack diversity. Also known as greedy decoding.
 */
class ArgmaxSampler : public Sampler {
 public:
  /**
   * @brief Constructs an argmax sampler
   * @param deviceType Device type for computation (CPU/CUDA)
   */
  explicit ArgmaxSampler(base::DeviceType deviceType) : Sampler(deviceType) {}

  /**
   * @brief Samples the token with maximum logit value (greedy selection)
   *
   * Finds and returns the index of the maximum value in the logits array.
   * This is a deterministic strategy that always selects the most probable
   * token, producing consistent but potentially less diverse outputs.
   *
   * @param logits Pointer to array of logit values
   * @param size Number of elements in the logits array (vocabulary size)
   * @param stream Optional CUDA stream for asynchronous operations
   * @return Index of the token with maximum logit value
   */
  size_t sample(const float* logits, size_t size,
                void* stream = nullptr) override;
};

}  // namespace sampler

#endif  // INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_
