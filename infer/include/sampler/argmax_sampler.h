#ifndef INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_
#define INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_

#include "base/base.h"
#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType deviceType) : Sampler(deviceType) {}

  size_t sample(const float* logits, size_t size,
                void* stream = nullptr) override;
};

}  // namespace sampler

#endif  // INFER_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_
