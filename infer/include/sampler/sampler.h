#ifndef INFER_INCLUDE_SAMPLER_SAMPLER_H_
#define INFER_INCLUDE_SAMPLER_SAMPLER_H_

#include "base/base.h"
namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType deviceType) : deviceType_(deviceType) {}

  virtual size_t sample(const float* logits, size_t size,
                        void* stream = nullptr) = 0;

 protected:
  base::DeviceType deviceType_;
};
}  // namespace sampler

#endif  // INFER_INCLUDE_SAMPLER_SAMPLER_H_
