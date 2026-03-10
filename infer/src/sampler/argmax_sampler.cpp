#include "sampler/argmax_sampler.h"

#include <algorithm>
#include <iterator>

#include "../op/kernels/cuda/argmax_kernel.cuh"
#include "base/base.h"

namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (deviceType_ == base::DeviceType::kDeviceCPU) {
    size_t next =
        std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    size_t next = kernel::argmaxKernelCu(logits, size, stream);
    return next;
  }
}
}  // namespace sampler
