#include "kernels_interface.h"

#include "base/base.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"

namespace kernel {
AddKernel getAddKernel(base::DeviceType deviceType) {
  if (deviceType == base::DeviceType::kDeviceCPU) {
    return addKernelCpu;
  } else if (deviceType == base::DeviceType::kDeviceCUDA) {
    return addKernelCu;
  } else {
    LOG(ERROR) << "Unsupported device type: " << static_cast<int>(deviceType);
    return nullptr;
  }
}
}  // namespace kernel
