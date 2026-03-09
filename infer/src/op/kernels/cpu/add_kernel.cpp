#include "add_kernel.h"

#include <armadillo>

#include "base/base.h"
namespace kernel {
void addKernelCpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                  const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.empty(), false);
  CHECK_EQ(input2.empty(), false);
  CHECK_EQ(output.empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  arma::fvec inputVec1(const_cast<float*>(input1.ptr<float>()), input1.size(),
                       false, true);
  arma::fvec inputVec2(const_cast<float*>(input2.ptr<float>()), input2.size(),
                       false, true);
  arma::fvec outputVec(const_cast<float*>(output.ptr<float>()), output.size(),
                       false, true);
  outputVec = inputVec1 + inputVec2;
}

}  // namespace kernel
