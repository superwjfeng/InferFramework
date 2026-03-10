#include "op/layer.h"

#include <cstdarg>

namespace op {
base::Status Layer::checkTensor(const tensor::Tensor& tensor,
                                base::DeviceType deviceType,
                                base::DataType dataType) const {
  if (tensor.empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.deviceType() != deviceType) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.dataType() != dataType) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::checkTensorWithDim(const tensor::Tensor& tensor,
                                       base::DeviceType deviceType,
                                       base::DataType dataType, ...) const {
  std::va_list args;
  if (tensor.empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.deviceType() != deviceType) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.dataType() != dataType) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, dataType);
  int32_t dimSize = tensor.dimSize();
  for (int32_t i = 0; i < dimSize; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.getDim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" +
                                          std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

void Layer::toCuda() {
  for (auto& input : inputs_) {
    if (!input.empty()) {
      input.toCuda(cudaConfig_ ? cudaConfig_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.empty()) {
      output.toCuda(cudaConfig_ ? cudaConfig_->stream : nullptr);
    }
  }
}

base::Status LayerParam::setWeight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.dataType() == base::DataType::kDataTypeFP32);
  if (!weight.empty()) {
    CHECK(weight.deviceType() == deviceType_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

base::Status LayerParam::setWeight(int32_t idx,
                                   const std::vector<int32_t>& dims,
                                   const void* weight_ptr,
                                   base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float),
                                std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
      size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->setDeviceType(device_type);
  }

  if (!isQuantLayer_) {
    tensor::Tensor weight(base::DataType::kDataTypeFP32, dims);
    weight.setDeviceType(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.setDeviceType(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % groupSize_ == 0);

    int32_t scale_nums = weight_size / groupSize_;
    scales_ = tensor::Tensor{
        base::DataType::kDataTypeFP32, scale_nums, false, nullptr,
        reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.setDeviceType(device_type);
  }

  return base::error::Success();
}

base::Status Layer::forward(const tensor::Tensor& input1,
                            const tensor::Tensor& output1) {
  this->setInput(0, input1);
  this->setOutput(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1,
                            const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->setInput(0, input1);
  this->setInput(1, input2);

  this->setOutput(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1,
                            const tensor::Tensor& input2,
                            const tensor::Tensor& input3,
                            const tensor::Tensor& output1) {
  this->setInput(0, input1);
  this->setInput(1, input2);
  this->setInput(2, input3);

  this->setOutput(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1,
                            const tensor::Tensor& input2,
                            const tensor::Tensor& input3,
                            const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->setInput(0, input1);
  this->setInput(1, input2);
  this->setInput(2, input3);
  this->setInput(3, input4);

  this->setOutput(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1,
                            const tensor::Tensor& input2,
                            const tensor::Tensor& input3,
                            const tensor::Tensor& input4,
                            const tensor::Tensor& input5,
                            const tensor::Tensor& output1) {
  this->setInput(0, input1);
  this->setInput(1, input2);
  this->setInput(2, input3);
  this->setInput(3, input4);
  this->setInput(4, input5);

  this->setOutput(0, output1);
  return this->forward();
}

}  // namespace op
