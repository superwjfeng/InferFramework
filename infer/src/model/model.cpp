#include "model//model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <string>

#include "base/base.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/encode.h"

namespace model {
Model::Model(base::TokenizerType tokenizerType, base::ModelType modelType,
             std::string tokenPath, std::string modelPath, bool isQuantModel)
    : tokenizerType_(tokenizerType),
      modelType_(modelType),
      tokenPath_(std::move(tokenPath)),
      modelPath_(std::move(modelPath)),
      isQuantModel_(isQuantModel) {}

std::pair<tensor::Tensor, tensor::Tensor> Model::sliceKvCache(
    int32_t layerIdx, int32_t tokenPos) const {
  int32_t layer_offset =
      layerIdx * transformerConfig_->seqLen * transformerConfig_->kvDim;
  int32_t cache_offset = layer_offset + tokenPos * transformerConfig_->kvDim;

  float* key_cache_ptr = const_cast<float*>(
      getBuffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr = const_cast<float*>(
      getBuffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFP32, transformerConfig_->kvDim,
                     false, nullptr, key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFP32, transformerConfig_->kvDim,
                     false, nullptr, val_cache_ptr);
  key.setDeviceType(deviceType_);
  val.setDeviceType(deviceType_);
  return {key, val};
}

tensor::Tensor Model::fillInput(const tensor::Tensor& posTensor,
                                const op::EmbeddingOutput& embeddingOutput,
                                bool isPrompt) const {
  const int32_t pos = posTensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embeddingOutput;

  int32_t index = 0;
  if (isPrompt) {
    index = pos;
  }
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(
          transformerConfig_->dim * sizeof(float), nullptr,
          input_embeddings.ptr<float>(index * transformerConfig_->dim), true);
  tensor::Tensor input(base::DataType::kDataTypeFP32, transformerConfig_->dim);
  input.assign(input_emb_buffer);
  input.setDeviceType(deviceType_);
  return input;
}

base::Status Model::insertBuffer(ModelBufferType bufferIdx,
                                 const tensor::Tensor& tensor) {
  if (buffers_.count(bufferIdx) > 0) {
    return base::error::KeyHasExits(
        std::to_string(static_cast<int>(bufferIdx)) +
        "has existed in the buffer");
  }
  if (tensor.empty()) {
    return base::error::InvalidArgument(
        "The tensor is empty for inserting buffer.");
  }
  buffers_.insert({bufferIdx, tensor});
  return base::error::Success();
}

base::Status Model::readModelFile() {
  using namespace base;
  if (modelPath_.empty()) {
    return error::PathNotValid(
        "Failed to open the weight file, the model path is empty!");
  }

  int32_t fd = open(modelPath_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + modelPath_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(modelPath_.data(), "rb");
  if (!file) {
    return error::PathNotValid(
        "Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig();
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  if (isQuantModel_) {
    if (fread(&groupSize_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  auto gen_status = generateModelInfos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!isQuantModel_) {
    rawModelData_ = std::make_shared<RawModelDataFp32>();
  } else {
    rawModelData_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  rawModelData_->fileSize = sb.st_size;

  rawModelData_->fd = fd;
  rawModelData_->data = mmap(nullptr, rawModelData_->fileSize, PROT_READ,
                             MAP_PRIVATE, rawModelData_->fd, 0);

  if (rawModelData_->data == MAP_FAILED || rawModelData_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " +
                                  modelPath_ + " into memory.");
  }
  if (!isQuantModel_) {
    rawModelData_->weightData =
        static_cast<int8_t*>(rawModelData_->data) + sizeof(ModelConfig);
  } else {
    rawModelData_->weightData = static_cast<int8_t*>(rawModelData_->data) +
                                sizeof(ModelConfig) + sizeof(groupSize_);
  }
  if (rawModelData_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError(
        "Failed to map the weight file " + modelPath_ +
        " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::createEncodeLayer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizerType_ == TokenizerType::kEncodeSpe) {
    encodeLayer_ =
        std::make_unique<op::SpeEncodeLayer>(this->tokenPath_, true, false);
  } else {
  }
  if (!encodeLayer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  transformerConfig_->vocabSize = encodeLayer_->vocabSize();
  if (transformerConfig_->vocabSize <= 0) {
    return error::InternalError(
        "The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::genModelFromFile() {
  using namespace base;
  transformerConfig_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = createEncodeLayer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = readModelFile();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << modelPath_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = createLayers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << modelPath_
               << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

base::Status Model::generateModelInfos(const ModelConfig& config) const {
  transformerConfig_->dim = config.dim;
  transformerConfig_->hiddenDim = config.hiddenDim;
  transformerConfig_->layerNum = config.layerNum;
  transformerConfig_->headNum = config.headNum;
  transformerConfig_->kvHeadNum = config.kvHeadNum;
  transformerConfig_->seqLen = config.seqLen;

  transformerConfig_->kvDim = (config.dim * config.kvHeadNum) / config.headNum;
  transformerConfig_->kvMul = config.headNum / config.kvHeadNum;
  transformerConfig_->headSize = config.dim / config.headNum;
#if defined(QWEN3_SUPPORT)
  transformerConfig_->immediateDim = config.immediateDim;
#endif
  if (transformerConfig_->vocabSize > 0) {
    transformerConfig_->isSharedWeight = true;
  } else {
    transformerConfig_->isSharedWeight = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token
  //       list.");
  // }
  transformerConfig_->vocabSize = std::abs(config.vocabSize);
  return base::error::Success();
}
}  // namespace model
