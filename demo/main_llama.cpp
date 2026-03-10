#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "base/base.h"
#include "model/llama3.h"
#include "tensor/tensor.h"

int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int totalSteps, bool needOutput = false) {
  auto tokens = model.encode(sentence);
  int32_t promptLen = tokens.size();
  LOG_IF(FATAL, promptLen <= 0) << "The prompt is empty after encoding.";

  int32_t pos = 0;
  int32_t next = -1;
  bool isPrompt = true;
  const auto& promptEmbedding = model.embedding(tokens);
  tensor::Tensor posTensor = model.getBuffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < totalSteps) {
    posTensor.index<int32_t>(0) = pos;
    if (pos < promptLen - 1) {
      // prefill
      tensor::Tensor input =
          model.fillInput(posTensor, promptEmbedding, isPrompt);
      model.predict(input, posTensor, isPrompt, next);
    } else {
      // decode
      isPrompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& tokenEmbedding = model.embedding(tokens);
      tensor::Tensor input =
          model.fillInput(posTensor, tokenEmbedding, isPrompt);
      model.predict(input, posTensor, isPrompt, next);
    }
    if (model.isSentenceEnding(next)) {
      break;
    }
    if (isPrompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (needOutput) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, totalSteps);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: " << argv[0] << "checkpoint_path tokenizer_path";
  }

  const char* checkpointPath = argv[1];  // e.g., out/model.bin
  const char* tokenizerPath = argv[2];

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizerPath,
                           checkpointPath, false);
  auto initStatus = model.init(base::DeviceType::kDeviceCUDA);

  if (!initStatus) {
    LOG(ERROR) << "Model init failed: " << initStatus.getErrMsg();
    return -1;
  }

  const std::string& sentence = "hello";
  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);

  return 0;
}
