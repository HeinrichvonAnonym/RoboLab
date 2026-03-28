#pragma once

#include <NvInfer.h>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace robo_lab {
namespace inference {

struct IOTensorIndiceMap {
  std::string topic;
  int start_indice{};
  int end_indice{};
};

class IOProcessor {
 public:
  IOProcessor();
  ~IOProcessor();

  bool initialize(YAML::Node& config);
  /// Zenoh payload -> flat input tensor (tensor range expected by TRT).
  bool preProcess(const std::string& key, const std::string& payload);
  /// Flat output tensor -> binary float32 payload for Zenoh publish.
  bool postProcess(const std::string& key, std::string* out_payload);

  [[nodiscard]] const std::vector<std::string>& input_topics() const { return input_topics_; }
  [[nodiscard]] const std::vector<std::string>& output_topics() const { return output_topics_; }

  std::vector<float> input_buffer;
  std::vector<float> output_buffer;

 private:
  float input_tensor_min_{-1.f};
  float input_tensor_max_{1.f};
  float output_tensor_min_{-1.f};
  float output_tensor_max_{1.f};
  std::vector<std::string> input_topics_;
  std::vector<std::string> output_topics_;
  std::vector<std::string> input_types_;
  std::vector<std::string> output_types_;
  std::vector<IOTensorIndiceMap> input_tensor_indice_map_;
  std::vector<IOTensorIndiceMap> output_tensor_indice_map_;
  std::vector<float> input_offsets_;
  std::vector<float> input_scales_;
  std::vector<float> output_offsets_;
  std::vector<float> output_scales_;
  std::vector<bool> input_use_clips_;
  std::vector<bool> output_use_clips_;
};

struct TrtIoStaging {
  std::string name;
  void* hostPinned{};
  void* devicePtr{};
  size_t nbytes{};
};

class TRTSession {
 public:
  TRTSession();
  ~TRTSession();

  bool loadEngine(const std::string& engine_path);
  bool findDynamicalInputTensors();
  bool buildIOStagings();
  bool writeToInputStaging(const std::vector<float>& input);
  bool AsynInfer();
  bool readFromOutputStaging(std::vector<float>& output);
  bool synchronize();
  bool destroy();

 private:
  void freeBuffers();

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<void*> hostBuffers_;
  std::vector<void*> deviceBuffers_;
  std::vector<void*> pinnedHostBuffers_;
  std::vector<TrtIoStaging> inputStaging_;
  std::vector<TrtIoStaging> outputStaging_;
  cudaStream_t stream_{};
};

class InferenceInterface {
 public:
  InferenceInterface();
  ~InferenceInterface();

  bool initialize(YAML::Node& config);
  bool preProcess();
  bool inference();
  bool postProcess();

  std::unique_ptr<TRTSession> trt_session_;
  std::unique_ptr<IOProcessor> io_processor_;
};

}  // namespace inference
}  // namespace robo_lab
