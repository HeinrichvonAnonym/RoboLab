#include "inference/inference_interface.h"

#include <iostream>

namespace robo_lab {
namespace inference {

InferenceInterface::InferenceInterface() = default;

InferenceInterface::~InferenceInterface() = default;

bool InferenceInterface::initialize(YAML::Node& config) {
  if (!config["obs"]) {
    std::cerr << "obs not found in config" << std::endl;
    return false;
  }
  if (!config["action"]) {
    std::cerr << "action not found in config" << std::endl;
    return false;
  }
  io_processor_ = std::make_unique<IOProcessor>();
  if (!io_processor_->initialize(config)) {
    std::cerr << "failed to initialize IO processor" << std::endl;
    return false;
  }

  if (!config["session"]) {
    std::cerr << "session not found in config" << std::endl;
    return false;
  }
  if (config["session"]["type"] && config["session"]["type"].as<std::string>() == "trt") {
    trt_session_ = std::make_unique<TRTSession>();
    std::string model_path = config["session"]["model_path"].as<std::string>();
    if (!trt_session_->loadEngine(model_path)) {
      std::cerr << "failed to load TRT engine" << std::endl;
      return false;
    }
  } else {
    std::cerr << "unsupported or missing session type" << std::endl;
    return false;
  }

  return true;
}

bool InferenceInterface::preProcess() { return true; }

bool InferenceInterface::inference() {
  if (!trt_session_ || !io_processor_) {
    return false;
  }
  if (!trt_session_->writeToInputStaging(io_processor_->input_buffer)) {
    return false;
  }
  if (!trt_session_->AsynInfer()) {
    return false;
  }
  if (!trt_session_->synchronize()) {
    return false;
  }
  if (!trt_session_->readFromOutputStaging(io_processor_->output_buffer)) {
    return false;
  }
  // std::cout << "InferenceInterface: inference success" << std::endl;
  return true;
}

bool InferenceInterface::postProcess() { return true; }

}  // namespace inference
}  // namespace robo_lab
