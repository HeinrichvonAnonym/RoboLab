#pragma once

#include "inference/inference_interface.h"
#include "message_system.h"
#include "plugin_interface.h"

#include <atomic>
#include <memory>
#include <string>

#include <demo_inference.pb.h>

namespace robo_lab {

/// Loads a TensorRT policy via InferenceInterface and bridges Zenoh IO.
/// Observation payloads: `demo_inference::Observation` protobuf (preferred) or raw float32 bytes.
/// Action payloads: `demo_inference::Action` protobuf.
class DemoInferencePlugin : public Plugin {
 public:
  DemoInferencePlugin() = default;
  ~DemoInferencePlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  // proto 
  std::string config_path_;
  std::atomic<bool> stop_{false};

  // std::string action_topic_;

  std::unique_ptr<MessageSystem> message_system_;
  std::unique_ptr<inference::InferenceInterface> inference_;

  int frequency_hz_{50};

  void on_obs_message(const std::string& key, const std::string& payload);

  // bool publish_action(const std::string& key, const std::string& payload);
};

}  // namespace robo_lab
