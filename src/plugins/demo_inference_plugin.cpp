#include "plugins/demo_inference_plugin.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "demo_inference.pb.h"

namespace robo_lab {
namespace {

std::filesystem::path resolve_inference_yaml(const std::filesystem::path& plugin_config_path,
                                               const std::string& inference_field) {
  std::filesystem::path rel(inference_field);
  if (rel.is_absolute()) {
    return rel;
  }
  std::filesystem::path next_to_plugin = plugin_config_path.parent_path() / rel;
  if (std::filesystem::exists(next_to_plugin)) {
    return next_to_plugin;
  }
  if (std::filesystem::exists(rel)) {
    return rel;
  }
  return next_to_plugin;
}

}  // namespace

bool DemoInferencePlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "demo_inference_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
    return false;
  }

  std::string infer_key = "inference_config";
  if (!root[infer_key] && root["inferenc_config"]) {
    infer_key = "inferenc_config";
  }
  if (!root[infer_key]) {
    std::cerr << "demo_inference_plugin: inference_config (or inferenc_config) missing in " << config_path << '\n';
    return false;
  }

  const std::string infer_rel = root[infer_key].as<std::string>();
  const std::filesystem::path infer_path =
      resolve_inference_yaml(std::filesystem::path(config_path), infer_rel);
  if (!std::filesystem::exists(infer_path)) {
    std::cerr << "demo_inference_plugin: inference file not found: " << infer_path << '\n';
    return false;
  }

  YAML::Node infer_root;
  try {
    infer_root = YAML::LoadFile(infer_path.string());
  } catch (const YAML::Exception& e) {
    std::cerr << "demo_inference_plugin: YAML error in " << infer_path << ": " << e.what() << '\n';
    return false;
  }

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();

  inference_ = std::make_unique<inference::InferenceInterface>();
  if (!inference_->initialize(infer_root)) {
    std::cerr << "demo_inference_plugin: InferenceInterface::initialize failed\n";
    return false;
  }

  if (infer_root["policy"] && infer_root["policy"]["frequency"]) {
    frequency_hz_ = infer_root["policy"]["frequency"].as<int>();
    if (frequency_hz_ < 1) {
      frequency_hz_ = 1;
    }
  }

  if (!inference_->io_processor_) {
    std::cerr << "demo_inference_plugin: internal error (no IO processor)\n";
    return false;
  }


  for (const auto& topic : inference_->io_processor_->input_topics()) {
    message_system_->subscribe(topic, [this](const std::string& key, const std::string& payload) {
      on_obs_message(key, payload);
    });
  }

  std::cout << "demo_inference_plugin: initialized (inference=" << infer_path.string()
            << ", freq_hz=" << frequency_hz_ << ")\n";
  return true;
}

void DemoInferencePlugin::on_obs_message(const std::string& key, const std::string& payload) {
  if (!inference_ || !inference_->io_processor_) {
    return;
  }
  // std::cout << "demo_inference_plugin: on_obs_message: key=" << key << " payload=" << payload << std::endl;
  std::string tensor_payload;
  demo_inference::Observation obs;
  if (obs.ParseFromString(payload)) {
    const int expected = static_cast<int>(inference_->io_processor_->input_buffer.size());
    if (obs.values_size() != expected) {
      std::cerr << "demo_inference_plugin: Observation.values size " << obs.values_size()
                << " != config input dim " << expected << '\n';
      return;
    }
    tensor_payload.resize(static_cast<size_t>(expected) * sizeof(float));
    for (int i = 0; i < expected; ++i) {
      float v = obs.values(i);
      std::memcpy(tensor_payload.data() + static_cast<size_t>(i) * sizeof(float), &v, sizeof(float));
    }
  } else {
    tensor_payload = payload;
  }

  if (!inference_->io_processor_->preProcess(key, tensor_payload)) {
    return;
  }
  if (!inference_->inference()) {
    std::cerr << "demo_inference_plugin: inference failed\n";
    return;
  }
  for (const auto& out_topic : inference_->io_processor_->output_topics()) {
    std::string out_raw;
    if (!inference_->io_processor_->postProcess(out_topic, &out_raw)) {
      std::cerr << "demo_inference_plugin: postProcess failed for " << out_topic << '\n';
      continue;
    }
    demo_inference::Action act;
    const size_t nfloat = out_raw.size() / sizeof(float);
    for (size_t i = 0; i < nfloat; ++i) {
      float v = 0.f;
      std::memcpy(&v, out_raw.data() + i * sizeof(float), sizeof(float));
      act.add_values(v);
    }
    std::string out_serialized;
    if (!act.SerializeToString(&out_serialized)) {
      std::cerr << "demo_inference_plugin: Action.SerializeToString failed\n";
      continue;
    }
    message_system_->publish(out_topic, out_serialized);
  }
}

void DemoInferencePlugin::run() {
  stop_ = false;
  std::cout << "demo_inference_plugin: run loop started (event-driven on observation topics)\n";
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cout << "demo_inference_plugin: run loop exited\n";
}

void DemoInferencePlugin::stop() {
  stop_ = true;
  if (message_system_) {
    message_system_->close();
  }
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::DemoInferencePlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
