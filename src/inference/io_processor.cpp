#include "inference/inference_interface.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace robo_lab {
namespace inference {

IOProcessor::IOProcessor() = default;

IOProcessor::~IOProcessor() = default;

bool IOProcessor::initialize(YAML::Node& config) {
  try {
    if (!config["obs"] || !config["action"]) {
      std::cerr << "IOProcessor: obs and action are required" << std::endl;
      return false;
    }
    if (!config["obs"]["members"] || !config["action"]["members"]) {
      std::cerr << "IOProcessor: obs and action members are required" << std::endl;
      return false;
    }

    if (config["obs"]["tensor_min"] && config["obs"]["tensor_max"]) {
      input_tensor_min_ = config["obs"]["tensor_min"].as<float>();
      input_tensor_max_ = config["obs"]["tensor_max"].as<float>();
    }
    if (config["action"]["tensor_min"] && config["action"]["tensor_max"]) {
      output_tensor_min_ = config["action"]["tensor_min"].as<float>();
      output_tensor_max_ = config["action"]["tensor_max"].as<float>();
    }

    int input_tensor_indice = 0;
    for (const auto& member : config["obs"]["members"]) {
      input_topics_.push_back(member["topic"].as<std::string>());
      input_types_.push_back(member["type"].as<std::string>());
      int dim = member["dim"].as<int>();
      input_tensor_indice_map_.push_back(
          IOTensorIndiceMap{member["topic"].as<std::string>(), input_tensor_indice,
                            input_tensor_indice + dim - 1});
      for (int i = 0; i < dim; i++) {
        input_offsets_.push_back(member["offset"][i].as<float>());
        input_scales_.push_back(member["scale"][i].as<float>());
        input_use_clips_.push_back(member["use_clip"].as<bool>());

        input_buffer.push_back(0.0f);
      }
      input_tensor_indice += dim;
    }

    int output_tensor_indice = 0;
    for (const auto& member : config["action"]["members"]) {
      output_topics_.push_back(member["topic"].as<std::string>());
      output_types_.push_back(member["type"].as<std::string>());
      int dim = member["dim"].as<int>();
      output_tensor_indice_map_.push_back(
          IOTensorIndiceMap{member["topic"].as<std::string>(), output_tensor_indice,
                            output_tensor_indice + dim - 1});
      for (int i = 0; i < dim; i++) {
        output_offsets_.push_back(member["offset"][i].as<float>());
        output_scales_.push_back(member["scale"][i].as<float>());
        output_use_clips_.push_back(member["use_clip"].as<bool>());

        output_buffer.push_back(0.0f);
      }
      output_tensor_indice += dim;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "IOProcessor: initialize failed: " << e.what() << std::endl;
    return false;
  }
}

bool IOProcessor::preProcess(const std::string& key, const std::string& payload) {
  try {
    auto it = std::find(input_topics_.begin(), input_topics_.end(), key);
    if (it == input_topics_.end()) {
      std::cerr << "IOProcessor: key not found in input_topics_" << std::endl;
      return false;
    }

    int key_indice = static_cast<int>(it - input_topics_.begin());
    int start_indice = input_tensor_indice_map_[static_cast<size_t>(key_indice)].start_indice;
    int end_indice = input_tensor_indice_map_[static_cast<size_t>(key_indice)].end_indice;
    int dim = end_indice - start_indice + 1;

    if (input_types_[static_cast<size_t>(key_indice)] == "float32") {
      if (payload.size() != static_cast<size_t>(dim) * sizeof(float)) {
        std::cerr << "IOProcessor: float32 payload size mismatch (expected " << (dim * sizeof(float))
                  << " bytes)" << std::endl;
        return false;
      }
      const auto* pf = reinterpret_cast<const float*>(payload.data());
      for (int i = 0; i < dim; ++i) {
        int idx = start_indice + i;
        float physical = pf[i];
        float normalized = (physical - input_offsets_[static_cast<size_t>(idx)]) /
                           input_scales_[static_cast<size_t>(idx)];
        if (input_use_clips_[static_cast<size_t>(idx)]) {
          normalized = std::clamp(normalized, -1.0f, 1.0f);
        }
        input_buffer[static_cast<size_t>(idx)] =
            (input_tensor_min_ + input_tensor_max_) / 2.0f +
            normalized * (input_tensor_max_ - input_tensor_min_) / 2.0f;
      }
    } else {
      std::cerr << "IOProcessor: unsupported input type: " << input_types_[static_cast<size_t>(key_indice)]
                << std::endl;
      return false;
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "IOProcessor: preProcess failed: " << e.what() << std::endl;
    return false;
  }
}

bool IOProcessor::postProcess(const std::string& key, std::string* out_payload) {
  try {
    if (!out_payload) {
      return false;
    }
    auto it = std::find(output_topics_.begin(), output_topics_.end(), key);
    if (it == output_topics_.end()) {
      std::cerr << "IOProcessor: key not found in output_topics_" << std::endl;
      return false;
    }

    int key_indice = static_cast<int>(it - output_topics_.begin());
    int start_indice = output_tensor_indice_map_[static_cast<size_t>(key_indice)].start_indice;
    int end_indice = output_tensor_indice_map_[static_cast<size_t>(key_indice)].end_indice;
    int dim = end_indice - start_indice + 1;

    if (output_types_[static_cast<size_t>(key_indice)] != "float32") {
      std::cerr << "IOProcessor: unsupported output type" << std::endl;
      return false;
    }

    std::vector<float> physical(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
      int idx = start_indice + i;
      float tensor_val = output_buffer[static_cast<size_t>(idx)];
      float norm = (tensor_val - (output_tensor_min_ + output_tensor_max_) / 2.0f) * 2.0f /
                   (output_tensor_max_ - output_tensor_min_);
      if (output_use_clips_[static_cast<size_t>(idx)]) {
        norm = std::clamp(norm, -1.0f, 1.0f);
      }
      physical[static_cast<size_t>(i)] =
          norm * output_scales_[static_cast<size_t>(idx)] + output_offsets_[static_cast<size_t>(idx)];
    }

    out_payload->resize(static_cast<size_t>(dim) * sizeof(float));
    std::memcpy(out_payload->data(), physical.data(), out_payload->size());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "IOProcessor: postProcess failed: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace inference
}  // namespace robo_lab
