#pragma once

#include <kinect.pb.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace robo_lab {
namespace kinect_detail {

struct SampledImage {
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;  // Existing protocol: bytes per pixel, including float32.
  int32_t step = 0;
  int32_t proto_type = 0;
  std::string data;
};

// Nearest-neighbor byte copy. Float values (including NaN payloads) and RGB
// byte order are never interpreted. The destination owns all published bytes.
inline void sample_image(const unsigned char* source, size_t width, size_t height,
                         size_t bytes_per_pixel, int factor, int32_t proto_type,
                         SampledImage& output) {
  if (factor <= 0) {
    throw std::invalid_argument("downsample factor must be a positive integer");
  }
  if (!source || width == 0 || height == 0 || bytes_per_pixel == 0) {
    throw std::invalid_argument("empty Kinect image");
  }
  const size_t max_size = std::numeric_limits<size_t>::max();
  if (width > max_size / bytes_per_pixel || height > max_size / (width * bytes_per_pixel)) {
    throw std::overflow_error("Kinect image byte count overflow");
  }
  const size_t n = static_cast<size_t>(factor);
  const size_t out_width = 1 + (width - 1) / n;
  const size_t out_height = 1 + (height - 1) / n;
  const size_t out_step = out_width * bytes_per_pixel;
  const size_t max_field = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (out_width > max_field || out_height > max_field ||
      bytes_per_pixel > max_field || out_step > max_field) {
    throw std::overflow_error("Kinect image dimensions exceed protobuf fields");
  }

  output.data.resize(out_step * out_height);  // Retains storage for subsequent frames.
  if (factor == 1) {
    std::memcpy(&output.data[0], source, output.data.size());
  } else {
    for (size_t y = 0; y < out_height; ++y) {
      const unsigned char* row = source + y * n * width * bytes_per_pixel;
      char* destination = &output.data[y * out_step];
      for (size_t x = 0; x < out_width; ++x) {
        const unsigned char* pixel = row + x * n * bytes_per_pixel;
        // All three native Kinect streams have four-byte pixels. A fixed-size
        // memcpy avoids a function call per pixel even in unoptimized builds.
        if (bytes_per_pixel == 4) {
          std::memcpy(destination + x * 4, pixel, 4);
        } else {
          std::memcpy(destination + x * bytes_per_pixel, pixel, bytes_per_pixel);
        }
      }
    }
  }
  output.width = static_cast<int32_t>(out_width);
  output.height = static_cast<int32_t>(out_height);
  output.channels = static_cast<int32_t>(bytes_per_pixel);
  output.step = static_cast<int32_t>(out_step);
  output.proto_type = proto_type;
}

inline void serialize_image(const SampledImage& image, std::string& payload) {
  kinect::rgbImage message;
  message.set_width(image.width);
  message.set_height(image.height);
  message.set_channels(image.channels);
  message.set_step(image.step);
  message.set_type(image.proto_type);
  message.set_image(image.data.data(), image.data.size());
  if (!message.SerializeToString(&payload)) {
    throw std::runtime_error("Kinect image serialization failed");
  }
}

}  // namespace kinect_detail
}  // namespace robo_lab
