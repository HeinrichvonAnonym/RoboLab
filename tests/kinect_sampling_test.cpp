#include "plugins/kinect_sampling.h"

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <vector>

using robo_lab::kinect_detail::SampledImage;
using robo_lab::kinect_detail::sample_image;
using robo_lab::kinect_detail::serialize_image;

namespace {

std::vector<unsigned char> pixels(size_t width, size_t height, size_t bpp) {
  std::vector<unsigned char> source(width * height * bpp);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < bpp; ++c) {
        source[(y * width + x) * bpp + c] = static_cast<unsigned char>(y * 31 + x * 7 + c);
      }
    }
  }
  return source;
}

TEST(KinectSampling, NativeCopyOwnsBytes) {
  auto source = pixels(7, 5, 4);
  SampledImage output;
  sample_image(source.data(), 7, 5, 4, 1, 1, output);
  ASSERT_EQ(output.data.size(), source.size());
  EXPECT_EQ(std::memcmp(source.data(), output.data.data(), source.size()), 0);
  source.assign(source.size(), 0);
  EXPECT_NE(static_cast<unsigned char>(output.data[4]), 0);
}

TEST(KinectSampling, FactorsAndOddDimensionsCopyWholePixels) {
  for (const int factor : {1, 2, 3, 20}) {
    for (const size_t bpp : {size_t(1), size_t(3), size_t(4)}) {
      SCOPED_TRACE(::testing::Message() << "factor=" << factor << " bpp=" << bpp);
      const auto source = pixels(7, 5, bpp);
      SampledImage output;
      sample_image(source.data(), 7, 5, bpp, factor, 1, output);
      ASSERT_EQ(output.width, (7 + factor - 1) / factor);
      ASSERT_EQ(output.height, (5 + factor - 1) / factor);
      ASSERT_EQ(output.step, output.width * bpp);
      ASSERT_EQ(output.data.size(), output.step * output.height);
      for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
          EXPECT_EQ(std::memcmp(output.data.data() + y * output.step + x * bpp,
                                source.data() + (y * factor * 7 + x * factor) * bpp,
                                bpp), 0);
        }
      }
    }
  }
}

TEST(KinectSampling, FloatBitsAndInvalidValuesArePreserved) {
  // float32 bit patterns: non-canonical NaN payload, +/-Inf, signed zero,
  // negative invalid depth and an unconverted depth in millimeters.
  const std::array<uint32_t, 6> values = {
      0x7fc12345u, 0x7f800000u, 0xff800000u, 0x80000000u, 0xbf800000u, 0x449a5000u};
  std::array<uint32_t, 11> source{};
  for (size_t x = 0; x < values.size(); ++x) source[x * 2] = values[x];
  for (int type : {2, 4}) {
    SampledImage output;
    sample_image(reinterpret_cast<const unsigned char*>(source.data()), 11, 1, 4, 2, type, output);
    ASSERT_EQ(output.data.size(), values.size() * sizeof(uint32_t));
    EXPECT_EQ(std::memcmp(output.data.data(), values.data(), output.data.size()), 0);
    std::string payload;
    serialize_image(output, payload);
    kinect::rgbImage decoded;
    ASSERT_TRUE(decoded.ParseFromString(payload));
    EXPECT_EQ(decoded.image(), output.data);
    EXPECT_EQ(decoded.type(), type);
    EXPECT_EQ(decoded.channels(), 4);
  }
}

TEST(KinectSampling, BufferStorageIsReusedAndOverwritten) {
  auto source = pixels(1920, 1080, 4);
  SampledImage output;
  sample_image(source.data(), 1920, 1080, 4, 2, 1, output);
  const char* storage = output.data.data();
  const size_t capacity = output.data.capacity();
  source.assign(source.size(), 123);
  for (int i = 0; i < 3; ++i) {
    sample_image(source.data(), 1920, 1080, 4, 2, 1, output);
    EXPECT_EQ(output.data.data(), storage);
    EXPECT_EQ(output.data.capacity(), capacity);
    EXPECT_EQ(output.data, std::string(960 * 540 * 4, 123));
  }
}

TEST(KinectSampling, NativeKinectSizesAndWireMetadata) {
  size_t native_bytes = 0, sampled_bytes = 0;
  for (const auto spec : {std::array<int, 3>{1920, 1080, 1}, {512, 424, 4}, {512, 424, 2}}) {
    const auto source = pixels(spec[0], spec[1], 4);
    for (int factor : {1, 2, 3}) {
      SampledImage output;
      sample_image(source.data(), spec[0], spec[1], 4, factor, spec[2], output);
      std::string payload;
      serialize_image(output, payload);
      kinect::rgbImage decoded;
      ASSERT_TRUE(decoded.ParseFromString(payload));
      EXPECT_EQ(decoded.width(), (spec[0] + factor - 1) / factor);
      EXPECT_EQ(decoded.height(), (spec[1] + factor - 1) / factor);
      EXPECT_EQ(decoded.channels(), 4);
      EXPECT_EQ(decoded.type(), spec[2]);
      EXPECT_EQ(decoded.step(), decoded.width() * 4);
      EXPECT_EQ(decoded.image().size(), decoded.step() * decoded.height());
      EXPECT_EQ(decoded.image(), output.data);
      if (factor == 2) sampled_bytes += decoded.image().size();
    }
    native_bytes += source.size();
  }
  EXPECT_EQ(native_bytes, 10031104u);
  EXPECT_EQ(sampled_bytes * 4, native_bytes);
}

TEST(KinectSampling, InvalidFactorsAndImagesAreRejected) {
  const unsigned char source[4] = {};
  SampledImage output;
  for (int factor : {0, -1, -20}) {
    EXPECT_THROW(sample_image(source, 1, 1, 4, factor, 1, output), std::invalid_argument);
  }
  EXPECT_THROW(sample_image(nullptr, 1, 1, 4, 1, 1, output), std::invalid_argument);
  EXPECT_THROW(sample_image(source, 0, 1, 4, 1, 1, output), std::invalid_argument);
  EXPECT_THROW(sample_image(source, 1, 0, 4, 1, 1, output), std::invalid_argument);
  EXPECT_THROW(sample_image(source, 1, 1, 0, 1, 1, output), std::invalid_argument);
  EXPECT_THROW(sample_image(source, std::numeric_limits<size_t>::max(), 2, 4, 1, 1, output),
               std::overflow_error);
  EXPECT_THROW(sample_image(source, size_t(std::numeric_limits<int32_t>::max()) + 1, 1, 1, 1, 1, output),
               std::overflow_error);
}

}  // namespace

int main(int argc, char** argv) {
  // Export bytes from the actual plugin sampler/serializer for the Python RGB
  // decoder interoperability check; no camera, Zenoh or driver initialization.
  if (argc == 3 && std::string(argv[1]) == "--write-rgb") {
    const auto source = pixels(5, 3, 4);
    SampledImage image;
    sample_image(source.data(), 5, 3, 4, 2, 1, image);
    std::string payload;
    serialize_image(image, payload);
    std::ofstream file(argv[2], std::ios::binary);
    file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    return file.good() ? 0 : 1;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
