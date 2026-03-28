#include "inference/inference_interface.h"

#include <NvInfer.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace robo_lab {
namespace inference {

namespace {

std::vector<char> readEngineFile(char const* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::string("failed to open engine file: ") + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error(std::string("failed to read engine file: ") + path);
  }
  return buffer;
}

size_t volume(nvinfer1::Dims const& d) {
  if (d.nbDims < 0) {
    return 0;
  }
  size_t v = 1;
  for (int32_t i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) {
      return 0;
    }
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}

void* allocHostTensor(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 256, bytes) != 0) {
    return nullptr;
  }
  std::memset(p, 0, bytes);
  return p;
}

bool checkCuda(cudaError_t e, char const* what) {
  if (e != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(e) << std::endl;
    return false;
  }
  return true;
}

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << std::endl;
    }
  }
};

TrtLogger g_trt_logger{};

size_t trtElementSizeBytes(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT4:
    // case nvinfer1::DataType::kFP4:
      return 1;
    default:
      return 0;
  }
}

}  // namespace

TRTSession::TRTSession() = default;

TRTSession::~TRTSession() { destroy(); }

void TRTSession::freeBuffers() {
  for (void* p : deviceBuffers_) {
    if (p) {
      cudaFree(p);
    }
  }
  deviceBuffers_.clear();

  for (void* p : hostBuffers_) {
    if (p) {
      std::free(p);
    }
  }
  hostBuffers_.clear();

  for (void* p : pinnedHostBuffers_) {
    if (p) {
      cudaFreeHost(p);
    }
  }
  pinnedHostBuffers_.clear();

  inputStaging_.clear();
  outputStaging_.clear();
}

bool TRTSession::destroy() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
  }
  context_.reset();
  freeBuffers();
  engine_.reset();
  runtime_.reset();
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  return true;
}

bool TRTSession::loadEngine(const std::string& engine_path) {
  std::vector<char> engineBlob;
  try {
    engineBlob = readEngineFile(engine_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  if (engineBlob.empty()) {
    std::cerr << "failed to read engine file: " << engine_path << std::endl;
    return false;
  }

  runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
  if (!runtime_) {
    std::cerr << "failed to create runtime" << std::endl;
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engineBlob.data(), engineBlob.size()));
  if (!engine_) {
    std::cerr << "failed to deserialize engine" << std::endl;
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "failed to create execution context" << std::endl;
    return false;
  }

  if (!checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate")) {
    return false;
  }

  if (!findDynamicalInputTensors()) {
    std::cerr << "failed to find dynamical input tensors" << std::endl;
    return false;
  }
  if (!buildIOStagings()) {
    std::cerr << "failed to build IO stagings" << std::endl;
    return false;
  }

  return true;
}

bool TRTSession::findDynamicalInputTensors() {
  try {
    int32_t const nbTensors = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nbTensors; ++i) {
      char const* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) {
        continue;
      }
      nvinfer1::Dims shape = context_->getTensorShape(name);
      bool dynamic = false;
      for (int32_t j = 0; j < shape.nbDims; ++j) {
        if (shape.d[j] < 0) {
          dynamic = true;
          break;
        }
      }
      if (dynamic) {
        nvinfer1::Dims opt = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
        if (!context_->setInputShape(name, opt)) {
          std::cerr << "setInputShape failed for tensor " << name << std::endl;
          return false;
        }
      }
    }
  } catch (std::exception const& e) {
    std::cerr << "findDynamicalInputTensors: " << e.what() << std::endl;
    return false;
  }
  return true;
}

bool TRTSession::buildIOStagings() {
  try {
    int32_t const nbTensors = engine_->getNbIOTensors();
    std::vector<void*> ioPtr(static_cast<size_t>(nbTensors));
    std::vector<size_t> ioNbytes(static_cast<size_t>(nbTensors));

    for (int32_t i = 0; i < nbTensors; ++i) {
      char const* name = engine_->getIOTensorName(i);
      nvinfer1::Dims shape = context_->getTensorShape(name);
      size_t const elements = volume(shape);
      nvinfer1::DataType const dtype = engine_->getTensorDataType(name);
      size_t const bytesPerEl = trtElementSizeBytes(dtype);
      if (elements == 0 || bytesPerEl == 0) {
        std::cerr << "invalid tensor shape or dtype for " << name << std::endl;
        return false;
      }

      size_t const nbytes = elements * bytesPerEl;
      ioNbytes[static_cast<size_t>(i)] = nbytes;
      nvinfer1::TensorLocation const loc = engine_->getTensorLocation(name);
      void* buf = nullptr;

      if (loc == nvinfer1::TensorLocation::kDEVICE) {
        if (!checkCuda(cudaMalloc(&buf, nbytes), "cudaMalloc")) {
          return false;
        }
        if (!checkCuda(cudaMemsetAsync(buf, 0, nbytes, stream_), "cudaMemsetAsync")) {
          return false;
        }
        deviceBuffers_.push_back(buf);
      } else {
        buf = allocHostTensor(nbytes);
        if (!buf) {
          std::cerr << "host allocation failed" << std::endl;
          return false;
        }
        hostBuffers_.push_back(buf);
      }
      ioPtr[static_cast<size_t>(i)] = buf;
      if (!context_->setTensorAddress(name, buf)) {
        std::cerr << "setTensorAddress failed for " << name << std::endl;
        return false;
      }
    }

    for (int32_t i = 0; i < nbTensors; ++i) {
      char const* name = engine_->getIOTensorName(i);
      nvinfer1::TensorIOMode const mode = engine_->getTensorIOMode(name);
      nvinfer1::TensorLocation const loc = engine_->getTensorLocation(name);
      size_t const nbytes = ioNbytes[static_cast<size_t>(i)];
      void* const devOrHostBinding = ioPtr[static_cast<size_t>(i)];
      if (loc != nvinfer1::TensorLocation::kDEVICE) {
        continue;
      }
      void* pinned = nullptr;
      if (!checkCuda(cudaHostAlloc(&pinned, nbytes, cudaHostAllocDefault), "cudaHostAlloc")) {
        return false;
      }
      pinnedHostBuffers_.push_back(pinned);
      std::memset(pinned, 0, nbytes);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        std::cout << "buildIOStagings: input tensor " << name << " " << pinned << " " << devOrHostBinding << " " << nbytes << std::endl;
        inputStaging_.push_back(TrtIoStaging{std::string(name), pinned, devOrHostBinding, nbytes});
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        std::cout << "buildIOStagings: output tensor " << name << " " << pinned << " " << devOrHostBinding << " " << nbytes << std::endl;
        outputStaging_.push_back(TrtIoStaging{std::string(name), pinned, devOrHostBinding, nbytes});
      }
    }
    return true;
  } catch (std::exception const& e) {
    std::cerr << "buildIOStagings: " << e.what() << std::endl;
    return false;
  }
}

bool TRTSession::writeToInputStaging(const std::vector<float>& input) {
  if (inputStaging_.empty()) {
    std::cerr << "TRTSession: no input tensors" << std::endl;
    return false;
  }
  size_t total_floats = 0;
  for (const auto& s : inputStaging_) {
    if (s.nbytes % sizeof(float) != 0) {
      std::cerr << "TRTSession: input tensor size not float-aligned" << std::endl;
      return false;
    }
    total_floats += s.nbytes / sizeof(float);
  }
  if (input.size() != total_floats) {
    std::cerr << "TRTSession: input float count mismatch (got " << input.size() << ", need " << total_floats << ")"
              << std::endl;
    return false;
  }
  size_t offset = 0;
  for (const auto& s : inputStaging_) {
    size_t nfloat = s.nbytes / sizeof(float);
    // std::cout << "TRTSession: writeToInputStaging: " << s.name << " " << s.hostPinned << " " << s.devicePtr << " " << s.nbytes << std::endl;
    // std::cout << "TRTSession: writeToInputStaging: input.data() + offset = " << input.data() + offset << std::endl;
    // std::cout << "TRTSession: writeToInputStaging: input.data() + offset + s.nbytes = " << input.data() + offset + s.nbytes << std::endl;
    std::memcpy(s.hostPinned, input.data() + offset, s.nbytes);
    if (!checkCuda(
            cudaMemcpyAsync(s.devicePtr, s.hostPinned, s.nbytes, cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync H2D input")) {
      return false;
    }
    offset += nfloat;
  }
  return true;
}

bool TRTSession::AsynInfer() {
  if (!context_) {
    return false;
  }
  if (!context_->enqueueV3(stream_)) {
    std::cerr << "TRTSession: enqueueV3 failed" << std::endl;
    return false;
  }
  // std::cout << "TRTSession: enqueueV3 success" << std::endl;
  return true;
}

bool TRTSession::readFromOutputStaging(std::vector<float>& output) {
  if (outputStaging_.empty()) {
    output.clear();
    return true;
  }
  size_t total_floats = 0;
  for (const auto& s : outputStaging_) {
    if (s.nbytes % sizeof(float) != 0) {
      std::cerr << "TRTSession: output tensor size not float-aligned" << std::endl;
      return false;
    }
    total_floats += s.nbytes / sizeof(float);
  }
  output.resize(total_floats);
  for (const auto& s : outputStaging_) {
    if (!checkCuda(cudaMemcpy(s.hostPinned, s.devicePtr, s.nbytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) {
      return false;
    }
  }
  size_t offset = 0;
  for (const auto& s : outputStaging_) {
    size_t nfloat = s.nbytes / sizeof(float);
    std::memcpy(output.data() + offset, s.hostPinned, s.nbytes);
    offset += nfloat;
  }
  return true;
}

bool TRTSession::synchronize() {
  if (!stream_) {
    return true;
  }
  return checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace inference
}  // namespace robo_lab
