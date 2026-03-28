set(ROBOLAB_CUDA_ROOT
    "/usr/local/cuda-11.8"
    CACHE PATH "CUDA toolkit root for TensorRT inference")
set(ROBOLAB_TENSORRT_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/TensorRT-10.2.0.19"
    CACHE PATH "TensorRT root (headers + libnvinfer)")

find_library(
  ROBOLAB_CUDART
  NAMES cudart
  PATHS "${ROBOLAB_CUDA_ROOT}/lib64" "${ROBOLAB_CUDA_ROOT}/lib"
  NO_DEFAULT_PATH)
if(NOT ROBOLAB_CUDART)
  find_library(ROBOLAB_CUDART NAMES cudart)
endif()

find_library(
  ROBOLAB_NVINFER
  NAMES nvinfer
  PATHS "${ROBOLAB_TENSORRT_ROOT}/lib" "${ROBOLAB_TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib"
  NO_DEFAULT_PATH)
if(NOT ROBOLAB_NVINFER)
  find_library(ROBOLAB_NVINFER NAMES nvinfer)
endif()

if(NOT ROBOLAB_CUDART)
  message(FATAL_ERROR "libcudart not found (set ROBOLAB_CUDA_ROOT, default ${ROBOLAB_CUDA_ROOT})")
endif()
if(NOT ROBOLAB_NVINFER)
  message(
    FATAL_ERROR
      "libnvinfer not found under ${ROBOLAB_TENSORRT_ROOT} (set ROBOLAB_TENSORRT_ROOT)")
endif()

add_library(
  robo_lab_inference STATIC
  inference_interface.cpp
  io_processor.cpp
  trt_session.cpp)

# Required when this static archive is linked into a shared module (plugins/*.so).
set_target_properties(robo_lab_inference PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  robo_lab_inference
  PUBLIC "${CMAKE_SOURCE_DIR}/include"
         "${ROBOLAB_TENSORRT_ROOT}/include"
         "${ROBOLAB_CUDA_ROOT}/include")

target_link_libraries(
  robo_lab_inference
  PUBLIC yaml-cpp::yaml-cpp
  PUBLIC ${ROBOLAB_NVINFER} ${ROBOLAB_CUDART})

target_compile_features(robo_lab_inference PUBLIC cxx_std_17)
