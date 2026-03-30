# yaml-cpp is resolved in parent src/CMakeLists.txt

# Franka plugin
add_library(franka_plugin MODULE franka_plugin.cpp)

# Eigen
find_package(Eigen3 REQUIRED)

target_include_directories(franka_plugin PRIVATE 
"${CMAKE_SOURCE_DIR}/include"
"${EIGEN3_INCLUDE_DIRS}")

target_link_libraries(franka_plugin PRIVATE 
        robo_lab 
        robo_lab_proto 
        yaml-cpp::yaml-cpp 
        Franka::Franka
        Eigen3::Eigen
        )

# Plugin is dlopen'd; it must resolve MessageSystem etc. from the same librobo_lab.so as the executable.
set_target_properties(
  franka_plugin
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
             OUTPUT_NAME "franka_plugin"
             POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "${CMAKE_BINARY_DIR}/src;$<TARGET_FILE_DIR:franka>")

target_compile_features(franka_plugin PRIVATE cxx_std_17)

# 
# Kinect plugin
add_library(
  kinect_plugin
  MODULE
  kinect_plugin.cpp
  ${CMAKE_SOURCE_DIR}/third_party/libfreenect2/examples/viewer.cpp
  ${CMAKE_SOURCE_DIR}/third_party/libfreenect2/src/flextGL.cpp
)
# libfreenect2 uses directory-scoped include_directories(); that often does not populate
# INTERFACE_INCLUDE_DIRECTORIES, so linking freenect2 alone may not add its headers for consumers.
# what a piece of shit 
# lucky dudes who are using realsense
set(_robo_freenect2_src "${CMAKE_SOURCE_DIR}/third_party/libfreenect2")
set(_robo_freenect2_bin "${CMAKE_BINARY_DIR}/third_party/libfreenect2")
target_include_directories(
  kinect_plugin
  PRIVATE "${CMAKE_SOURCE_DIR}/include"
          "${_robo_freenect2_src}/include"
          "${_robo_freenect2_src}/src"
          "${_robo_freenect2_bin}")

target_link_libraries(kinect_plugin PRIVATE robo_lab robo_lab_proto yaml-cpp::yaml-cpp freenect2)

set_target_properties(
  kinect_plugin
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
             OUTPUT_NAME "kinect_plugin"
             POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "${CMAKE_BINARY_DIR}/src;$<TARGET_FILE_DIR:freenect2>")

target_compile_features(kinect_plugin PRIVATE cxx_std_17)

# Realsense plugin
if(TARGET realsense2)
  add_library(realsense_plugin MODULE realsense_plugin.cpp)

  target_include_directories(realsense_plugin PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(realsense_plugin PRIVATE robo_lab robo_lab_proto yaml-cpp::yaml-cpp realsense2)

  set_target_properties(
    realsense_plugin
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
               OUTPUT_NAME "realsense_plugin"
               POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "${CMAKE_BINARY_DIR}/src;$<TARGET_FILE_DIR:realsense2>")

  target_compile_features(realsense_plugin PRIVATE cxx_std_17)
endif()

# Demo: TensorRT inference bridge (Zenoh float32 payloads)
add_library(demo_inference_plugin MODULE demo_inference_plugin.cpp)

target_include_directories(demo_inference_plugin PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(
  demo_inference_plugin
  PRIVATE robo_lab
          robo_lab_proto
          yaml-cpp::yaml-cpp
          robo_lab_inference)

set(_robo_trt_lib "${CMAKE_SOURCE_DIR}/third_party/TensorRT-10.2.0.19/lib")
set_target_properties(
  demo_inference_plugin
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
             OUTPUT_NAME "demo_inference_plugin"
             POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "${CMAKE_BINARY_DIR}/src;${_robo_trt_lib}")

target_compile_features(demo_inference_plugin PRIVATE cxx_std_17)