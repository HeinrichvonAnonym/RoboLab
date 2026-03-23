find_package(yaml-cpp QUIET CONFIG)

if(NOT yaml-cpp_FOUND)
  include(FetchContent)
  set(YAML_CPP_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS
      OFF
      CACHE BOOL "" FORCE)
  set(YAML_BUILD_SHARED_LIBS
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.8.0)
  FetchContent_MakeAvailable(yaml-cpp)
endif()

# Franka plugin
add_library(franka_plugin MODULE franka_plugin.cpp)

target_include_directories(franka_plugin PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(franka_plugin PRIVATE robo_lab robo_lab_proto yaml-cpp::yaml-cpp Franka::Franka)

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