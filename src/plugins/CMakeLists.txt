add_library(franka_plugin MODULE franka_plugin.cpp)

target_include_directories(franka_plugin PRIVATE "${CMAKE_SOURCE_DIR}/include")

set_target_properties(
  franka_plugin
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
             OUTPUT_NAME "franka_plugin"
             POSITION_INDEPENDENT_CODE ON)

target_compile_features(franka_plugin PRIVATE cxx_std_17)
