find_package(Threads REQUIRED)

add_library(
  robo_lab SHARED
  robo_lab.cpp
  message_system.cpp
  framework.cpp)

target_include_directories(
  robo_lab
  PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(
  robo_lab
  PUBLIC zenohc::lib
  PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

target_compile_features(robo_lab PUBLIC cxx_std_17)

# yaml-cpp before inference/plugins (both need yaml-cpp::yaml-cpp)
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

add_subdirectory(inference)
add_subdirectory(plugins)
