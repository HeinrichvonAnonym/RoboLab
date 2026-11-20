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
    GIT_TAG 0.8.0)
  FetchContent_MakeAvailable(yaml-cpp)
endif()
if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()

if(ROBOLAB_BUILD_TENSORRT)
  add_subdirectory(inference)
endif()
if(TARGET realsense2)
  find_package(Eigen3 REQUIRED)
  add_library(robo_rgbd STATIC camera/rgbd.cpp)
  target_include_directories(robo_rgbd PUBLIC "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(robo_rgbd PUBLIC robo_lab_proto Eigen3::Eigen yaml-cpp::yaml-cpp realsense2)
  set_target_properties(robo_rgbd PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_subdirectory(cartesian_control)
add_subdirectory(plugins)
