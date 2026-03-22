cmake_minimum_required(VERSION 3.16)

project(roboLab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C++ code uses [zenoh-c](https://github.com/eclipse-zenoh/zenoh-c) (CMake + zenoh.h), not the Rust `zenoh` repo.
# Leave ROBOLAB_ZENOHC_ROOT empty to auto-pick the first existing path below.
set(ROBOLAB_ZENOHC_ROOT
    ""
    CACHE
      PATH
      "Path to zenoh-c sources (empty = auto: third_party/zenoh-c, then third_party/zenoh)")

set(_robo_zenohc_src "${ROBOLAB_ZENOHC_ROOT}")
if(_robo_zenohc_src STREQUAL "")
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/zenoh-c/CMakeLists.txt")
    set(_robo_zenohc_src "${CMAKE_SOURCE_DIR}/third_party/zenoh-c")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/zenoh/CMakeLists.txt")
    set(_robo_zenohc_src "${CMAKE_SOURCE_DIR}/third_party/zenoh")
  endif()
endif()

find_package(zenohc CONFIG QUIET)

if(zenohc_FOUND)
  # Installed/config-mode zenoh-c (CMAKE_PREFIX_PATH, etc.).
elseif(EXISTS "${_robo_zenohc_src}/CMakeLists.txt")
  set(ZENOHC_BUILD_IN_SOURCE_TREE
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory("${_robo_zenohc_src}" "${CMAKE_BINARY_DIR}/third_party/zenohc-build")
  if(NOT TARGET zenohc::lib)
    message(
      FATAL_ERROR
        "zenoh-c at ${_robo_zenohc_src} did not define target zenohc::lib")
  endif()
else()
  message(
    FATAL_ERROR
      "zenoh-c not found for C/C++ (target zenohc::lib).\n"
      "  The Rust tree at third_party/zenoh is not the same as zenoh-c.\n"
      "  Add zenoh-c next to it, e.g. clone into third_party/zenoh-c:\n"
      "    git clone --depth 1 --branch 1.8.0 https://github.com/eclipse-zenoh/zenoh-c.git third_party/zenoh-c\n"
      "  Or install zenohc and set CMAKE_PREFIX_PATH, or set -DROBOLAB_ZENOHC_ROOT=/path/to/zenoh-c")
endif()

# Protobuf: every *.proto under proto/ is compiled automatically (no per-file CMake edits).
find_package(Protobuf REQUIRED)
file(
  GLOB_RECURSE ROBO_LAB_PROTO_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/proto/*.proto")
list(SORT ROBO_LAB_PROTO_FILES)
if(NOT ROBO_LAB_PROTO_FILES)
  message(FATAL_ERROR "No .proto files under ${CMAKE_SOURCE_DIR}/proto")
endif()
protobuf_generate_cpp(ROBO_LAB_PROTO_SRCS ROBO_LAB_PROTO_HDRS ${ROBO_LAB_PROTO_FILES})
add_library(robo_lab_proto STATIC ${ROBO_LAB_PROTO_SRCS} ${ROBO_LAB_PROTO_HDRS})
target_include_directories(robo_lab_proto PUBLIC "${CMAKE_BINARY_DIR}")
set_target_properties(robo_lab_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET protobuf::libprotobuf)
  target_link_libraries(robo_lab_proto PUBLIC protobuf::libprotobuf)
else()
  target_include_directories(robo_lab_proto PUBLIC ${PROTOBUF_INCLUDE_DIRS})
  target_link_libraries(robo_lab_proto PUBLIC ${PROTOBUF_LIBRARIES})
endif()

# Third-party libs before plugins so targets (Franka::Franka, freenect2) exist when src/plugins links them.
add_subdirectory(third_party/libfranka-0.9.0)

# libfreenect2 defaults ENABLE_CUDA=ON; nvcc then needs a working host C++ compiler (g++/cc1plus).
# Turn CUDA off here unless you explicitly opt in (avoids "cannot execute cc1plus" when g++ isn't set up for nvcc).
option(ROBOLAB_LIBFREENECT2_CUDA "Build bundled libfreenect2 with CUDA depth processor" ON)
set(ENABLE_CUDA
    ${ROBOLAB_LIBFREENECT2_CUDA}
    CACHE BOOL "libfreenect2: CUDA support" FORCE)

add_subdirectory(third_party/libfreenect2)

add_subdirectory(src)
add_subdirectory(apps)
