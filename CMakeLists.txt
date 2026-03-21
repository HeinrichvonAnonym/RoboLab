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

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(third_party/libfranka-0.9.0)
