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
# TensorRT does not affect this.
#
# Common failure: Anaconda protoc is first on PATH / cached as Protobuf_PROTOC_EXECUTABLE, so
# generated *.pb.h reference google/protobuf/port_def.inc (protobuf 3.21+), while
# find_package links system libprotobuf 3.6.x from /usr — those headers have no port_def.inc.
# Fix: use the same protoc family as libprotobuf (prefer system /usr/bin/protoc), or upgrade
# libprotobuf-dev, or use conda for both headers and library (CMAKE_PREFIX_PATH).
set(ROBOLAB_PROTOC_EXECUTABLE
    ""
    CACHE FILEPATH "Override protoc for .pb codegen (must match libprotobuf C++ headers)")
option(ROBOLAB_PREFER_SYSTEM_PROTOC
       "Prefer /usr/bin/protoc over conda so codegen matches distro libprotobuf-dev" ON)
if(ROBOLAB_PROTOC_EXECUTABLE)
  set(Protobuf_PROTOC_EXECUTABLE "${ROBOLAB_PROTOC_EXECUTABLE}" CACHE FILEPATH "" FORCE)
elseif(ROBOLAB_PREFER_SYSTEM_PROTOC)
  find_program(_ROBO_SYS_PROTOC protoc PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
  if(_ROBO_SYS_PROTOC)
    set(Protobuf_PROTOC_EXECUTABLE "${_ROBO_SYS_PROTOC}" CACHE FILEPATH "" FORCE)
  endif()
endif()

find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()

file(
  GLOB_RECURSE ROBO_LAB_PROTO_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/proto/*.proto")
list(SORT ROBO_LAB_PROTO_FILES)
if(NOT ROBO_LAB_PROTO_FILES)
  message(FATAL_ERROR "No .proto files under ${CMAKE_SOURCE_DIR}/proto")
endif()
protobuf_generate_cpp(ROBO_LAB_PROTO_SRCS ROBO_LAB_PROTO_HDRS ${ROBO_LAB_PROTO_FILES})

# FindProtobuf uses Protobuf_INCLUDE_DIR (singular) and/or Protobuf_INCLUDE_DIRS.
set(_ROBO_PROTO_INCLUDES "")
if(DEFINED Protobuf_INCLUDE_DIRS AND Protobuf_INCLUDE_DIRS)
  list(APPEND _ROBO_PROTO_INCLUDES ${Protobuf_INCLUDE_DIRS})
endif()
if(DEFINED Protobuf_INCLUDE_DIR AND Protobuf_INCLUDE_DIR)
  list(APPEND _ROBO_PROTO_INCLUDES "${Protobuf_INCLUDE_DIR}")
endif()
if(DEFINED PROTOBUF_INCLUDE_DIRS AND PROTOBUF_INCLUDE_DIRS)
  list(APPEND _ROBO_PROTO_INCLUDES ${PROTOBUF_INCLUDE_DIRS})
endif()
if(TARGET protobuf::libprotobuf)
  get_target_property(_robo_pb_iface_inc protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
  if(_robo_pb_iface_inc AND NOT _robo_pb_iface_inc MATCHES "-NOTFOUND$")
    list(APPEND _ROBO_PROTO_INCLUDES ${_robo_pb_iface_inc})
  endif()
endif()
if(NOT _ROBO_PROTO_INCLUDES)
  find_path(
    _ROBO_PROTOBUF_PARENT_DIR
    google/protobuf/message.h
    PATHS /usr/include
          /usr/local/include
          "$ENV{CONDA_PREFIX}/include"
          "$ENV{HOME}/miniconda3/include"
          "$ENV{HOME}/anaconda3/include")
  if(_ROBO_PROTOBUF_PARENT_DIR)
    list(APPEND _ROBO_PROTO_INCLUDES "${_ROBO_PROTOBUF_PARENT_DIR}")
  endif()
endif()
if(_ROBO_PROTO_INCLUDES)
  list(REMOVE_DUPLICATES _ROBO_PROTO_INCLUDES)
endif()

# Fail fast if generated headers need port_def.inc but no include tree provides it.
set(_ROBO_PB_SAMPLE "${CMAKE_BINARY_DIR}/franka.pb.h")
if(NOT EXISTS "${_ROBO_PB_SAMPLE}")
  set(_ROBO_PB_SAMPLE "${CMAKE_BINARY_DIR}/demo_inference.pb.h")
endif()
if(EXISTS "${_ROBO_PB_SAMPLE}")
  file(READ "${_ROBO_PB_SAMPLE}" LIMIT 800 _ROBO_PB_SNIP)
  if(_ROBO_PB_SNIP MATCHES "port_def\\.inc")
    set(_ROBO_PORT_DEF_OK OFF)
    foreach(_d IN LISTS _ROBO_PROTO_INCLUDES)
      if(EXISTS "${_d}/google/protobuf/port_def.inc")
        set(_ROBO_PORT_DEF_OK ON)
        break()
      endif()
    endforeach()
    if(NOT _ROBO_PORT_DEF_OK)
      message(
        FATAL_ERROR
          "Protobuf mismatch: ${Protobuf_PROTOC_EXECUTABLE} generated code that needs "
          "google/protobuf/port_def.inc, but installed C++ headers under ${_ROBO_PROTO_INCLUDES} "
          "are too old (typical: conda protoc + system libprotobuf-dev).\n"
          "  Fix: reconfigure with -DROBOLAB_PREFER_SYSTEM_PROTOC=ON (default) and delete "
          "build/*.pb.cc build/*.pb.h, OR upgrade libprotobuf-dev, OR use conda for both "
          "(set CMAKE_PREFIX_PATH and -DROBOLAB_PROTOC_EXECUTABLE=.../protoc).")
    endif()
  endif()
endif()

add_library(robo_lab_proto SHARED ${ROBO_LAB_PROTO_SRCS} ${ROBO_LAB_PROTO_HDRS})
target_include_directories(robo_lab_proto PUBLIC "${CMAKE_BINARY_DIR}")
if(_ROBO_PROTO_INCLUDES)
  target_include_directories(robo_lab_proto PUBLIC ${_ROBO_PROTO_INCLUDES})
endif()
set_target_properties(
  robo_lab_proto
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             BUILD_RPATH "${CMAKE_BINARY_DIR}/src")
if(TARGET protobuf::libprotobuf)
  target_link_libraries(robo_lab_proto PUBLIC protobuf::libprotobuf)
elseif(Protobuf_LIBRARIES)
  target_link_libraries(robo_lab_proto PUBLIC ${Protobuf_LIBRARIES})
else()
  target_link_libraries(robo_lab_proto PUBLIC ${PROTOBUF_LIBRARIES})
endif()
if(NOT _ROBO_PROTO_INCLUDES)
  message(
    FATAL_ERROR
      "Protobuf C++ headers not found. Install libprotobuf-dev or set Protobuf_INCLUDE_DIR.")
endif()

# Third-party libs before plugins so targets (Franka::Franka, freenect2) exist when src/plugins links them.
# Default to library-only third-party builds (no examples/tools/tests executables).
# zenoh-c is intentionally not touched.
option(ROBOLAB_THIRDPARTY_EXECUTABLES "Enable executables in third_party modules (except zenoh/zenoh-c)" OFF)

set(BUILD_TESTS
    ${ROBOLAB_THIRDPARTY_EXECUTABLES}
    CACHE BOOL "libfranka: build tests" FORCE)
set(BUILD_EXAMPLES
    ${ROBOLAB_THIRDPARTY_EXECUTABLES}
    CACHE BOOL "libfranka/libfreenect2/librealsense: build examples" FORCE)
add_subdirectory(third_party/libfranka-0.9.0)

# libfreenect2 defaults ENABLE_CUDA=ON; nvcc then needs a working host C++ compiler (g++/cc1plus).
# Turn CUDA off here unless you explicitly opt in (avoids "cannot execute cc1plus" when g++ isn't set up for nvcc).
option(ROBOLAB_LIBFREENECT2_CUDA "Build bundled libfreenect2 with CUDA depth processor" ON)
set(ENABLE_CUDA
    ${ROBOLAB_LIBFREENECT2_CUDA}
    CACHE BOOL "libfreenect2: CUDA support" FORCE)

add_subdirectory(third_party/libfreenect2)



# librealsense is optional and currently not required for Kinect v2 plugin path.
# Keep it disabled by default to avoid nested-project path assumptions
# (e.g. examples/tools including ${CMAKE_SOURCE_DIR}/CMake/*).
option(ROBOLAB_BUILD_LIBREALSENSE "Build bundled librealsense" ON)
if(ROBOLAB_BUILD_LIBREALSENSE)
  set(BUILD_TOOLS
      ${ROBOLAB_THIRDPARTY_EXECUTABLES}
      CACHE BOOL "librealsense: build tools" FORCE)
  set(BUILD_UNIT_TESTS
      ${ROBOLAB_THIRDPARTY_EXECUTABLES}
      CACHE BOOL "librealsense: build unit tests" FORCE)
  set(BUILD_GRAPHICAL_EXAMPLES
      ${ROBOLAB_THIRDPARTY_EXECUTABLES}
      CACHE BOOL "librealsense: build graphical examples" FORCE)
  set(BUILD_PC_STITCHING
      ${ROBOLAB_THIRDPARTY_EXECUTABLES}
      CACHE BOOL "librealsense: build point-cloud stitching tool" FORCE)
  set(CHECK_FOR_UPDATES
      OFF
      CACHE BOOL "librealsense: online update checks in tools" FORCE)
  set(IMPORT_DEPTH_CAM_FW
      OFF
      CACHE BOOL "librealsense: download embedded FW resources" FORCE)
  add_subdirectory(third_party/librealsense)
endif()

# add_subdirectory(third_party/ugv_sdk)
add_subdirectory(src)
add_subdirectory(apps)
