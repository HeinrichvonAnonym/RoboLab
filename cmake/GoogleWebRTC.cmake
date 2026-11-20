# No FetchContent/ExternalProject downloads during configure or build.
find_package(Threads REQUIRED)
set(ROBOLAB_WEBRTC_ROOT "${CMAKE_SOURCE_DIR}/third_party/webrtc/src" CACHE PATH "Pinned Google WebRTC sources")
set(ROBOLAB_WEBRTC_BUILD_DIR "${CMAKE_SOURCE_DIR}/build/third_party/webrtc" CACHE PATH "Explicitly prepared WebRTC GN build")
set(_webrtc_library "${ROBOLAB_WEBRTC_BUILD_DIR}/obj/libwebrtc.a")
if(NOT EXISTS "${_webrtc_library}" OR NOT EXISTS "${ROBOLAB_WEBRTC_BUILD_DIR}/robolab-build.json")
  message(FATAL_ERROR "Google WebRTC is not built. Run: bash scripts/setup_webrtc.sh && bash scripts/build_webrtc.sh")
endif()
execute_process(COMMAND git -C "${ROBOLAB_WEBRTC_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE _webrtc_revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _webrtc_git_result)
file(READ "${ROBOLAB_WEBRTC_BUILD_DIR}/robolab-build.json" _webrtc_manifest)
file(SHA256 "${CMAKE_SOURCE_DIR}/config/webrtc/args.gn" _webrtc_args_hash)
file(SHA256 "${CMAKE_SOURCE_DIR}/config/webrtc/libstdcxx11.patch" _webrtc_patch_hash)
execute_process(COMMAND git -C "${ROBOLAB_WEBRTC_ROOT}" diff --no-ext-diff --binary HEAD
  OUTPUT_VARIABLE _webrtc_local_patch RESULT_VARIABLE _webrtc_diff_result)
string(SHA256 _webrtc_local_patch_hash "${_webrtc_local_patch}")
if(NOT _webrtc_git_result EQUAL 0 OR NOT _webrtc_revision STREQUAL "95554e4a0c5c343fdc989387d9d0ce18e2a2294b"
   OR NOT _webrtc_manifest MATCHES "${_webrtc_revision}" OR NOT _webrtc_manifest MATCHES "${_webrtc_args_hash}"
   OR NOT _webrtc_manifest MATCHES "${_webrtc_patch_hash}" OR NOT _webrtc_diff_result EQUAL 0
   OR NOT _webrtc_local_patch_hash STREQUAL _webrtc_patch_hash)
  message(FATAL_ERROR "WebRTC source/build profile mismatch. Re-run scripts/setup_webrtc.sh and scripts/build_webrtc.sh")
endif()
add_library(RoboLab::WebRTC STATIC IMPORTED GLOBAL)
set_target_properties(RoboLab::WebRTC PROPERTIES
  IMPORTED_LOCATION "${_webrtc_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${ROBOLAB_WEBRTC_ROOT};${ROBOLAB_WEBRTC_ROOT}/third_party/abseil-cpp;${ROBOLAB_WEBRTC_BUILD_DIR}/gen"
  INTERFACE_COMPILE_FEATURES cxx_std_20
  INTERFACE_COMPILE_DEFINITIONS "WEBRTC_POSIX;WEBRTC_LINUX;WEBRTC_USE_H264;WEBRTC_WITHOUT_RUST;NDEBUG;ABSL_ALLOCATOR_NOTHROW=1"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};rt")
set_property(TARGET RoboLab::WebRTC PROPERTY INTERFACE_LINK_OPTIONS "-fuse-ld=lld")
# The pinned headers use Clang constructs rejected by GCC 11. Compile this consumer
# with the same compiler as GN; keep the parent project's compiler unchanged.
set(_webrtc_clang "${ROBOLAB_WEBRTC_ROOT}/third_party/llvm-build/Release+Asserts/bin/clang++")
if(NOT EXISTS "${_webrtc_clang}")
  message(FATAL_ERROR "Pinned Clang missing; run scripts/setup_webrtc.sh")
endif()
get_target_property(_webrtc_includes RoboLab::WebRTC INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_webrtc_definitions RoboLab::WebRTC INTERFACE_COMPILE_DEFINITIONS)
set(_webrtc_compile_flags)
foreach(_include IN LISTS _webrtc_includes)
  list(APPEND _webrtc_compile_flags "-I${_include}")
endforeach()
foreach(_definition IN LISTS _webrtc_definitions)
  list(APPEND _webrtc_compile_flags "-D${_definition}")
endforeach()
set(_webrtc_smoke "${CMAKE_BINARY_DIR}/apps/webrtc_smoke_test")
execute_process(COMMAND /usr/bin/g++ -print-libgcc-file-name OUTPUT_VARIABLE _webrtc_gcc_lib OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _webrtc_gcc_result)
if(NOT _webrtc_gcc_result EQUAL 0)
  message(FATAL_ERROR "The WebRTC system libstdc++ profile requires /usr/bin/g++")
endif()
get_filename_component(_webrtc_gcc_install "${_webrtc_gcc_lib}" DIRECTORY)
add_custom_command(OUTPUT "${_webrtc_smoke}"
  COMMAND "${_webrtc_clang}" "--gcc-install-dir=${_webrtc_gcc_install}" -std=c++20 -stdlib=libstdc++ -O2 -Wno-deprecated-declarations -Wno-nullability-completeness
    ${_webrtc_compile_flags} "${CMAKE_SOURCE_DIR}/tests/webrtc_smoke_test.cpp"
    "${_webrtc_library}" -fuse-ld=lld -pthread -ldl -lrt -o "${_webrtc_smoke}"
  DEPENDS "${CMAKE_SOURCE_DIR}/tests/webrtc_smoke_test.cpp" "${_webrtc_library}"
    "${ROBOLAB_WEBRTC_BUILD_DIR}/robolab-build.json"
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(webrtc_smoke_test ALL DEPENDS "${_webrtc_smoke}")
add_test(NAME webrtc_smoke COMMAND "${_webrtc_smoke}")
set_tests_properties(webrtc_smoke PROPERTIES TIMEOUT 90)
