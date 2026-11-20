add_executable(robo_lab_main main.cpp)

target_link_libraries(robo_lab_main PRIVATE robo_lab)

set_target_properties(robo_lab_main PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/src")

if(TARGET franka_plugin)
  add_dependencies(robo_lab_main franka_plugin)
endif()

target_compile_features(robo_lab_main PRIVATE cxx_std_17)

if(ROBOLAB_BUILD_RGBD_PREVIEW)
  if(NOT TARGET robo_rgbd)
    message(FATAL_ERROR "RGB-D preview requires ROBOLAB_BUILD_LIBREALSENSE=ON")
  endif()
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
  add_library(robo_rgbd_renderer STATIC ../src/camera/rgbd_renderer.cpp)
  target_link_libraries(robo_rgbd_renderer PUBLIC robo_rgbd glfw OpenGL::GL)
  add_executable(rgbd_stereo_preview rgbd_stereo_preview.cpp)
  target_link_libraries(rgbd_stereo_preview PRIVATE robo_lab robo_rgbd_renderer)
endif()

# Read-only hardware diagnosis; built explicitly, without adding work to the
# default build. Check arm and gripper separately: libfranka's network error
# text otherwise calls both failures an FCI failure.
if(TARGET Franka::Franka)
  add_executable(franka_connection_check EXCLUDE_FROM_ALL franka_connection_check.cpp)
  target_link_libraries(franka_connection_check PRIVATE Franka::Franka)
  target_compile_features(franka_connection_check PRIVATE cxx_std_17)
  add_executable(robotiq_connection_check EXCLUDE_FROM_ALL robotiq_connection_check.cpp)
  target_link_libraries(robotiq_connection_check PRIVATE robotiq_gripper)
endif()
