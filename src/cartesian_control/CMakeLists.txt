# Cartesian control library
find_package(Eigen3 REQUIRED)

add_library(
  robo_lab_cartesian_control SHARED
  jacobian_calculator.cpp
  cartesian_controller.cpp
)

target_include_directories(
  robo_lab_cartesian_control
  PUBLIC "${CMAKE_SOURCE_DIR}/include"
         "${EIGEN3_INCLUDE_DIRS}"
)

target_link_libraries(
  robo_lab_cartesian_control
  PUBLIC Eigen3::Eigen
)

target_compile_features(robo_lab_cartesian_control PUBLIC cxx_std_17)
