find_package(Threads REQUIRED)

add_library(
  robo_lab STATIC
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

add_subdirectory(plugins)
