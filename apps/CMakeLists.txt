add_executable(robo_lab_main main.cpp)

target_link_libraries(robo_lab_main PRIVATE robo_lab)

add_dependencies(robo_lab_main franka_plugin)

target_compile_features(robo_lab_main PRIVATE cxx_std_17)
