find_package(GTest REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(plugin_load_test plugin_load_test.cpp)
target_include_directories(plugin_load_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(plugin_load_test PRIVATE ${CMAKE_DL_LIBS})
set(_plugins)
foreach(_plugin franka_plugin kinect_plugin realsense_plugin demo_inference_plugin
    cartesian_control_plugin recorder_plugin franka_replayer_plugin keyboard_plugin
    imu_teleop_plugin n100_imu_plugin ros_bridge_plugin)
  if(TARGET ${_plugin})
    list(APPEND _plugins "$<TARGET_FILE:${_plugin}>")
    add_dependencies(plugin_load_test ${_plugin})
  endif()
endforeach()
add_test(NAME plugin_load COMMAND plugin_load_test ${_plugins})

add_executable(record_replay_test record_replay_test.cpp ../src/plugins/franka_replayer.cpp)
target_include_directories(record_replay_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(record_replay_test PRIVATE robo_lab robo_lab_proto GTest::gtest_main ${CMAKE_DL_LIBS})
find_package(HDF5 REQUIRED COMPONENTS C)
target_include_directories(record_replay_test PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(record_replay_test PRIVATE ${HDF5_LIBRARIES})
target_compile_definitions(record_replay_test PRIVATE
  RECORDER_LIBRARY="$<TARGET_FILE:recorder_plugin>"
  REPLAYER_LIBRARY="$<TARGET_FILE:franka_replayer_plugin>")
add_dependencies(record_replay_test recorder_plugin franka_replayer_plugin)
add_test(NAME record_replay COMMAND record_replay_test)
set_tests_properties(record_replay PROPERTIES TIMEOUT 60)

if(ROBOLAB_BUILD_ROS_BRIDGE)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  add_executable(ros_bridge_test ros_bridge_test.cpp)
  target_include_directories(ros_bridge_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(ros_bridge_test robo_lab robo_lab_proto GTest::gtest_main ${CMAKE_DL_LIBS})
  ament_target_dependencies(ros_bridge_test rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs)
  target_compile_definitions(ros_bridge_test PRIVATE BRIDGE_LIBRARY="$<TARGET_FILE:ros_bridge_plugin>")
  add_dependencies(ros_bridge_test ros_bridge_plugin)
  add_test(NAME ros_bridge COMMAND ros_bridge_test)
  add_test(NAME bridge_shutdown COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bridge_shutdown_test.py"
    "$<TARGET_FILE:robo_lab_main>" "$<TARGET_FILE:ros_bridge_plugin>")
  set_tests_properties(ros_bridge bridge_shutdown PROPERTIES TIMEOUT 90
    ENVIRONMENT "ROS_LOCALHOST_ONLY=1;ROS_DOMAIN_ID=197")
endif()
