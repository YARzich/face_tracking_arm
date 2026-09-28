# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(face_tracking_arm)

# The collision constraints run in the control loop. An unspecified build type
# otherwise compiles them without optimization, consuming most of a 10 ms tick.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_servo REQUIRED)
find_package(osqp CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(ros_gz_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Threads REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackingTarget.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)
rosidl_get_typesupport_target(tracking_typesupport ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Shared C++ defaults; each target below declares its own dependencies explicitly.
function(add_tracking_library target)
  add_library(${target} SHARED ${ARGN})
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_tracking_library(tracking_core src/tracking_core.cpp)
target_link_libraries(tracking_core PUBLIC Eigen3::Eigen)

add_tracking_library(rolling_statistics src/rolling_statistics.cpp)

add_tracking_library(control_time_grid src/control_time_grid.cpp)

add_tracking_library(following_error_policy src/following_error_policy.cpp)

add_tracking_library(published_trajectory_history src/published_trajectory_history.cpp)
target_link_libraries(published_trajectory_history Eigen3::Eigen)
ament_target_dependencies(
  published_trajectory_history
  control_msgs
  moveit_servo
  rclcpp
)

add_tracking_library(hierarchical_velocity_qp src/hierarchical_velocity_qp.cpp)
target_link_libraries(hierarchical_velocity_qp PUBLIC Eigen3::Eigen osqp::osqp)

add_tracking_library(tracking_velocity_task src/tracking_velocity_task.cpp)
target_link_libraries(tracking_velocity_task PUBLIC hierarchical_velocity_qp Eigen3::Eigen)

add_tracking_library(target_motion_estimator src/target_motion_estimator.cpp)
target_link_libraries(target_motion_estimator PUBLIC Eigen3::Eigen)

add_tracking_library(joint_path_follower src/joint_path_follower.cpp)
target_link_libraries(joint_path_follower PUBLIC Eigen3::Eigen)

add_tracking_library(background_path_planner src/background_path_planner.cpp)
target_link_libraries(background_path_planner Eigen3::Eigen Threads::Threads)
ament_target_dependencies(background_path_planner
  moveit_core moveit_ros_planning moveit_msgs rclcpp)

add_tracking_library(emergency_brake_tail src/emergency_brake_tail.cpp)
target_link_libraries(emergency_brake_tail PUBLIC Eigen3::Eigen)

add_tracking_library(collision_constraints
  src/collision_constraints.cpp src/robot_geometry_bounds.cpp)
target_link_libraries(collision_constraints hierarchical_velocity_qp Eigen3::Eigen)
ament_target_dependencies(collision_constraints moveit_core)

add_tracking_library(collision_aware_servo_component
  src/collision_aware_servo_component.cpp
  src/collision_aware_servo_control.cpp
  src/collision_aware_servo_feedback.cpp
  src/collision_aware_servo_trajectory.cpp
  src/collision_aware_servo_recovery.cpp
  src/collision_aware_servo_diagnostics.cpp
  src/collision_aware_parameters.cpp
  src/motion_reference.cpp
  src/idle_search.cpp
)
target_link_libraries(
  collision_aware_servo_component
  collision_constraints
  control_time_grid
  emergency_brake_tail
  following_error_policy
  hierarchical_velocity_qp
  tracking_velocity_task
  target_motion_estimator
  joint_path_follower
  background_path_planner
  published_trajectory_history
  rolling_statistics
  Eigen3::Eigen
  moveit_servo::moveit_servo_lib_cpp
  "${tracking_typesupport}"
)
set_target_properties(
  collision_aware_servo_component
  PROPERTIES INSTALL_RPATH "\$ORIGIN"
)
ament_target_dependencies(
  collision_aware_servo_component
  controller_manager_msgs
  control_msgs
  diagnostic_msgs
  geometry_msgs
  lifecycle_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning
  moveit_servo
  rclcpp
  rclcpp_components
  realtime_tools
  std_srvs
  trajectory_msgs
)
rclcpp_components_register_nodes(
  collision_aware_servo_component
  "face_tracking_arm::CollisionAwareServoComponent"
)

add_tracking_library(face_tracking_component src/face_tracking_component.cpp)
target_link_libraries(face_tracking_component tracking_core Eigen3::Eigen "${tracking_typesupport}")
set_target_properties(
  face_tracking_component
  PROPERTIES INSTALL_RPATH "\$ORIGIN"
)
ament_target_dependencies(
  face_tracking_component
  controller_manager_msgs
  diagnostic_msgs
  geometry_msgs
  moveit_msgs
  rclcpp
  rclcpp_components
  sensor_msgs
  shape_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)
rclcpp_components_register_nodes(
  face_tracking_component
  "face_tracking_arm::FaceTrackingComponent"
)

# Private, ROS-independent scenario math shared by the publisher and unit tests.
add_library(test_face_scenarios STATIC src/test_face_scenarios.cpp)
target_compile_features(test_face_scenarios PUBLIC cxx_std_17)
target_include_directories(test_face_scenarios PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_face_target_publisher src/test_face_target_publisher_node.cpp)
target_compile_features(test_face_target_publisher PUBLIC cxx_std_17)
target_link_libraries(test_face_target_publisher test_face_scenarios)
ament_target_dependencies(
  test_face_target_publisher
  diagnostic_msgs
  geometry_msgs
  rclcpp
)

add_executable(face_target_marker src/face_target_marker_node.cpp)
target_compile_features(face_target_marker PUBLIC cxx_std_17)
ament_target_dependencies(
  face_target_marker
  geometry_msgs
  rclcpp
  ros_gz_interfaces
  tf2
  tf2_geometry_msgs
  tf2_ros
)

add_executable(graceful_component_container src/graceful_component_container.cpp)
target_compile_features(graceful_component_container PUBLIC cxx_std_17)
target_link_libraries(graceful_component_container Threads::Threads)
ament_target_dependencies(graceful_component_container rclcpp rclcpp_components)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS
    test_face_scenarios test_face_target_publisher face_target_marker graceful_component_container
  )
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endforeach()
endif()

# The vendored Lite 6 description keeps its upstream package references intact.
# Generate a build-only integration copy that resolves those references through
# this package without modifying the pinned upstream snapshot.
set(lite6_upstream_xacro
  "${CMAKE_CURRENT_SOURCE_DIR}/description/ufactory_lite6/urdf/lite6/lite6.urdf.xacro"
)
file(READ "${lite6_upstream_xacro}" lite6_integration_xacro)
string(FIND "${lite6_integration_xacro}" "$(find xarm_description)" xarm_reference_position)
if(xarm_reference_position EQUAL -1)
  message(FATAL_ERROR "The upstream Lite 6 Xacro no longer contains the expected xarm_description reference")
endif()
string(
  REPLACE
    "$(find xarm_description)"
    "$(find face_tracking_arm)/description/ufactory_lite6"
    lite6_integration_xacro
    "${lite6_integration_xacro}"
)
set(generated_description_directory "${CMAKE_CURRENT_BINARY_DIR}/generated_description")
file(MAKE_DIRECTORY "${generated_description_directory}")
# These upstream meshes contain repeated triangles. Keep the source snapshot
# intact and preserve the exact surface while reducing FCL traversal work.
set(collision_mesh_generator "${CMAKE_CURRENT_SOURCE_DIR}/scripts/deduplicate_collision_mesh.py")
foreach(link_name IN ITEMS link4 link5)
  set(source_collision_mesh
    "${CMAKE_CURRENT_SOURCE_DIR}/description/ufactory_lite6/meshes/lite6/collision/${link_name}.stl")
  set(generated_collision_mesh
    "${generated_description_directory}/collision/${link_name}.stl")
  add_custom_command(
    OUTPUT "${generated_collision_mesh}"
    COMMAND "${Python3_EXECUTABLE}" "${collision_mesh_generator}"
      "${source_collision_mesh}" "${generated_collision_mesh}"
    DEPENDS "${collision_mesh_generator}" "${source_collision_mesh}"
    VERBATIM
  )
  list(APPEND generated_collision_meshes "${generated_collision_mesh}")
  string(REPLACE
    "mesh_filename=\"lite6/collision/${link_name}\""
    "mesh_filename=\"../../generated/collision/${link_name}\""
    lite6_integration_xacro "${lite6_integration_xacro}"
  )
endforeach()
add_custom_target(generate_collision_meshes ALL DEPENDS ${generated_collision_meshes})
file(
  WRITE
  "${generated_description_directory}/lite6.urdf.xacro"
  "${lite6_integration_xacro}"
)

# Keep the xArm6 snapshot unmodified and resolve its package references locally.
file(READ
  "${CMAKE_CURRENT_SOURCE_DIR}/description/ufactory_xarm6/urdf/xarm6/xarm6.urdf.xacro"
  xarm6_integration_xacro)
string(REPLACE "$(find xarm_description)"
  "$(find face_tracking_arm)/description/ufactory_xarm6"
  xarm6_integration_xacro "${xarm6_integration_xacro}")
file(WRITE "${generated_description_directory}/xarm6.urdf.xacro"
  "${xarm6_integration_xacro}")

install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION share/${PROJECT_NAME})

install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "calibration" EXCLUDE
  PATTERN "*.local.yaml" EXCLUDE
)
# Private Python modules sit beside their entry point; no overlap with rosidl's
# generated face_tracking_arm Python package and no build-time ML dependencies.
install(DIRECTORY python/face_tracking_perception DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY python/face_tracking_bringup DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(PROGRAMS
  scripts/face_detection_node.py
  scripts/face_position_node.py
  scripts/face_depth_reference.py
  scripts/face_appearances.py
  DESTINATION lib/${PROJECT_NAME})
install(
  FILES "${generated_description_directory}/lite6.urdf.xacro"
    "${generated_description_directory}/xarm6.urdf.xacro"
  DESTINATION share/${PROJECT_NAME}/description/generated
)
install(
  FILES ${generated_collision_meshes}
  DESTINATION share/${PROJECT_NAME}/description/generated/collision
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS
    tracking_core
    rolling_statistics
    control_time_grid
    following_error_policy
    published_trajectory_history
    hierarchical_velocity_qp
    tracking_velocity_task
    target_motion_estimator
    joint_path_follower
    background_path_planner
    emergency_brake_tail
    collision_constraints
    collision_aware_servo_component
    face_tracking_component
    test_face_target_publisher
    face_target_marker
    graceful_component_container
  EXPORT export_face_tracking_arm
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
ament_export_targets(export_face_tracking_arm HAS_LIBRARY_TARGET)
ament_export_dependencies(
  control_msgs Eigen3 moveit_core moveit_msgs moveit_ros_planning moveit_servo
  osqp rclcpp rosidl_default_runtime Threads
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(joint_trajectory_controller REQUIRED)
  # Sources use short SPDX headers referring to the repository-level MIT LICENSE.
  # ament_copyright expects the full license text in every source file.
  set(ament_cmake_copyright_FOUND TRUE)
  set(AMENT_LINT_AUTO_FILE_EXCLUDE
    .gitignore
    CONTRIBUTING.md
  )
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(
    test_monitor_description
    test/test_monitor_description.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_tracking_configuration
    test/test_tracking_configuration.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_target_marker
    test/test_target_marker.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_collision_mesh_generation
    test/test_collision_mesh_generation.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_face_detection test/test_face_detection.py TIMEOUT 30)
  ament_add_pytest_test(test_face_depth test/test_face_depth.py TIMEOUT 30)
  ament_add_pytest_test(test_face_appearances test/test_face_appearances.py TIMEOUT 30)
  ament_add_pytest_test(test_stress_motion test/test_stress_motion.py TIMEOUT 30)
  ament_add_pytest_test(test_fast_stress_motion test/test_fast_stress_motion.py TIMEOUT 30)
  ament_add_pytest_test(test_simulation_start test/test_simulation_start.py TIMEOUT 60)
  ament_add_pytest_test(test_face_selection test/test_face_selection.py TIMEOUT 30)
  ament_add_pytest_test(test_point_filter test/test_point_filter.py TIMEOUT 30)
  ament_add_pytest_test(test_kalman_point_filter test/test_kalman_point_filter.py TIMEOUT 30)
  ament_add_pytest_test(test_close_stress_motion test/test_close_stress_motion.py TIMEOUT 30)
  ament_add_pytest_test(test_hardware_config test/test_hardware_config.py TIMEOUT 30)
  ament_add_pytest_test(test_hardware_zero_geometry test/test_hardware_zero_geometry.py TIMEOUT 30)
  ament_add_pytest_test(test_mount_collisions test/test_mount_collisions.py TIMEOUT 30)
  ament_add_pytest_test(test_container_tools test/test_container_tools.py TIMEOUT 30)
  ament_add_pytest_test(test_camera_controls test/test_camera_controls.py TIMEOUT 30)
  ament_add_pytest_test(test_robot_profile test/test_robot_profile.py TIMEOUT 30)
  ament_add_pytest_test(test_hardware_probe test/test_hardware_probe.py TIMEOUT 30)
  ament_add_pytest_test(test_hardware_activation_guard
    test/test_hardware_activation_guard.py TIMEOUT 30)
  ament_add_pytest_test(test_controller_shutdown test/test_controller_shutdown.py TIMEOUT 60)
  ament_add_pytest_test(test_rectification test/test_rectification.py TIMEOUT 30)
  ament_add_gtest(test_table_geometry test/test_table_geometry.cpp)
  target_include_directories(test_table_geometry PRIVATE include)
  target_link_libraries(test_table_geometry Eigen3::Eigen)
  ament_target_dependencies(test_table_geometry moveit_msgs shape_msgs)
  ament_add_gtest(test_tracking_core test/test_tracking_core.cpp)
  ament_add_gtest(test_tracking_velocity_task test/test_tracking_velocity_task.cpp)
  target_link_libraries(test_tracking_velocity_task tracking_velocity_task)
  ament_add_gtest(test_target_motion_estimator test/test_target_motion_estimator.cpp)
  target_link_libraries(test_target_motion_estimator target_motion_estimator)
  ament_add_gtest(test_posture_recovery_policy test/test_posture_recovery_policy.cpp)
  target_include_directories(test_posture_recovery_policy PRIVATE include)
  ament_add_gtest(test_joint_path_follower test/test_joint_path_follower.cpp)
  ament_add_gtest(test_idle_search test/test_idle_search.cpp src/idle_search.cpp)
  target_include_directories(test_idle_search PRIVATE include)
  target_link_libraries(test_idle_search Eigen3::Eigen)
  target_link_libraries(test_joint_path_follower joint_path_follower)
  ament_add_gtest(test_background_path_planner test/test_background_path_planner.cpp)
  target_link_libraries(test_background_path_planner background_path_planner)
  # Expand the installed integration description at test time, after colcon
  # installs this package. This also works for a first build in a clean workspace.
  find_program(XACRO_EXECUTABLE xacro REQUIRED)
  add_test(NAME motion_reference_description
    COMMAND "${XACRO_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/description/xarm6_table.urdf.xacro"
      -o "${CMAKE_CURRENT_BINARY_DIR}/test_robot.urdf")
  set_tests_properties(motion_reference_description PROPERTIES
    FIXTURES_SETUP motion_reference_description)
  ament_add_gtest(test_motion_reference test/test_motion_reference.cpp)
  target_compile_definitions(test_motion_reference PRIVATE
    TEST_ROBOT_URDF_FILE="${CMAKE_CURRENT_BINARY_DIR}/test_robot.urdf"
    TEST_ROBOT_SRDF_FILE="${CMAKE_CURRENT_SOURCE_DIR}/config/moveit/xarm6.srdf")
  target_link_libraries(test_motion_reference collision_aware_servo_component)
  set_tests_properties(test_motion_reference PROPERTIES
    FIXTURES_REQUIRED motion_reference_description)
  target_link_libraries(test_tracking_core tracking_core Eigen3::Eigen)
  ament_add_gtest(test_face_scenarios_test test/test_face_scenarios.cpp)
  target_link_libraries(test_face_scenarios_test test_face_scenarios)
  ament_add_gtest(test_rolling_statistics test/test_rolling_statistics.cpp)
  target_link_libraries(test_rolling_statistics rolling_statistics)
  ament_add_gtest(test_control_time_grid test/test_control_time_grid.cpp)
  target_link_libraries(test_control_time_grid control_time_grid)
  ament_add_gtest(
    test_following_error_policy test/test_following_error_policy.cpp
  )
  target_link_libraries(test_following_error_policy following_error_policy)
  ament_add_gtest(
    test_published_trajectory_history test/test_published_trajectory_history.cpp
  )
  target_link_libraries(
    test_published_trajectory_history published_trajectory_history
  )
  ament_add_gtest(
    test_joint_trajectory_interpolation
    test/test_joint_trajectory_interpolation.cpp
  )
  target_link_libraries(
    test_joint_trajectory_interpolation
    joint_trajectory_controller::joint_trajectory_controller
    moveit_servo::moveit_servo_lib_cpp
  )
  ament_add_gtest(
    test_hierarchical_velocity_qp test/test_hierarchical_velocity_qp.cpp
  )
  target_link_libraries(
    test_hierarchical_velocity_qp
    emergency_brake_tail
    hierarchical_velocity_qp
    Eigen3::Eigen
  )
  ament_add_gtest(test_emergency_brake_tail test/test_emergency_brake_tail.cpp)
  target_link_libraries(test_emergency_brake_tail emergency_brake_tail Eigen3::Eigen)
  ament_add_gtest(
    test_collision_constraints test/test_collision_constraints.cpp
  )
  target_link_libraries(test_collision_constraints collision_constraints Eigen3::Eigen)
  ament_target_dependencies(test_collision_constraints moveit_core)
endif()

ament_package()
