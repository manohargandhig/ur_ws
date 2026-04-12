cmake_minimum_required(VERSION 3.8)
project(ur_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==============================
# FIND DEPENDENCIES
# ==============================

find_package(ament_cmake REQUIRED)

# 🔥 MESSAGE GENERATION
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

# Core
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# MoveIt
find_package(moveit_ros_planning_interface REQUIRED)

# Attach/Detach
find_package(linkattacher_msgs REQUIRED)
find_package(ros2_linkattacher REQUIRED)

# Utils
find_package(xacro REQUIRED)

# ==============================
# 🔥 GENERATE CUSTOM MESSAGES
# ==============================

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/DetectedObjectArray.msg"
  DEPENDENCIES geometry_msgs
)

# ==============================
# EXECUTABLES
# ==============================

add_executable(test_ik src/test_ik.cpp)
ament_target_dependencies(test_ik rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs)

add_executable(test_ik_collision_obj src/test_ik_collision_obj.cpp)
ament_target_dependencies(test_ik_collision_obj rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs)

add_executable(pick_place src/pick_place.cpp)
ament_target_dependencies(pick_place rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs linkattacher_msgs ros2_linkattacher)

add_executable(pick_place_v2 src/pick_place_v2.cpp)
ament_target_dependencies(pick_place_v2 rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs linkattacher_msgs ros2_linkattacher)

add_executable(pick_place_vision src/pick_place_vision.cpp)
ament_target_dependencies(pick_place_vision rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs linkattacher_msgs ros2_linkattacher)

add_executable(pick_place_vision_v3 src/pick_place_vision_v3.cpp)
ament_target_dependencies(pick_place_vision_v3 rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs linkattacher_msgs ros2_linkattacher)

# ==============================
# 🔥 LINK CUSTOM MSG TO EXECUTABLES
# ==============================

rosidl_target_interfaces(pick_place_vision
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

rosidl_target_interfaces(pick_place_vision_v3
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ==============================
# INSTALL
# ==============================

install(DIRECTORY launch urdf config meshes worlds rviz models
  DESTINATION share/${PROJECT_NAME})

install(TARGETS 
  test_ik 
  test_ik_collision_obj 
  pick_place 
  pick_place_v2 
  pick_place_vision 
  pick_place_vision_v3
  DESTINATION lib/${PROJECT_NAME}/
)

# ==============================
# 🔥 EXPORT MESSAGE RUNTIME
# ==============================

ament_export_dependencies(rosidl_default_runtime)

# ==============================
# TESTING
# ==============================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
