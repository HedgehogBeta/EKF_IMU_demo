cmake_minimum_required(VERSION 3.8)
project(eskf_imu)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(imu_player_node src/imu_player_node.cpp)
ament_target_dependencies(imu_player_node rclcpp sensor_msgs)
target_include_directories(imu_player_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIR})

add_executable(pose_player_node src/pose_player_node.cpp)
ament_target_dependencies(pose_player_node rclcpp geometry_msgs)
target_include_directories(pose_player_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# M2：手写四元数单元测试（不依赖 ROS，与 Eigen 对照）
add_executable(test_quaternion test/test_quaternion.cpp)
target_include_directories(test_quaternion PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIR})

# M2：纯陀螺积分节点
add_executable(gyro_integral_node src/gyro_integral_node.cpp)
ament_target_dependencies(gyro_integral_node rclcpp sensor_msgs)
target_include_directories(gyro_integral_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# ESKF 主体（M3/M4）
# add_executable(eskf_node src/eskf_node.cpp)
# ament_target_dependencies(eskf_node rclcpp sensor_msgs geometry_msgs Eigen3)

install(TARGETS imu_player_node pose_player_node gyro_integral_node test_quaternion
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

ament_package()
