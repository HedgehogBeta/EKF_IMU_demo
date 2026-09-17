cmake_minimum_required(VERSION 3.8)
project(eskf_imu)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 数据回放：imu.csv -> /imu
add_executable(imu_player_node src/imu_player_node.cpp)
ament_target_dependencies(imu_player_node rclcpp sensor_msgs)
target_include_directories(imu_player_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# 数据回放：pose_cov.csv -> /pose_cov（作为观测）
add_executable(pose_player_node src/pose_player_node.cpp)
ament_target_dependencies(pose_player_node rclcpp geometry_msgs)
target_include_directories(pose_player_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# ESKF：IMU 预测 + 位姿观测更新（姿态+位置）
add_executable(eskf_node src/eskf_node.cpp)
ament_target_dependencies(eskf_node rclcpp sensor_msgs geometry_msgs)
target_include_directories(eskf_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS imu_player_node pose_player_node eskf_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

ament_package()
