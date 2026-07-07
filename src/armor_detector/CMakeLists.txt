cmake_minimum_required(VERSION 3.8)
project(armor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依赖查找
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# OpenVINO 依赖（编译前必须先 source setupvars.sh）
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)

add_compile_definitions(_USE_MATH_DEFINES)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件路径
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenVINO_INCLUDE_DIRS}
)

# 可执行文件
add_executable(armor_detector_node
  src/detector_node.cpp
  src/detector.cpp
)

add_executable(gimbal_tf_broadcaster
  src/gimbal_tf_broadcaster.cpp
)

add_executable(fire_control_node
  src/fire_control_node.cpp
)

# 视觉节点依赖
ament_target_dependencies(armor_detector_node
  rclcpp sensor_msgs std_msgs cv_bridge image_transport
  auto_aim_interfaces geometry_msgs tf2_ros tf2_geometry_msgs
  ament_index_cpp
)
target_link_libraries(armor_detector_node
  ${OpenCV_LIBS}
  Eigen3::Eigen
  openvino::runtime
  openvino::frontend::onnx
  pugixml
)

# TF广播节点
ament_target_dependencies(gimbal_tf_broadcaster
  rclcpp tf2_ros geometry_msgs std_msgs
)

# 火控节点
ament_target_dependencies(fire_control_node
  rclcpp auto_aim_interfaces std_msgs geometry_msgs
  tf2_ros tf2_geometry_msgs
)
target_link_libraries(fire_control_node Eigen3::Eigen)

# 安装规则
install(TARGETS
  armor_detector_node gimbal_tf_broadcaster fire_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/armor_detector/
  DESTINATION include/${PROJECT_NAME}/
)

install(DIRECTORY launch config model
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()