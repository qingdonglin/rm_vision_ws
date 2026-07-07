cmake_minimum_required(VERSION 3.8)
project(auto_aim_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/Armors.msg"
  "msg/Target.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# 正确安装生成的头文件
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/rosidl_generator_cpp/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)
ament_package()