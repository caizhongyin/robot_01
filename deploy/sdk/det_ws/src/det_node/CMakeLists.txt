cmake_minimum_required(VERSION 4.0)
project(s009_det_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

# opencv
find_package(OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# yolo
include_directories(
  "include"
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
)

# cuda
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

# tensorrt
#include_directories("/home/caizhongyin/installs/TensorRT-10.3.0.26/include")
#link_directories("/home/caizhongyin/installs/TensorRT-10.3.0.26/lib")

# target and lib
#add_executable(${PROJECT_NAME} src/main.cpp)

# 创建发布者可执行文件
add_executable(S009DetectNode src/S009DetectNode.cpp src/YoloDet.cpp)
ament_target_dependencies(S009DetectNode rclcpp std_msgs)

target_link_libraries(S009DetectNode  
    ${OpenCV_LIBS}
    nvinfer
    #nvparsers
    cudart
    cublas
    cudnn
)

# 安装可执行文件
install(TARGETS
  S009DetectNode
  DESTINATION lib/${PROJECT_NAME})
