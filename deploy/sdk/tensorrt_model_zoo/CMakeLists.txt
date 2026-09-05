cmake_minimum_required(VERSION 4.0)
project(yolov11)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# opencv
find_package(OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("./include")
#include_directories("/usr/include/aarch64-linux-gnu")
#link_directories("/usr/lib/aarch64-linux-gnu/nvidia")
#link_directories("/usr/lib/aarch64-linux-gnu")

# cuda
include_directories("/usr/local/cuda-12.6/include")
link_directories("/usr/local/cuda-12.6/lib64")

# tensorrt
include_directories("/home/caizhongyin/installs/TensorRT-10.3.0.26/include")
link_directories("/home/caizhongyin/installs/TensorRT-10.3.0.26/lib")

# target and lib
add_executable(${PROJECT_NAME} src/main.cpp src/YoloDet.cpp src/Class.cpp)

target_link_libraries(${PROJECT_NAME}  
    ${OpenCV_LIBS}  
    nvinfer
    #nvparsers
    cudart
    cublas
    cudnn
)
