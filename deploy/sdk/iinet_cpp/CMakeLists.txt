cmake_minimum_required(VERSION 3.16)
project(IINet_TRT_Inference LANGUAGES CXX CUDA)

# get current platform
execute_process(COMMAND bash "-c" "uname -m | tr -d '\n'" OUTPUT_VARIABLE PLATFORM_TYPE)
message(STATUS "Platform Type:" ${PLATFORM_TYPE})
add_definitions(-DPLATFORM_TYPE="${PLATFORM_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Aggressive optimization flags for maximum performance (without LTO for CUDA)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# tensorrt
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    # tensorrt
    set(TensorRT_DIR /usr/src/tensorrt)
    include_directories(${TensorRT_DIR}/include)
    link_directories(${TensorRT_DIR}/lib)

else() # x86_64
    message("embed_platform off")
    # # cuda
    # include_directories(/usr/local/cuda/include)
    # link_directories(/usr/local/cuda/lib64)
    # tensorrt (根据本机实际安装路径修改)
    set(TensorRT_DIR /usr/src/tensorrt)
    include_directories(${TensorRT_DIR}/include)
    link_directories(${TensorRT_DIR}/lib)
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(CPP_SOURCES
    src/main.cpp
    src/trt_infer.cpp
    src/align_image.cpp
    src/point_cloud.cpp
    src/realsense_reader.cpp
    src/image_processor.cpp
)

# Check if CUDA source exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/align_image_cuda.cu")
    set(CUDA_SOURCES src/align_image_cuda.cu)
    set(USE_CUDA_ALIGN TRUE)
else()
    set(CUDA_SOURCES "")
    set(USE_CUDA_ALIGN FALSE)
    message(STATUS "CUDA alignment disabled - using CPU version")
endif()

# Create executable
if(USE_CUDA_ALIGN)
    add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA_ALIGN=1)
else()
    add_executable(${PROJECT_NAME} ${CPP_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA_ALIGN=0)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    nvonnxparser
    nvinfer_plugin
    yaml-cpp
    Threads::Threads
)

# Compiler-specific options for maximum performance (without LTO for CUDA)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -Wall -Wextra
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer -DNDEBUG>
        $<$<CONFIG:Debug>:-g -O0>)
endif()

# Auto-detect CUDA architecture - only if using CUDA
if(USE_CUDA_ALIGN)
    # Check if CMAKE_CUDA_ARCHITECTURES was set manually
    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "Using manually set CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Let CMake auto-detect CUDA architectures for optimal performance
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            # For embedded platforms, use common architectures
            set(CMAKE_CUDA_ARCHITECTURES "72;87")
            message(STATUS "Using CUDA architectures for embedded platform: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            # For x86_64, use common modern architectures including RTX 4060 (8.9)
            # Include multiple architectures for better compatibility
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
            message(STATUS "Using CUDA architectures for x86_64: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
    
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    
    # CUDA optimization flags
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)
    
    # Force disable LTO for CUDA compatibility
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
else()
    # No LTO even for CPU to keep consistency
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
endif()
