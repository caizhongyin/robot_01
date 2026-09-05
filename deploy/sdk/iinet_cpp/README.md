# IINet TensorRT C++ Inference

This project is a C++ implementation of IINet using TensorRT inference.


## Installation
### Ubuntu
```bash

# Install yaml-cpp
sudo apt install libyaml-cpp-dev

```
### Building
```bash
# Clone and build
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Usage
```bash
# Run with default parameters
./IINet_TRT_Inference

# Specify custom paths
./IINet_TRT_Inference [engine_path] [dataset_path] [ratio]
```

### Example
```bash
./IINet_TRT_Inference \
    ../checkpoints/iinet.engine \
    ../20250429_104224_images/20250429_104240_213 \
    0.9
```

## Input Data Structure

Dataset directory should contain:
```
dataset_path/
├── left.png          # Left stereo image
├── right.png         # Right stereo image
├── rgb.png           # RGB color image (optional)
├── cam_params.yaml   # Camera parameters
├── K_color.txt       # RGB camera intrinsic matrix (3x3)
├── K_depth.txt       # Depth camera intrinsic matrix (3x3)
└── color2depth.txt   # Transformation matrix (4x4)
```

## Output

- **Depth Visualization**: `cpp_depth_visualization.png` - False-color depth map

## Code Structure

```
cpp/
├── include/
│   ├── trt_infer.h         # TensorRT inference engine
│   ├── realsense_reader.h  # Camera data reader
│   ├── image_processor.h   # Image preprocessing
│   ├── point_cloud.h       # Point cloud operations
│   └── align_image.h       # Image/depth alignment
├── src/
│   ├── main.cpp            # Main application
│   ├── trt_infer.cpp       # TensorRT implementation
│   ├── realsense_reader.cpp # Data loading
│   ├── image_processor.cpp # Image processing
│   ├── point_cloud.cpp     # Point cloud utilities
│   └── align_image.cpp     # Alignment algorithms
└── CMakeLists.txt          # Build configuration
```

