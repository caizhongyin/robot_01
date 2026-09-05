#include "align_image_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        } \
    } while(0)

// CUDA内核：深度对齐
__global__ void depthAlignmentKernel(
    const float* __restrict__ depth,
    float* __restrict__ result,
    const float* __restrict__ K_rgb,
    const float* __restrict__ K_depth_inv,
    const float* __restrict__ rgb2depth_inv,
    int width,
    int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int u = idx % width;
    int v = idx / width;
    float d = depth[idx];
    
    if (d <= 0.0f) return;
    
    // 深度相机像素坐标转3D点 (使用齐次坐标)
    float pixel_depth[3] = {(float)u * d, (float)v * d, d};
    
    // 转换到深度相机3D坐标系 (K_depth_inv * pixel_depth)
    float point_3d_depth[3] = {
        K_depth_inv[0] * pixel_depth[0] + K_depth_inv[1] * pixel_depth[1] + K_depth_inv[2] * pixel_depth[2],
        K_depth_inv[3] * pixel_depth[0] + K_depth_inv[4] * pixel_depth[1] + K_depth_inv[5] * pixel_depth[2],
        K_depth_inv[6] * pixel_depth[0] + K_depth_inv[7] * pixel_depth[1] + K_depth_inv[8] * pixel_depth[2]
    };
    
    // 转换到RGB相机3D坐标系 (rgb2depth_inv * [x, y, z, 1])
    float point_3d_rgb[4] = {point_3d_depth[0], point_3d_depth[1], point_3d_depth[2], 1.0f};
    float temp[4];
    
    for (int i = 0; i < 4; i++) {
        temp[i] = rgb2depth_inv[i*4 + 0] * point_3d_rgb[0] +
                  rgb2depth_inv[i*4 + 1] * point_3d_rgb[1] +
                  rgb2depth_inv[i*4 + 2] * point_3d_rgb[2] +
                  rgb2depth_inv[i*4 + 3] * point_3d_rgb[3];
    }
    
    // 投影到RGB图像平面 (K_rgb * temp[:3])
    float projected[3] = {
        K_rgb[0] * temp[0] + K_rgb[1] * temp[1] + K_rgb[2] * temp[2],
        K_rgb[3] * temp[0] + K_rgb[4] * temp[1] + K_rgb[5] * temp[2],
        K_rgb[6] * temp[0] + K_rgb[7] * temp[1] + K_rgb[8] * temp[2]
    };
    
    if (projected[2] > 0.0f) {
        int u_new = __float2int_rn(projected[0] / projected[2]);
        int v_new = __float2int_rn(projected[1] / projected[2]);
        
        if (u_new >= 0 && u_new < width && v_new >= 0 && v_new < height) {
            int new_idx = v_new * width + u_new;
            // 使用原子操作来避免竞争条件
            float old_val = atomicExch(&result[new_idx], projected[2]);
            if (old_val != 0.0f && old_val < projected[2]) {
                atomicExch(&result[new_idx], old_val);
            }
        }
    }
}

// CUDA内核：填充零值
__global__ void fillZerosKernel(float* depth, int width, int height) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (u >= width || v >= height) return;
    
    int idx = v * width + u;
    if (depth[idx] != 0.0f) return;
    
    float sum = 0.0f;
    int count = 0;
    
    // 3x3邻域
    for (int dv = -1; dv <= 1; dv++) {
        for (int du = -1; du <= 1; du++) {
            if (du == 0 && dv == 0) continue;
            
            int nu = u + du;
            int nv = v + dv;
            
            if (nu >= 0 && nu < width && nv >= 0 && nv < height) {
                float val = depth[nv * width + nu];
                if (val > 0.0f) {
                    sum += val;
                    count++;
                }
            }
        }
    }
    
    if (count > 0) {
        depth[idx] = sum / count;
    }
}

AlignImageCUDA::AlignImageCUDA() : 
    stream_(nullptr),
    d_depth_(nullptr),
    d_K_rgb_(nullptr),
    d_K_depth_(nullptr),
    d_rgb2depth_(nullptr),
    d_result_(nullptr),
    max_width_(0),
    max_height_(0),
    initialized_(false)
{
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

AlignImageCUDA::~AlignImageCUDA() {
    cleanup();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void AlignImageCUDA::initializeGPUMemory(int width, int height) {
    if (initialized_ && width <= max_width_ && height <= max_height_) {
        return;
    }
    
    cleanup();
    
    max_width_ = width;
    max_height_ = height;
    int total_pixels = width * height;
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&d_depth_, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result_, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_rgb_, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_depth_, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb2depth_, 16 * sizeof(float)));
    
    initialized_ = true;
}

void AlignImageCUDA::cleanup() {
    if (d_depth_) { cudaFree(d_depth_); d_depth_ = nullptr; }
    if (d_result_) { cudaFree(d_result_); d_result_ = nullptr; }
    if (d_K_rgb_) { cudaFree(d_K_rgb_); d_K_rgb_ = nullptr; }
    if (d_K_depth_) { cudaFree(d_K_depth_); d_K_depth_ = nullptr; }
    if (d_rgb2depth_) { cudaFree(d_rgb2depth_); d_rgb2depth_ = nullptr; }
    
    initialized_ = false;
}

cv::Mat AlignImageCUDA::alignImageByDepth(const cv::Mat& rgb,
                                          const cv::Mat& depth,
                                          const cv::Mat& K_rgb,
                                          const cv::Mat& K_depth,
                                          const cv::Mat& rgb2depth,
                                          bool to_depth,
                                          bool fill_zero) {
    if (depth.empty() || K_rgb.empty() || K_depth.empty() || rgb2depth.empty()) {
        std::cerr << "Error: Invalid input parameters to alignImageByDepth" << std::endl;
        return cv::Mat();
    }
    
    int height = depth.rows;
    int width = depth.cols;
    
    if (to_depth) {
        return rgb.clone();
    }
    
    // 初始化GPU内存
    initializeGPUMemory(width, height);
    
    // 转换数据类型
    cv::Mat depth_f, K_rgb_f, K_depth_f, rgb2depth_f;
    depth.convertTo(depth_f, CV_32F);
    K_rgb.convertTo(K_rgb_f, CV_32F);
    K_depth.convertTo(K_depth_f, CV_32F);
    rgb2depth.convertTo(rgb2depth_f, CV_32F);
    
    // 计算逆矩阵
    cv::Mat K_rgb_inv, K_depth_inv, rgb2depth_inv;
    cv::invert(K_rgb_f, K_rgb_inv);
    cv::invert(K_depth_f, K_depth_inv);
    cv::invert(rgb2depth_f, rgb2depth_inv);
    
    // 拷贝数据到GPU
    CUDA_CHECK(cudaMemcpyAsync(d_depth_, depth_f.ptr<float>(), 
                               width * height * sizeof(float), 
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_K_rgb_, K_rgb_f.ptr<float>(), 
                               9 * sizeof(float), 
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_K_depth_, K_depth_inv.ptr<float>(), 
                               9 * sizeof(float), 
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_rgb2depth_, rgb2depth_inv.ptr<float>(), 
                               16 * sizeof(float), 
                               cudaMemcpyHostToDevice, stream_));
    
    // 初始化结果为0
    CUDA_CHECK(cudaMemsetAsync(d_result_, 0, width * height * sizeof(float), stream_));
    
    // 启动CUDA内核
    dim3 blockSize(256);
    dim3 gridSize((width * height + blockSize.x - 1) / blockSize.x);
    
    depthAlignmentKernel<<<gridSize, blockSize, 0, stream_>>>(
        d_depth_, d_result_, d_K_rgb_, d_K_depth_, d_rgb2depth_, width, height);
    
    CUDA_CHECK(cudaGetLastError());
    
    // 填充零值
    if (fill_zero) {
        dim3 blockSize2D(16, 16);
        dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                        (height + blockSize2D.y - 1) / blockSize2D.y);
        
        fillZerosKernel<<<gridSize2D, blockSize2D, 0, stream_>>>(
            d_result_, width, height);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 拷贝结果回CPU
    cv::Mat result(height, width, CV_32F);
    CUDA_CHECK(cudaMemcpyAsync(result.ptr<float>(), d_result_, 
                               width * height * sizeof(float), 
                               cudaMemcpyDeviceToHost, stream_));
    
    // 等待GPU操作完成
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return result;
}
