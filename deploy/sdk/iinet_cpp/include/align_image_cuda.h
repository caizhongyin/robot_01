#ifndef ALIGN_IMAGE_CUDA_H
#define ALIGN_IMAGE_CUDA_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

class AlignImageCUDA {
public:
    AlignImageCUDA();
    ~AlignImageCUDA();
    
    /**
     * GPU加速的深度对齐函数
     * @param rgb RGB图像 (可选)
     * @param depth 深度图
     * @param K_rgb RGB相机内参矩阵
     * @param K_depth 深度相机内参矩阵
     * @param rgb2depth RGB到深度相机的变换矩阵
     * @param to_depth 是否对齐到深度坐标系
     * @param fill_zero 是否填充零值
     * @return 对齐后的深度图
     */
    cv::Mat alignImageByDepth(const cv::Mat& rgb,
                             const cv::Mat& depth,
                             const cv::Mat& K_rgb,
                             const cv::Mat& K_depth,
                             const cv::Mat& rgb2depth,
                             bool to_depth = false,
                             bool fill_zero = true);

private:
    // CUDA流
    cudaStream_t stream_;
    
    // GPU内存缓存
    float* d_depth_;
    float* d_K_rgb_;
    float* d_K_depth_;
    float* d_rgb2depth_;
    float* d_result_;
    
    int max_width_;
    int max_height_;
    bool initialized_;
    
    void initializeGPUMemory(int width, int height);
    void cleanup();
};

#endif // ALIGN_IMAGE_CUDA_H
