#include "image_processor.h"
#include <iostream>
#include <cstring>

const std::vector<float> ImageProcessor::IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> ImageProcessor::IMAGENET_STD = {0.229f, 0.224f, 0.225f};

bool ImageProcessor::preprocessImage(const std::string& image_path, 
                                   cv::Mat& processed_img, 
                                   int& top_pad, 
                                   int& right_pad) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Cannot load image from " << image_path << std::endl;
        return false;
    }
    
    return preprocessImage(img, processed_img, top_pad, right_pad);
}

bool ImageProcessor::preprocessImage(const cv::Mat& img,
                                   cv::Mat& processed_img,
                                   int& top_pad,
                                   int& right_pad) {
    if (img.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return false;
    }
    
     int h = img.rows;
    int w = img.cols;

    // 计算 padding
    top_pad   = (32 - (h % 32)) % 32;
    right_pad = (32 - (w % 32)) % 32;

    cv::Mat padded_img;
    if (top_pad != 0 || right_pad != 0) {
        cv::copyMakeBorder(img, padded_img, top_pad, 0, 0, right_pad,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        padded_img = img;
    }

    // 使用 blobFromImage 处理：
    // - BGR -> RGB 内部完成
    // - float32
    // - scale [0,1]
    // - 减均值
    cv::dnn::blobFromImage(
        padded_img,                                 // 输入图像，BGR
        processed_img,                              // 输出 blob [1,3,H,W]
        1.0 / 255.0f,                               // scale
        cv::Size(),                                 // 保持原尺寸
        cv::Scalar(IMAGENET_MEAN[0],
                   IMAGENET_MEAN[1],
                   IMAGENET_MEAN[2]), // mean
        true,                                       // swapRB = true，BGR->RGB
        false                                       // crop = false
    );

    // 标准差归一化
    int channels = processed_img.size[1];
    int height   = processed_img.size[2];
    int width    = processed_img.size[3];
    float* data_ptr = reinterpret_cast<float*>(processed_img.data);

    for (int c = 0; c < channels; ++c) {
        float scale = 1.0f / ImageProcessor::IMAGENET_STD[c];
        for (int i = 0; i < height * width; ++i) {
            data_ptr[c * height * width + i] *= scale;
        }
    }

    // reshape 为 [C*H, W] 方便 TensorRT 输入
    processed_img = processed_img.reshape(1, {channels * height, width});

    return true;
}

void ImageProcessor::visualizeDepth(const cv::Mat& depth, cv::Mat& depth_vis, float max_depth) {
    cv::Mat normalized_depth;
    depth.convertTo(normalized_depth, CV_8U, 255.0 / max_depth);
    cv::applyColorMap(normalized_depth, depth_vis, cv::COLORMAP_JET);
}
