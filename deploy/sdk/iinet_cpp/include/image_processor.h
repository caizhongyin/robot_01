#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

class ImageProcessor {
public:
    ImageProcessor() = default;
    ~ImageProcessor() = default;
    
    /**
     * @brief Preprocess image for neural network inference
     * @param image_path Path to input image
     * @param processed_img Output processed image as float array
     * @param top_pad Top padding applied
     * @param right_pad Right padding applied
     * @return true if successful
     */
    static bool preprocessImage(const std::string& image_path, 
                               cv::Mat& processed_img, 
                               int& top_pad, 
                               int& right_pad);
    
    /**
     * @brief Preprocess OpenCV Mat for neural network inference
     * @param img Input image
     * @param processed_img Output processed image as float array
     * @param top_pad Top padding applied
     * @param right_pad Right padding applied
     * @return true if successful
     */
    static bool preprocessImage(const cv::Mat& img,
                               cv::Mat& processed_img,
                               int& top_pad,
                               int& right_pad);
    
    /**
     * @brief Convert depth image to visualization
     * @param depth Input depth map
     * @param depth_vis Output visualization
     * @param max_depth Maximum depth for visualization
     */
    static void visualizeDepth(const cv::Mat& depth, cv::Mat& depth_vis, float max_depth = 5.0f);
    
private:
    static const std::vector<float> IMAGENET_MEAN;
    static const std::vector<float> IMAGENET_STD;
};
