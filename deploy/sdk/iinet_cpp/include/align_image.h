#pragma once

#include <opencv2/opencv.hpp>

class AlignImage {
public:
    AlignImage() = default;
    ~AlignImage() = default;
    
    /**
     * @brief Align RGB image with depth using camera parameters
     * High-performance implementation matching Python align_image2.py behavior
     * @param rgb RGB image from color camera
     * @param depth Depth map from depth camera  
     * @param K_rgb Intrinsic matrix of RGB camera (3x3)
     * @param K_depth Intrinsic matrix of depth camera (3x3)
     * @param rgb2depth Extrinsic transformation from RGB to depth (4x4)
     * @param to_depth If true, align RGB to depth frame; if false, align depth to RGB frame
     * @param fill_zero If true, fill zero regions with neighbor interpolation
     * @return Aligned image/depth map
     */
    static cv::Mat alignImageByDepth(const cv::Mat& rgb,
                                   const cv::Mat& depth,
                                   const cv::Mat& K_rgb,
                                   const cv::Mat& K_depth,
                                   const cv::Mat& rgb2depth,
                                   bool to_depth = false,
                                   bool fill_zero = true);
    
    /**
     * @brief Replace zeros with average of neighbors (matching Python behavior)
     * @param input Input 2D array
     * @return Array with zeros replaced
     */
    static cv::Mat replaceZerosWithNeighbors(const cv::Mat& input);
};
