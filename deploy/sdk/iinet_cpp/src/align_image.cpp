#include "align_image.h"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>

cv::Mat AlignImage::replaceZerosWithNeighbors(const cv::Mat& input) {
    cv::Mat result = input.clone();
    cv::Mat mask = (input == 0);
    
    // 3x3 kernel for neighbors (excluding center) - matching Python
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        1, 1, 1,
        1, 0, 1,
        1, 1, 1);
    
    // Calculate sum of neighbors
    cv::Mat neighbor_sum;
    cv::filter2D(input, neighbor_sum, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    
    // Calculate count of non-zero neighbors
    cv::Mat non_zero_mask = (input != 0);
    cv::Mat neighbor_count;
    non_zero_mask.convertTo(non_zero_mask, CV_32F);
    cv::filter2D(non_zero_mask, neighbor_count, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    
    // Calculate average (avoid division by zero)
    cv::Mat neighbor_avg;
    cv::divide(neighbor_sum, neighbor_count, neighbor_avg);
    
    // Replace zeros with neighbor average where possible (matching Python logic)
    cv::Mat valid_avg_mask = (neighbor_count > 0) & mask;
    neighbor_avg.copyTo(result, valid_avg_mask);
    
    return result;
}

cv::Mat AlignImage::alignImageByDepth(const cv::Mat& rgb,
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
    
    // 使用float32精度
    cv::Mat depth_f, K_rgb_f, K_depth_f, rgb2depth_f;
    depth.convertTo(depth_f, CV_32F);
    K_rgb.convertTo(K_rgb_f, CV_32F);
    K_depth.convertTo(K_depth_f, CV_32F);
    rgb2depth.convertTo(rgb2depth_f, CV_32F);
    
    cv::Mat K_depth_inv, rgb2depth_inv;
    cv::invert(K_depth_f, K_depth_inv);
    cv::invert(rgb2depth_f, rgb2depth_inv);
    
    cv::Mat result = cv::Mat::zeros(height, width, CV_32F);
    
    // 简化的单线程版本，确保正确性
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float d = depth_f.at<float>(v, u);
            if (d <= 0.0f) continue;
            
            // 深度相机像素坐标 -> 3D点
            cv::Mat pixel_depth = (cv::Mat_<float>(3, 1) << u * d, v * d, d);
            
            // 深度相机坐标系 -> 3D坐标
            cv::Mat point_3d_depth = K_depth_inv * pixel_depth;
            
            // 齐次坐标
            cv::Mat point_3d_depth_h = (cv::Mat_<float>(4, 1) << 
                point_3d_depth.at<float>(0), 
                point_3d_depth.at<float>(1), 
                point_3d_depth.at<float>(2), 
                1.0f);
            
            // 深度相机 -> RGB相机 坐标系转换
            cv::Mat point_3d_rgb_h = rgb2depth_inv * point_3d_depth_h;
            
            // RGB相机3D坐标
            cv::Mat point_3d_rgb = (cv::Mat_<float>(3, 1) << 
                point_3d_rgb_h.at<float>(0), 
                point_3d_rgb_h.at<float>(1), 
                point_3d_rgb_h.at<float>(2));
            
            // 投影到RGB图像平面
            cv::Mat pixel_rgb_h = K_rgb_f * point_3d_rgb;
            
            float z = pixel_rgb_h.at<float>(2);
            if (z > 0.0f) {
                int u_new = static_cast<int>(std::round(pixel_rgb_h.at<float>(0) / z));
                int v_new = static_cast<int>(std::round(pixel_rgb_h.at<float>(1) / z));
                
                if (u_new >= 0 && u_new < width && v_new >= 0 && v_new < height) {
                    float& current_depth = result.at<float>(v_new, u_new);
                    if (current_depth == 0.0f || z < current_depth) {
                        current_depth = z;
                    }
                }
            }
        }
    }
    
    // Fill zeros if requested
    if (fill_zero) {
        result = replaceZerosWithNeighbors(result);
    }
    
    return result;
}
