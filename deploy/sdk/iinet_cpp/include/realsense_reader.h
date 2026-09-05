#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

struct CameraParams {
    cv::Mat K_color;      // 3x3 color camera intrinsic matrix
    cv::Mat K_depth;      // 3x3 depth camera intrinsic matrix  
    cv::Mat color2depth;  // 4x4 transformation matrix from color to depth
    double baseline;      // Baseline distance between cameras
    std::string model;    // Camera model
    std::string serial_number;
    int width;
    int height;
    int fps;
};

struct Frame {
    cv::Mat rgb;
    cv::Mat left;
    cv::Mat right;
    cv::Mat depth;
};

class RealsenseReader {
public:
    RealsenseReader() = default;
    ~RealsenseReader() = default;
    
    /**
     * @brief Read frame data from directory
     * @param frame_dir Directory containing frame data
     * @param frame Output frame data
     * @param cam_params Output camera parameters
     * @return true if successful
     */
    static bool readFrame(const std::string& frame_dir, Frame& frame, CameraParams& cam_params);
    
    /**
     * @brief Load camera parameters from YAML file
     * @param yaml_path Path to camera parameters YAML file
     * @param cam_params Output camera parameters
     * @return true if successful
     */
    static bool loadCameraParams(const std::string& yaml_path, CameraParams& cam_params);
    
    /**
     * @brief Load matrix from text file
     * @param file_path Path to matrix file (.txt or .npy)
     * @param matrix Output matrix
     * @return true if successful
     */
    static bool loadMatrix(const std::string& file_path, cv::Mat& matrix);
    
private:
    static bool fileExists(const std::string& path);
    static std::vector<std::string> findFiles(const std::string& dir, const std::string& pattern);
};
