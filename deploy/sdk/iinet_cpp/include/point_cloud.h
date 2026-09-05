#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief 3D Point structure
 */
struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
    
    Point3D() : x(0), y(0), z(0), r(255), g(255), b(255) {}
    Point3D(float x_, float y_, float z_, uint8_t r_ = 255, uint8_t g_ = 255, uint8_t b_ = 255) 
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
};

class PointCloud {
public:
    PointCloud() = default;
    ~PointCloud() = default;
    
    /**
     * @brief Convert depth map to point cloud
     * @param depth Input depth map
     * @param K Camera intrinsic matrix (3x3)
     * @param rgb Optional RGB image for coloring
     * @param R Optional rotation matrix (3x3)
     * @param t Optional translation vector (3x1)
     * @return Vector of 3D points
     */
    static std::vector<Point3D> depthToPointCloud(const cv::Mat& depth, 
                                                 const cv::Mat& K,
                                                 const cv::Mat& rgb = cv::Mat(),
                                                 const cv::Mat& R = cv::Mat::eye(3, 3, CV_32F),
                                                 const cv::Mat& t = cv::Mat::zeros(3, 1, CV_32F));
    
    /**
     * @brief Save point cloud to PLY file
     * @param points Vector of 3D points
     * @param filename Output PLY filename
     * @return true if successful
     */
    static bool savePointCloudPLY(const std::vector<Point3D>& points, const std::string& filename);
    
    /**
     * @brief Convert depth map to point cloud and save to PLY file
     * @param depth Input depth map
     * @param K Camera intrinsic matrix (3x3)
     * @param filename Output PLY filename
     * @param rgb Optional RGB image for coloring
     * @param R Optional rotation matrix (3x3)
     * @param t Optional translation vector (3x1)
     * @return true if successful
     */
    static bool depth2PCD(const cv::Mat& depth,
                         const cv::Mat& K,
                         const std::string& filename,
                         const cv::Mat& rgb = cv::Mat(),
                         const cv::Mat& R = cv::Mat::eye(3, 3, CV_32F),
                         const cv::Mat& t = cv::Mat::zeros(3, 1, CV_32F));
    
    /**
     * @brief Filter outliers from point cloud
     * @param points Input points
     * @param ratio Ratio of outliers to remove (0.0 - 1.0)
     * @return Filtered points
     */
    static std::vector<Point3D> filterOutliers(const std::vector<Point3D>& points, float ratio = 0.1f);
};
