#include "point_cloud.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>

std::vector<Point3D> PointCloud::depthToPointCloud(const cv::Mat& depth, 
                                                   const cv::Mat& K,
                                                   const cv::Mat& rgb,
                                                   const cv::Mat& R,
                                                   const cv::Mat& t) {
    std::vector<Point3D> points;
    if (depth.empty() || K.empty()) {
        return points;
    }
    int height = depth.rows;
    int width = depth.cols;
    // 采样步长，和Python保持一致，建议2或1
    const int stride = 2; // 可调节，和Python一致
    points.reserve(height * width / stride / stride);
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);
    cv::Mat R_use = R.empty() ? cv::Mat::eye(3, 3, CV_32F) : R;
    cv::Mat t_use = t.empty() ? cv::Mat::zeros(3, 1, CV_32F) : t;
    float r00 = R_use.at<float>(0, 0), r01 = R_use.at<float>(0, 1), r02 = R_use.at<float>(0, 2);
    float r10 = R_use.at<float>(1, 0), r11 = R_use.at<float>(1, 1), r12 = R_use.at<float>(1, 2);
    float r20 = R_use.at<float>(2, 0), r21 = R_use.at<float>(2, 1), r22 = R_use.at<float>(2, 2);
    float tx = t_use.at<float>(0, 0), ty = t_use.at<float>(1, 0), tz = t_use.at<float>(2, 0);
    for (int v = 0; v < height; v += stride) {
        const float* depth_row = depth.ptr<float>(v);
        for (int u = 0; u < width; u += stride) {
            float d = depth_row[u];
            // 过滤条件：z>0且非nan/inf
            if (d <= 0.0f || std::isnan(d) || std::isinf(d)) continue;
            float x = (u - cx) * d / fx;
            float y = (v - cy) * d / fy;
            float z = d;
            Point3D pt;
            pt.x = r00 * x + r01 * y + r02 * z + tx;
            pt.y = r10 * x + r11 * y + r12 * z + ty;
            pt.z = r20 * x + r21 * y + r22 * z + tz;
            // 颜色顺序统一为RGB
            if (!rgb.empty() && v < rgb.rows && u < rgb.cols) {
                const cv::Vec3b& color = rgb.at<cv::Vec3b>(v, u);
                pt.r = color[2]; // OpenCV BGR->RGB
                pt.g = color[1];
                pt.b = color[0];
            } else {
                pt.r = pt.g = pt.b = 255;
            }
            points.push_back(pt);
        }
    }
    return points;
}

bool PointCloud::savePointCloudPLY(const std::vector<Point3D>& points, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    // Write binary PLY header
    std::string header =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex " + std::to_string(points.size()) + "\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n";
    file.write(header.c_str(), header.size());

    // Write point data in binary
    for (const auto& point : points) {
        file.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.r), sizeof(uint8_t));
        file.write(reinterpret_cast<const char*>(&point.g), sizeof(uint8_t));
        file.write(reinterpret_cast<const char*>(&point.b), sizeof(uint8_t));
    }

    file.close();
    return true;
}

bool PointCloud::depth2PCD(const cv::Mat& depth,
                          const cv::Mat& K,
                          const std::string& filename,
                          const cv::Mat& rgb,
                          const cv::Mat& R,
                          const cv::Mat& t) {
    auto points = depthToPointCloud(depth, K, rgb, R, t);
    return savePointCloudPLY(points, filename);
}

std::vector<Point3D> PointCloud::filterOutliers(const std::vector<Point3D>& points, float ratio) {
    if (points.empty() || ratio <= 0.0f || ratio >= 1.0f) {
        return points;
    }
    
    // Calculate centroid
    float sum_x = 0, sum_y = 0, sum_z = 0;
    for (const auto& point : points) {
        sum_x += point.x;
        sum_y += point.y;
        sum_z += point.z;
    }
    
    float centroid_x = sum_x / points.size();
    float centroid_y = sum_y / points.size();
    float centroid_z = sum_z / points.size();
    
    // Calculate distances from centroid
    std::vector<std::pair<float, size_t>> distances;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        float dx = point.x - centroid_x;
        float dy = point.y - centroid_y;
        float dz = point.z - centroid_z;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        distances.emplace_back(dist, i);
    }
    
    // Sort by distance
    std::sort(distances.begin(), distances.end());
    
    // Keep only the closest (1 - ratio) fraction of points
    size_t num_inliers = static_cast<size_t>(points.size() * (1.0f - ratio));
    
    std::vector<Point3D> filtered_points;
    filtered_points.reserve(num_inliers);
    
    for (size_t i = 0; i < num_inliers; ++i) {
        filtered_points.push_back(points[distances[i].second]);
    }
    
    return filtered_points;
}
