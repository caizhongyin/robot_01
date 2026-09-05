#include "realsense_reader.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

bool RealsenseReader::fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::vector<std::string> RealsenseReader::findFiles(const std::string& dir, const std::string& pattern) {
    std::vector<std::string> files;
    
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(pattern) != std::string::npos) {
                files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

bool RealsenseReader::readFrame(const std::string& frame_dir, Frame& frame, CameraParams& cam_params) {
    if (!std::filesystem::exists(frame_dir) || !std::filesystem::is_directory(frame_dir)) {
        std::cerr << "Error: " << frame_dir << " is not a directory" << std::endl;
        return false;
    }
    
    // Initialize frame
    frame = Frame();
    
    // Load images
    std::vector<std::pair<std::string, cv::Mat*>> image_mappings = {
        {"rgb.png", &frame.rgb},
        {"left.png", &frame.left},
        {"right.png", &frame.right},
        {"depth.png", &frame.depth}
    };
    
    for (const auto& mapping : image_mappings) {
        std::string img_path = frame_dir + "/" + mapping.first;
        if (fileExists(img_path)) {
            if (mapping.first == "depth.png") {
                // Read depth as 16-bit
                *mapping.second = cv::imread(img_path, cv::IMREAD_ANYDEPTH);
            } else {
                *mapping.second = cv::imread(img_path, cv::IMREAD_COLOR);
            }
        }
    }
    
    // Load camera parameters
    std::string cam_params_path = frame_dir + "/cam_params.yaml";
    if (!fileExists(cam_params_path)) {
        cam_params_path = std::filesystem::path(frame_dir).parent_path() / "cam_params.yaml";
    }
    
    if (fileExists(cam_params_path)) {
        if (!loadCameraParams(cam_params_path, cam_params)) {
            std::cerr << "Warning: Failed to load camera parameters from " << cam_params_path << std::endl;
        }
    } else {
        std::cerr << "Warning: Camera parameters file not found" << std::endl;
    }
    
    // Load matrices
    std::vector<std::string> matrix_names = {"K_color", "K_depth", "color2depth"};
    std::vector<cv::Mat*> matrix_ptrs = {&cam_params.K_color, &cam_params.K_depth, &cam_params.color2depth};
    
    for (size_t i = 0; i < matrix_names.size(); ++i) {
        auto files = findFiles(frame_dir, matrix_names[i]);
        if (files.empty()) {
            files = findFiles(std::filesystem::path(frame_dir).parent_path(), matrix_names[i]);
        }
        
        if (!files.empty()) {
            if (!loadMatrix(files[0], *matrix_ptrs[i])) {
                std::cerr << "Warning: Failed to load " << matrix_names[i] << " from " << files[0] << std::endl;
            }
        } else {
            std::cerr << "Warning: Cannot find " << matrix_names[i] << " in " << frame_dir << std::endl;
        }
    }
    
    return true;
}

bool RealsenseReader::loadCameraParams(const std::string& yaml_path, CameraParams& cam_params) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        if (config["model"]) {
            cam_params.model = config["model"].as<std::string>();
        }
        
        if (config["serial_number"]) {
            cam_params.serial_number = config["serial_number"].as<std::string>();
        }
        
        if (config["baseline"]) {
            cam_params.baseline = config["baseline"].as<double>();
        }
        
        if (config["W"]) {
            cam_params.width = config["W"].as<int>();
        }
        
        if (config["H"]) {
            cam_params.height = config["H"].as<int>();
        }
        
        if (config["fps"]) {
            cam_params.fps = config["fps"].as<int>();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return false;
    }
}

bool RealsenseReader::loadMatrix(const std::string& file_path, cv::Mat& matrix) {
    if (!fileExists(file_path)) {
        return false;
    }
    
    std::string extension = std::filesystem::path(file_path).extension();
    
    if (extension == ".txt") {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return false;
        }
        
        std::vector<std::vector<float>> data;
        std::string line;
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<float> row;
            float val;
            
            while (iss >> val) {
                row.push_back(val);
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        
        if (data.empty()) {
            return false;
        }
        
        int rows = data.size();
        int cols = data[0].size();
        matrix = cv::Mat::zeros(rows, cols, CV_32F);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols && j < static_cast<int>(data[i].size()); ++j) {
                matrix.at<float>(i, j) = data[i][j];
            }
        }
        
        return true;
    } else {
        std::cerr << "Unsupported matrix file format: " << extension << std::endl;
        return false;
    }
}
