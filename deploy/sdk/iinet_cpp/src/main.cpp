#include "trt_infer.h"
#include "realsense_reader.h"
#include "image_processor.h"
#include <iostream>
#include <fstream>
#include <filesystem>

int main(int argc, char* argv[]) {
    // Default parameters
    std::string engine_path = "./checkpoints/iinet.engine";
    std::string dataset_path = "./20250429_104224_images/20250429_104240_213";
    float ratio = 0.9f;

    // Parse command line arguments (optional)
    // Usage: ./main <engine_path> <dataset_path> <ratio>
    if (argc >= 2) engine_path = argv[1];
    if (argc >= 3) dataset_path = argv[2];
    if (argc >= 4) ratio = std::stof(argv[3]);

    // Check if files exist
    if (!std::filesystem::exists(engine_path)) {
        std::cerr << "Error: Engine file '" << engine_path << "' not found!" << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "Error: Dataset path '" << dataset_path << "' not found!" << std::endl;
        return -1;
    }
    std::string left_img_path = dataset_path + "/left.png";
    std::string right_img_path = dataset_path + "/right.png";
    if (!std::filesystem::exists(left_img_path) || !std::filesystem::exists(right_img_path)) {
        std::cerr << "Error: Left or right image not found!" << std::endl;
        return -1;
    }

    try {
        // 1. Load TensorRT engine
        TRTInference inference;
        if (!inference.loadEngine(engine_path)) {
            std::cerr << "Error: Failed to load TensorRT engine!" << std::endl;
            return -1;
        }

        // 2. Load frame data
        Frame frame;
        CameraParams cam_params;
        if (!RealsenseReader::readFrame(dataset_path, frame, cam_params)) {
            std::cerr << "Error: Failed to read frame data" << std::endl;
            return -1;
        }

        // 3. Preprocess images
        cv::Mat left_img, right_img;
        int top_pad, right_pad;
        if (!ImageProcessor::preprocessImage(left_img_path, left_img, top_pad, right_pad)) {
            std::cerr << "Error: Failed to preprocess left image" << std::endl;
            return -1;
        }
        int dummy_top_pad, dummy_right_pad;
        if (!ImageProcessor::preprocessImage(right_img_path, right_img, dummy_top_pad, dummy_right_pad)) {
            std::cerr << "Error: Failed to preprocess right image" << std::endl;
            return -1;
        }

        // 4. Run inference
        cv::Mat model_output;
        if (!inference.runInference(left_img, right_img, model_output)) {
            std::cerr << "Error: Inference failed" << std::endl;
            return -1;
        }

        // 5. Post-process output to get depth map
        auto input_dims = inference.getInputDimensions();
        cv::Mat depth = inference.postprocessOutput(model_output, input_dims, cam_params, ratio, top_pad, right_pad, frame.rgb);
        
        if (depth.empty()) {
            std::cerr << "Error: Post-processing failed" << std::endl;
            return -1;
        }

        // 6. Save results
        double min_depth, max_depth;
        cv::minMaxLoc(depth, &min_depth, &max_depth);
        

        // Save depth visualization
        cv::Mat depth_vis;
        ImageProcessor::visualizeDepth(depth, depth_vis, static_cast<float>(max_depth));
        cv::imwrite("cpp_depth_visualization.png", depth_vis);

        std::cout << "Inference completed successfully." << std::endl;
        std::cout << "Results saved as 'cpp_depth_visualization.png'" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}