#include "trt_infer.h"
#include "realsense_reader.h"
#include "image_processor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <iomanip>

int main(int argc, char* argv[]) {
    // Default parameters
    std::string engine_path = "./checkpoints/iinet.engine";
    std::string dataset_path = "./20250429_104224_images/20250429_104240_213";
    float ratio = 0.9f;
    int num_runs = 20;

    // Parse command line arguments (optional)
    if (argc >= 2) engine_path = argv[1];
    if (argc >= 3) dataset_path = argv[2];
    if (argc >= 4) ratio = std::stof(argv[3]);
    if (argc >= 5) num_runs = std::stoi(argv[4]);

    // Check if files exist
    if (!std::filesystem::exists(engine_path)) {
        std::cerr << "Error: Engine file '" << engine_path << "' not found!" << std::endl;
        std::cerr << "Please make sure the TensorRT engine file exists." << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "Error: Dataset path '" << dataset_path << "' not found!" << std::endl;
        std::cerr << "Please provide a valid dataset path with left.png and right.png files." << std::endl;
        return -1;
    }
    std::string left_img_path = dataset_path + "/left.png";
    std::string right_img_path = dataset_path + "/right.png";
    if (!std::filesystem::exists(left_img_path)) {
        std::cerr << "Error: Left image '" << left_img_path << "' not found!" << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(right_img_path)) {
        std::cerr << "Error: Right image '" << right_img_path << "' not found!" << std::endl;
        return -1;
    }

    try {
        // 1. Load TensorRT engine once
        auto engine_load_start = std::chrono::high_resolution_clock::now();
        std::cout << "Loading TensorRT engine..." << std::endl;
        TRTInference inference;
        if (!inference.loadEngine(engine_path)) {
            std::cerr << "Error: Failed to load TensorRT engine!" << std::endl;
            return -1;
        }
        auto engine_load_end = std::chrono::high_resolution_clock::now();
        auto engine_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(engine_load_end - engine_load_start);

        // 2. Load data once (reuse for all measurements)
        std::cout << "Loading frame data..." << std::endl;
        Frame frame;
        CameraParams cam_params;
        if (!RealsenseReader::readFrame(dataset_path, frame, cam_params)) {
            std::cerr << "Error: Failed to read frame data" << std::endl;
            return -1;
        }

        // 3. Preprocess images once (reuse for all measurements) 
        std::cout << "Preprocessing images..." << std::endl;
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

        // 4. Warmup inference (discard timing)
        std::cout << "Warming up inference..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            cv::Mat warmup_output;
            inference.runInference(left_img, right_img, warmup_output);
        }

        // 5. Measure inference timing
        std::cout << "Measuring inference performance (" << num_runs << " runs)..." << std::endl;
        std::vector<double> inference_times;
        std::vector<double> postprocess_times;
        inference_times.reserve(num_runs);
        postprocess_times.reserve(num_runs);

        cv::Mat final_depth;
        for (int i = 0; i < num_runs; ++i) {
            // Measure inference time
            auto infer_start = std::chrono::high_resolution_clock::now();
            cv::Mat model_output;
            if (!inference.runInference(left_img, right_img, model_output)) {
                std::cerr << "Error: Inference failed on run " << i << std::endl;
                return -1;
            }
            auto infer_end = std::chrono::high_resolution_clock::now();
            auto infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start);
            inference_times.push_back(infer_time.count());

            // Measure postprocessing time
            auto post_start = std::chrono::high_resolution_clock::now();
            auto input_dims = inference.getInputDimensions();
            cv::Mat depth = inference.postprocessOutput(model_output, input_dims, cam_params, ratio, top_pad, right_pad, frame.rgb);
            auto post_end = std::chrono::high_resolution_clock::now();
            auto post_time = std::chrono::duration<double, std::milli>(post_end - post_start);
            postprocess_times.push_back(post_time.count());

            if (depth.empty()) {
                std::cerr << "Error: Post-processing failed on run " << i << std::endl;
                return -1;
            }

            // Keep the last result for saving
            if (i == num_runs - 1) {
                final_depth = depth.clone();
            }
        }

        // 6. Calculate statistics
        auto calc_stats = [](const std::vector<double>& times) {
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            double variance = 0.0;
            for (const auto& t : times) {
                variance += (t - mean) * (t - mean);
            }
            variance /= times.size();
            double std_dev = std::sqrt(variance);
            return std::make_pair(mean, std_dev);
        };

        auto [inf_mean, inf_std] = calc_stats(inference_times);
        auto [post_mean, post_std] = calc_stats(postprocess_times);

        // 7. Save final results
        std::cout << "Saving final results..." << std::endl;
        double min_depth, max_depth;
        cv::minMaxLoc(final_depth, &min_depth, &max_depth);
        
        // Save binary depth
        std::ofstream depth_bin("cpp_depth.bin", std::ios::out | std::ios::binary);
        if (depth_bin.is_open()) {
            if (final_depth.isContinuous()) {
                depth_bin.write(reinterpret_cast<const char*>(final_depth.ptr<float>(0)), 
                               final_depth.rows * final_depth.cols * sizeof(float));
            } else {
                for (int i = 0; i < final_depth.rows; ++i) {
                    depth_bin.write(reinterpret_cast<const char*>(final_depth.ptr<float>(i)), 
                                   final_depth.cols * sizeof(float));
                }
            }
            depth_bin.close();
        } else {
            std::cerr << "Failed to save depth to 'cpp_depth.bin'" << std::endl;
        }

        // Save depth visualization
        auto save_start = std::chrono::high_resolution_clock::now();
        cv::Mat depth_vis;
        ImageProcessor::visualizeDepth(final_depth, depth_vis, static_cast<float>(max_depth));
        cv::imwrite("depth_visualization.png", depth_vis);
        auto save_end = std::chrono::high_resolution_clock::now();
        auto save_time = std::chrono::duration<double, std::milli>(save_end - save_start);
        std::cout << "Saving time: " << save_time.count() << " ms" << std::endl;

        // 8. Print comprehensive performance report
        std::cout << std::endl << "=== C++ IINet TensorRT Performance Analysis ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Engine loading time: " << engine_load_time.count() << " ms" << std::endl;
        std::cout << std::endl << "Performance measurements (" << num_runs << " runs):" << std::endl;
        std::cout << "Inference time (GPU compute):" << std::endl;
        std::cout << "  Average: " << inf_mean << " ms" << std::endl;
        std::cout << "  Std dev: " << inf_std << " ms" << std::endl;
        std::cout << "Post-processing time (depth alignment):" << std::endl;
        std::cout << "  Average: " << post_mean << " ms" << std::endl;
        std::cout << "  Std dev: " << post_std << " ms" << std::endl;
        std::cout << "Total pipeline time:" << std::endl;
        std::cout << "  Average: " << (inf_mean + post_mean) << " ms" << std::endl;
        std::cout << std::endl << "Generated depth map:" << std::endl;
        std::cout << "  Shape: [" << final_depth.rows << " x " << final_depth.cols << "] (H x W)" << std::endl;
        std::cout << "  Depth range: [" << min_depth << ", " << max_depth << "] meters" << std::endl;
        std::cout << "  Saved as: 'cpp_depth.bin' and 'depth_visualization.png'" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}