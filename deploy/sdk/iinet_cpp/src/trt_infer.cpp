// Copyright 2019 XingshenTech Inc.
// Version: 2.0
// Author: Meng Deyuan
// Last Edited Date: 2019-06-05
// This file has defined a class to get trt engine

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cfloat>
#include <unordered_map>
#include <chrono>
#include "trt_infer.h"
#include "point_cloud.h"
#include "align_image.h"
#ifdef USE_CUDA_ALIGN
#include "align_image_cuda.h"
#endif
#include "NvInferPlugin.h"

// Logger implementation
void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

// CHECK macro implementation
void TRTInference::CHECK(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

TRTInference::TRTInference() : m_engine_status(TrtUtilsErrorType::INIT_SUCESS)
{
}

TRTInference::~TRTInference()
{
    if(m_stream != nullptr)
        cudaStreamDestroy(m_stream);

    if(std::string(PLATFORM_TYPE) == std::string("aarch64"))
    {
        for(uint i=0; i<m_input_tensor.size(); ++i)
        {
            if(m_input_tensor[i].cpu_memory != nullptr)
                CHECK(cudaFree(m_input_tensor[i].cpu_memory));
        }
        for(uint i=0; i<m_output_tensor.size(); ++i)
        {
            if(m_output_tensor[i].cpu_memory != nullptr)
                CHECK(cudaFree(m_output_tensor[i].cpu_memory));
        }
    }
    else
    {
        for(uint i=0; i<m_input_tensor.size(); ++i)
        {
            if(m_input_tensor[i].cpu_memory != nullptr)
                CHECK(cudaFreeHost(m_input_tensor[i].cpu_memory));
            if(m_input_tensor[i].gpu_memory != nullptr)
                CHECK(cudaFree(m_input_tensor[i].gpu_memory));
        }
        for(uint i=0; i<m_output_tensor.size(); ++i)
        {
            if(m_output_tensor[i].cpu_memory != nullptr)
                CHECK(cudaFreeHost(m_output_tensor[i].cpu_memory));
            if(m_output_tensor[i].gpu_memory != nullptr)
                CHECK(cudaFree(m_output_tensor[i].gpu_memory));
        }
    }
}

bool TRTInference::loadEngine(const std::string& engine_path)
{
    m_engine_status = GetEngine(engine_path);
    return m_engine_status == TrtUtilsErrorType::INIT_SUCESS;
}

bool TRTInference::runInference(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& output)
{
    if (m_engine_status != TrtUtilsErrorType::INIT_SUCESS)
    {
        std::cerr << "Engine not initialized properly" << std::endl;
        return false;
    }
    
    try
    {
        // Copy input images to tensor (optimized)
        copyImageToInputTensor(left_img, right_img);
        
        // Run inference with precise timing (matching Python version)
        DoInference(1);
        
        // Extract output tensor to cv::Mat (optimized)
        output = extractOutputTensor();
        
        return !output.empty();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int> TRTInference::getInputDimensions()
{
    if (m_input_tensor.empty())
        return {};
    
    return m_input_tensor[0].shape;  // Return first input tensor shape
}

cv::Mat TRTInference::postprocessOutput(const cv::Mat& model_output, const std::vector<int>& input_dims,
                                       const CameraParams& cam_params, float ratio, int top_pad, int right_pad,
                                       const cv::Mat& rgb_img)
{
    try
    {
        // Extract disparity from model output
        cv::Mat disparity;
        if (model_output.channels() == 1)
        {
            disparity = model_output.clone();
        }
        else
        {
            std::vector<cv::Mat> channels;
            cv::split(model_output, channels);
            disparity = channels[0];
        }
        
        // Remove padding (exactly matching Python)
        int original_height = disparity.rows - top_pad;
        int original_width  = (right_pad != 0) ? (disparity.cols - right_pad) : disparity.cols;
        cv::Rect crop_rect(0, top_pad, original_width, original_height);  
        disparity = disparity(crop_rect);

        // Scale disparity - exactly matching Python version
        disparity *= 16.0f;
        
        // Convert disparity to depth - using K_depth focal length (matching Python)
        cv::Mat depth;
        float focal_length = cam_params.K_depth.at<float>(0, 0);
        float baseline = cam_params.baseline;
        
        disparity.convertTo(depth, CV_32F);
        
        // Avoid division by zero - set very small disparity values to 0 (matching Python 1e-6)
        cv::Mat valid_disp_mask = disparity > 1e-6f;
        cv::divide(focal_length * baseline, depth, depth);
        depth.setTo(0, ~valid_disp_mask);
        
        // Apply ratio scaling (matching Python)
        depth *= ratio;
        
        // Resize to match RGB image size if needed (ensure exact matching)
        if (depth.size() != rgb_img.size())
        {
            cv::resize(depth, depth, rgb_img.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Apply the alignment algorithm (CUDA if available, CPU fallback)
#ifdef USE_CUDA_ALIGN
        static AlignImageCUDA cuda_aligner;
        cv::Mat aligned_depth = cuda_aligner.alignImageByDepth(
            rgb_img, depth, 
            cam_params.K_color, cam_params.K_depth, 
            cam_params.color2depth,
            false,  // to_depth = false (align depth to RGB frame, matching Python)
            true    // fill_zero = true (matching Python)
        );
#else
        cv::Mat aligned_depth = AlignImage::alignImageByDepth(
            rgb_img, depth, 
            cam_params.K_color, cam_params.K_depth, 
            cam_params.color2depth,
            false,  // to_depth = false (align depth to RGB frame, matching Python)
            true    // fill_zero = true (matching Python)
        );
#endif
        
        return aligned_depth;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in postprocessOutput: " << e.what() << std::endl;
        return cv::Mat();
    }
}

void TRTInference::copyImageToInputTensor(const cv::Mat& left_img, const cv::Mat& right_img)
{
    if (m_input_tensor.size() < 2)
    {
        throw std::runtime_error("Expected at least 2 input tensors (left and right)");
    }
    
    // Find left and right input tensors
    int left_idx = -1, right_idx = -1;
    for (size_t i = 0; i < m_input_tensor.size(); ++i)
    {
        if (m_input_tensor[i].name == "left")
        {
            left_idx = i;
        }
        else if (m_input_tensor[i].name == "right")
        {
            right_idx = i;
        }
    }
    
    if (left_idx == -1 || right_idx == -1)
    {
        throw std::runtime_error("Could not find left and right input tensors");
    }
    
    // Get tensor shapes
    auto& left_tensor = m_input_tensor[left_idx];
    auto& right_tensor = m_input_tensor[right_idx];
    
    // Copy left image data (already preprocessed)
    float* left_data = static_cast<float*>(left_tensor.cpu_memory);
    copyPreprocessedImageToTensor(left_img, left_data, left_tensor.shape);
    
    // Copy right image data (already preprocessed)
    float* right_data = static_cast<float*>(right_tensor.cpu_memory);
    copyPreprocessedImageToTensor(right_img, right_data, right_tensor.shape);
}

void TRTInference::copyImageDataToTensor(const cv::Mat& img, float* tensor_data, const std::vector<int>& shape)
{
    // Assuming tensor format is [batch, channels, height, width] = [1, 3, H, W]
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3)
    {
        throw std::runtime_error("Expected tensor shape [1, 3, H, W]");
    }
    
    int height = shape[2];
    int width = shape[3];
    int channels = shape[1];
    
    // Verify image dimensions match tensor
    if (img.rows != height || img.cols != width)
    {
        throw std::runtime_error("Image dimensions don't match tensor shape");
    }
    
    // Copy data in CHW format (channels first)
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                tensor_data[c * height * width + h * width + w] = 
                    img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

void TRTInference::copyPreprocessedImageToTensor(const cv::Mat& preprocessed_img, float* tensor_data, const std::vector<int>& shape)
{
    // The preprocessed image is in format [C*H, W] from ImageProcessor
    // Tensor expects format [1, C, H, W]
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3)
    {
        throw std::runtime_error("Expected tensor shape [1, 3, H, W]");
    }
    
    int height = shape[2];
    int width = shape[3];
    int channels = shape[1];
    
    // Verify dimensions match
    if (preprocessed_img.rows != channels * height || preprocessed_img.cols != width)
    {
        std::cout << "Preprocessed image shape: [" << preprocessed_img.cols << " x " << preprocessed_img.rows << "]" << std::endl;
        std::cout << "Expected shape: [" << width << " x " << (channels * height) << "]" << std::endl;
        throw std::runtime_error("Preprocessed image dimensions don't match expected tensor shape");
    }
    
    // The preprocessed image is already in CHW format, just copy the data
    const float* img_data = reinterpret_cast<const float*>(preprocessed_img.data);
    std::memcpy(tensor_data, img_data, preprocessed_img.total() * sizeof(float));
}

cv::Mat TRTInference::extractOutputTensor()
{
    if (m_output_tensor.empty())
    {
        std::cerr << "No output tensors available" << std::endl;
        return cv::Mat();
    }
    
    // Find the output tensor (should be "disp_pred_s0" based on the log)
    int output_idx = -1;
    for (size_t i = 0; i < m_output_tensor.size(); ++i)
    {
        if (m_output_tensor[i].name.find("disp") != std::string::npos)
        {
            output_idx = i;
            break;
        }
    }
    
    if (output_idx == -1)
    {
        output_idx = 0;  // Default to first output
    }
    
    const auto& output_tensor = m_output_tensor[output_idx];
    const float* output_data = static_cast<const float*>(output_tensor.cpu_memory);
    
    // Handle different tensor formats
    if (output_tensor.shape.size() == 4)
    {
        // Format: [batch, channels, height, width]
        int batch = output_tensor.shape[0];
        int channels = output_tensor.shape[1];
        int height = output_tensor.shape[2];
        int width = output_tensor.shape[3];
        
        if (channels == 1)
        {
            return cv::Mat(height, width, CV_32F, (void*)output_data).clone();
        }
        else
        {
            // Multi-channel output - take first channel as disparity
            cv::Mat output(height, width, CV_32F);
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    output.at<float>(h, w) = output_data[h * width + w];
                }
            }
            return output;
        }
    }
    else if (output_tensor.shape.size() == 3)
    {
        // Format: [channels, height, width]
        int channels = output_tensor.shape[0];
        int height = output_tensor.shape[1];
        int width = output_tensor.shape[2];
        
        if (channels == 1)
        {
            return cv::Mat(height, width, CV_32F, (void*)output_data).clone();
        }
        else
        {
            // Take first channel
            cv::Mat output(height, width, CV_32F);
            std::memcpy(output.data, output_data, height * width * sizeof(float));
            return output;
        }
    }
    else
    {
        std::cerr << "Unsupported output tensor shape" << std::endl;
        return cv::Mat();
    }
}

TrtUtilsErrorType TRTInference::GetEngine(const std::string& engine_path)
{
    // 0. 初始化TensorRT插件库
    if (!initLibNvInferPlugins(&m_logger, "")) {
        std::cout << "-----------could not initialize TensorRT plugins, exit-----------" << std::endl;
        return TrtUtilsErrorType::RUNTIME_ERROR;
    }
    
    // 1. 检查engine文件是否存在
    if (engine_path.empty())
    {
        std::cout << "-----------could not find model file, exit-----------" << std::endl;
        return TrtUtilsErrorType::FILE_LOST;
    }

    // 2. 从序列化plan文件中反序列化engine
    std::ifstream cache(engine_path);
    if(!cache.is_open())
    {
        std::cout << "-----------could not open file model, exit-----------" << std::endl;
        return TrtUtilsErrorType::FILE_ERROR;
    }

    // 2.2 从文件中读取字节流
    std::stringstream gieplanstream;
    gieplanstream << cache.rdbuf();
    cache.close();
    gieplanstream.seekg(0, std::ios::end);
    int modelsize = gieplanstream.tellg();
    gieplanstream.seekg(0, std::ios::beg);
    void* modelmem = malloc(modelsize);
    gieplanstream.read((char*)modelmem, modelsize);

    // 3. 反序列化模型
    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if(!m_runtime)
    {
        free(modelmem);
        std::cout << "-----------could not create runtime, exit-----------" << std::endl;
        return TrtUtilsErrorType::RUNTIME_ERROR;
    }
    
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(modelmem, modelsize));
    free(modelmem);
    if(!m_engine)
    {
        std::cout << "-----------could not create engine, exit-----------" << std::endl;
        return TrtUtilsErrorType::ENGINE_ERROR;
    }

    // 4. 获取网络输入输出
    std::cout << "IOTensor:" << m_engine->getNbIOTensors() << std::endl;
    for(int i=0; i<m_engine->getNbIOTensors(); ++i)
    {
        std::string name = m_engine->getIOTensorName(i);
        InOutTensorTRT io_tensor;
        io_tensor.name = name;
        io_tensor.cpu_memory = nullptr;
        io_tensor.gpu_memory = nullptr;
        nvinfer1::Dims io_dim = m_engine->getTensorShape(name.c_str());
        int size = 1;
        for(int j=0; j<io_dim.nbDims; ++j)
        {
            io_tensor.shape.push_back(io_dim.d[j]);
            size = size * io_dim.d[j];
        }
        io_tensor.size = size;
        io_tensor.data_type = m_engine->getTensorDataType(name.c_str());
        io_tensor.data_size_of = GetElementSize(io_tensor.data_type);
        
        switch (m_engine->getTensorIOMode(name.c_str()))
        {
        case nvinfer1::TensorIOMode::kINPUT:
            m_input_tensor.push_back(io_tensor);
            break;
        case nvinfer1::TensorIOMode::kOUTPUT:
            m_output_tensor.push_back(io_tensor);
            break;
        default:
            break;
        }
    }
    
    // 4.2 初始化输入输出所需的内存
    if(std::string(PLATFORM_TYPE) == std::string("aarch64"))
    {
        // malloc input data
        for(uint i=0; i<m_input_tensor.size(); ++i)
        {
            std::cout << m_input_tensor[i].name << " / " << m_input_tensor[i].size << " / "
                      << int(m_input_tensor[i].data_type) << " / " << m_input_tensor[i].data_size_of << std::endl;
            m_input_tensor[i].data_len = m_input_tensor[i].size * m_input_tensor[i].data_size_of;
            CHECK(cudaMallocManaged((void **)&m_input_tensor[i].cpu_memory, m_input_tensor[i].data_len, cudaMemAttachGlobal));
            cudaMemset((void *)m_input_tensor[i].cpu_memory, 0, m_input_tensor[i].data_len);
            cudaStreamAttachMemAsync(NULL, (void *)m_input_tensor[i].cpu_memory, 0, cudaMemAttachHost);
            cudaStreamSynchronize(NULL);
        }
        // malloc output data
        for(uint i=0; i<m_output_tensor.size(); ++i)
        {
            std::cout << m_output_tensor[i].name << " / " << m_output_tensor[i].size << " / "
                      << int(m_output_tensor[i].data_type) << " / " << m_output_tensor[i].data_size_of << std::endl;
            m_output_tensor[i].data_len = m_output_tensor[i].size * m_output_tensor[i].data_size_of;
            CHECK(cudaMallocManaged((void **)&m_output_tensor[i].cpu_memory, m_output_tensor[i].data_len, cudaMemAttachGlobal));
            cudaMemset((void *)m_output_tensor[i].cpu_memory, 0, m_output_tensor[i].data_len);
            cudaStreamAttachMemAsync(NULL, (void *)m_output_tensor[i].cpu_memory, 0, cudaMemAttachHost);
            cudaStreamSynchronize(NULL);
        }
    }
    else
    {
        // malloc input data
        for(uint i=0; i<m_input_tensor.size(); ++i)
        {
            std::cout << m_input_tensor[i].name << " / " << m_input_tensor[i].size << " / "
                      << int(m_input_tensor[i].data_type) << " / " << m_input_tensor[i].data_size_of << std::endl;
            m_input_tensor[i].data_len = m_input_tensor[i].size * m_input_tensor[i].data_size_of;
            CHECK(cudaMallocHost((void **)&m_input_tensor[i].cpu_memory, m_input_tensor[i].data_len));
            memset((void *)m_input_tensor[i].cpu_memory, 0, m_input_tensor[i].data_len);
            CHECK(cudaMalloc((void **)&m_input_tensor[i].gpu_memory, m_input_tensor[i].data_len));
        }

        for(uint i=0; i<m_output_tensor.size(); ++i)
        {
            std::cout << m_output_tensor[i].name << " / " << m_output_tensor[i].size << " / "
                      << int(m_output_tensor[i].data_type) << " / " << m_output_tensor[i].data_size_of << std::endl;
            m_output_tensor[i].data_len = m_output_tensor[i].size * m_output_tensor[i].data_size_of;
            CHECK(cudaMallocHost((void **)&m_output_tensor[i].cpu_memory, m_output_tensor[i].data_len));
            memset((void *)m_output_tensor[i].cpu_memory, 0, m_output_tensor[i].data_len);
            CHECK(cudaMalloc((void **)&m_output_tensor[i].gpu_memory, m_output_tensor[i].data_len));
        }
    }

    // 5. 创建cuda stream
    CHECK(cudaStreamCreate(&m_stream));
    
    // 6. 创建context
    std::cout << "Build context from plan deserializeCudaEngine." << std::endl;
    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if(!m_context)
    {
        std::cout << "-----------could not create context, exit-----------" << std::endl;
        return TrtUtilsErrorType::CONTEXT_ERROR;
    }
    
    // 7. 优化TensorRT性能设置
    // 设置优化profile (如果引擎支持)
    if (m_engine->getNbOptimizationProfiles() > 0) {
        m_context->setOptimizationProfileAsync(0, m_stream);
    }
    
    // 启用CUDA图优化 (TensorRT 8.5+)
    m_context->setEnqueueEmitsProfile(false);  // 减少profiling开销

    return TrtUtilsErrorType::INIT_SUCESS;
}

void TRTInference::DoInference(int batch_size)
{
    if(std::string(PLATFORM_TYPE) == std::string("aarch64"))
    {
        // 将输入输出UM转到全局，用于CUDA交互
        for(uint i=0; i<m_input_tensor.size(); ++i)
            cudaStreamAttachMemAsync(m_stream, m_input_tensor[i].cpu_memory, 0, cudaMemAttachGlobal);
        for(uint i=0; i<m_output_tensor.size(); ++i)
            cudaStreamAttachMemAsync(m_stream, m_output_tensor[i].cpu_memory, 0, cudaMemAttachGlobal);
        // TRT推理
        for(uint i=0; i<m_input_tensor.size(); ++i)
            m_context->setTensorAddress(m_input_tensor[i].name.c_str(), m_input_tensor[i].cpu_memory);
        for(uint i=0; i<m_output_tensor.size(); ++i)
            m_context->setTensorAddress(m_output_tensor[i].name.c_str(), m_output_tensor[i].cpu_memory);
        m_context->enqueueV3(m_stream);
        // 输出UM转换到本地，用于读取数据
        for(uint i=0; i<m_input_tensor.size(); ++i)
            cudaStreamAttachMemAsync(m_stream, m_input_tensor[i].cpu_memory, 0, cudaMemAttachHost);
        for(uint i=0; i<m_output_tensor.size(); ++i)
            cudaStreamAttachMemAsync(m_stream, m_output_tensor[i].cpu_memory, 0, cudaMemAttachHost);
        // 流同步
        cudaStreamSynchronize(m_stream);
    }
    else
    {
        // 输入Host to Device
        for(uint i=0; i<m_input_tensor.size(); ++i)
        {
            CHECK(cudaMemcpyAsync(m_input_tensor[i].gpu_memory, m_input_tensor[i].cpu_memory,
                                  batch_size*m_input_tensor[i].data_len, cudaMemcpyHostToDevice, m_stream));
        }
        // gpu推理
        for(uint i=0; i<m_input_tensor.size(); ++i)
            m_context->setTensorAddress(m_input_tensor[i].name.c_str(), m_input_tensor[i].gpu_memory);
        for(uint i=0; i<m_output_tensor.size(); ++i)
            m_context->setTensorAddress(m_output_tensor[i].name.c_str(), m_output_tensor[i].gpu_memory);
        m_context->enqueueV3(m_stream);
        // 输出Device to Host
        for(uint i=0; i<m_output_tensor.size(); ++i)
        {
            CHECK(cudaMemcpyAsync(m_output_tensor[i].cpu_memory, m_output_tensor[i].gpu_memory,
                                  batch_size*m_output_tensor[i].data_len, cudaMemcpyDeviceToHost, m_stream));
        }
        // 流同步
        cudaStreamSynchronize(m_stream);
    }
}

void TRTInference::GetIOTensor(std::vector<void *> &input_tensor, std::vector<void *> &output_tensor)
{
    for(uint i=0; i<m_input_tensor.size(); ++i)
    {
        input_tensor.push_back(m_input_tensor[i].cpu_memory);
    }
    for(uint i=0; i<m_output_tensor.size(); ++i)
    {
        output_tensor.push_back(m_output_tensor[i].cpu_memory);
    }
}

uint32_t TRTInference::GetElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4: return 1; // Note: Packed format, but minimum 1 byte
    }
    return 1;
}

cv::Mat TRTInference::replaceZerosWithNeighbors(const cv::Mat& input)
{
    cv::Mat result = input.clone();
    cv::Mat mask = (input == 0);
    
    // 3x3 kernel for neighbors (excluding center)
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
    
    // Replace zeros with neighbor average where possible
    cv::Mat valid_avg_mask = (neighbor_count > 0) & mask;
    neighbor_avg.copyTo(result, valid_avg_mask);
    
    return result;
}

cv::Mat TRTInference::alignImageByDepth(const cv::Mat& rgb, const cv::Mat& depth,
                                       const cv::Mat& K_rgb, const cv::Mat& K_depth,
                                       const cv::Mat& rgb2depth, bool to_depth, bool fill_zero)
{
    int height = depth.rows;
    int width = depth.cols;
    
    if (!to_depth)
    {
        // Convert matrices to double for precision
        cv::Mat K_depth_d, K_rgb_d, rgb2depth_d;
        K_depth.convertTo(K_depth_d, CV_64F);
        K_rgb.convertTo(K_rgb_d, CV_64F);
        rgb2depth.convertTo(rgb2depth_d, CV_64F);
        
        cv::Mat K_depth_inv, rgb2depth_inv;
        cv::invert(K_depth_d, K_depth_inv);
        cv::invert(rgb2depth_d, rgb2depth_inv);
        
        // Create coordinate meshgrid using OpenCV operations
        std::vector<cv::Point2f> grid_points;
        std::vector<float> depth_values;
        
        // Collect valid points and their depths
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                float d = depth.at<float>(v, u);
                if (d > 0) {
                    grid_points.push_back(cv::Point2f(u, v));
                    depth_values.push_back(d);
                }
            }
        }
        
        if (grid_points.empty()) {
            return cv::Mat::zeros(height, width, CV_32F);
        }
        
        int num_points = grid_points.size();
        
        // Vectorized computation using OpenCV matrix operations
        cv::Mat pixels_b = cv::Mat::zeros(3, num_points, CV_64F);
        
        // Fill pixel coordinates and multiply by depth
        for (int i = 0; i < num_points; i++) {
            double d = depth_values[i];
            pixels_b.at<double>(0, i) = grid_points[i].x * d;
            pixels_b.at<double>(1, i) = grid_points[i].y * d;
            pixels_b.at<double>(2, i) = d;
        }
        
        // Vectorized 3D projection
        cv::Mat points_3D_b = K_depth_inv * pixels_b;
        
        // Add homogeneous coordinates
        cv::Mat ones_row = cv::Mat::ones(1, num_points, CV_64F);
        cv::Mat points_3D_b_h;
        cv::vconcat(points_3D_b, ones_row, points_3D_b_h);
        
        // Vectorized coordinate transformation
        cv::Mat points_3D_a_h = rgb2depth_inv * points_3D_b_h;
        cv::Mat points_3D_a = points_3D_a_h.rowRange(0, 3);
        
        // Vectorized projection to camera A
        cv::Mat pixels_a_h = K_rgb_d * points_3D_a;
        
        // Extract projected coordinates
        cv::Mat x_coords = pixels_a_h.row(0);
        cv::Mat y_coords = pixels_a_h.row(1); 
        cv::Mat z_coords = pixels_a_h.row(2);
        
        // Normalize coordinates (vectorized division)
        cv::divide(x_coords, z_coords, x_coords);
        cv::divide(y_coords, z_coords, y_coords);
        
        // Initialize result depth map
        cv::Mat depth_a = cv::Mat::zeros(height, width, CV_32F);
        
        // Use hash map for efficient minimum depth assignment
        std::unordered_map<int, float> min_depth_map;
        
        // Process all projected points efficiently
        for (int i = 0; i < num_points; i++) {
            double x_a = x_coords.at<double>(0, i);
            double y_a = y_coords.at<double>(0, i);
            double depth_val = z_coords.at<double>(0, i);
            
            if (depth_val <= 0) continue;
            
            int map_x = static_cast<int>(std::round(x_a));
            int map_y = static_cast<int>(std::round(y_a));
            
            if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                int key = map_y * width + map_x;
                auto it = min_depth_map.find(key);
                if (it == min_depth_map.end() || depth_val < it->second) {
                    min_depth_map[key] = static_cast<float>(depth_val);
                }
            }
        }
        
        // Copy results to depth map
        for (const auto& pair : min_depth_map) {
            int y = pair.first / width;
            int x = pair.first % width;
            depth_a.at<float>(y, x) = pair.second;
        }
        
        // Fill zeros with neighbor average if requested
        if (fill_zero) {
            depth_a = replaceZerosWithNeighbors(depth_a);
        }
        
        return depth_a;
    }
    else
    {
        return rgb.clone();
    }
}
