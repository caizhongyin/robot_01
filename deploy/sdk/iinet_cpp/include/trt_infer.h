#ifndef TRT_INFER_H
#define TRT_INFER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "realsense_reader.h"

struct InOutTensorTRT
{
    std::string name;
    std::vector<int> shape;
    int size;
    nvinfer1::DataType data_type;
    int data_size_of;
    int data_len;
    void* cpu_memory;
    void* gpu_memory;
};

enum class TrtUtilsErrorType : int32_t
{
    INIT_SUCESS = 0,
    FILE_LOST = 1,
    FILE_ERROR = 2,
    RUNTIME_ERROR = 3,
    ENGINE_ERROR = 4,
    CONTEXT_ERROR = 5
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TRTInference
{
public:
    TRTInference();
    ~TRTInference();
    
    // Interface methods expected by main.cpp
    bool loadEngine(const std::string& engine_path);
    bool runInference(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& output);
    std::vector<int> getInputDimensions();
    cv::Mat postprocessOutput(const cv::Mat& model_output, const std::vector<int>& input_dims,
                             const CameraParams& cam_params, float ratio, int top_pad, int right_pad,
                             const cv::Mat& rgb_img);
    
    // Camera alignment functionality
    cv::Mat alignImageByDepth(const cv::Mat& rgb, const cv::Mat& depth, 
                             const cv::Mat& K_rgb, const cv::Mat& K_depth, 
                             const cv::Mat& rgb2depth, bool to_depth = false, bool fill_zero = true);
    cv::Mat replaceZerosWithNeighbors(const cv::Mat& input);
    
    // Internal methods adapted from trt8_utils
    TrtUtilsErrorType GetEngine(const std::string& engine_path);
    void DoInference(int batch_size);
    void GetIOTensor(std::vector<void *> &input_tensor, std::vector<void *> &output_tensor);

private:
    inline uint32_t GetElementSize(nvinfer1::DataType t);
    void copyImageToInputTensor(const cv::Mat& left_img, const cv::Mat& right_img);
    cv::Mat extractOutputTensor();
    void copyImageDataToTensor(const cv::Mat& img, float* tensor_data, const std::vector<int>& shape);
    void copyPreprocessedImageToTensor(const cv::Mat& preprocessed_img, float* tensor_data, const std::vector<int>& shape);

private:  
    std::vector<InOutTensorTRT> m_input_tensor;
    std::vector<InOutTensorTRT> m_output_tensor;
    int m_max_batchsize = 1;
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_stream = nullptr;
    TrtUtilsErrorType m_engine_status;
    Logger m_logger;
    
    // For CHECK macro
    void CHECK(cudaError_t status);
};

#endif // TRT_INFER_H
