#include "opencv2/opencv.hpp"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
//#include <NvInferRuntime.h>
#include <random>

#include <fstream>
#include <string>
#include <numeric>
#include <opencv2/dnn.hpp>

#define CHECK(status)                                                                      \
    do                                                                                     \
    {                                                                                      \
        auto ret = (status);                                                               \
        if (ret != 0)                                                                      \
        {                                                                                  \
            std::cerr << "Cuda failure: " << ret << std::endl;                             \
            abort();                                                                       \
        }                                                                                  \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : 
        severity_(severity) {}

    virtual void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if(severity <= severity_)
            std::cout << msg << std::endl;
    }

    nvinfer1::ILogger& getTRTLogger() noexcept
    {
        return *this;
    }
private:
    Severity severity_;
};


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

class YoloDet{
public:
    int init();
    int inference(cv::Mat& img);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    void postprocess(cv::Mat& frame, const cv::Mat outs);
    int del();

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    //SampleUniquePtr<nvinfer1::IExecutionContext> context;
    nvinfer1::IExecutionContext* context;
    //std::vector<void*> bindings;
    //cv::Mat pred(cv::Size(8400, 84), CV_32F, {255,255,255});
private:
    float confThreshold = 0.25f;
    float scoreThreshold = 0.45f;
    float nmsThreshold = 0.5f;
    float inpWidth = 640.f;
    float inpHeight = 640.f;
    int classesSize = 80;
};
