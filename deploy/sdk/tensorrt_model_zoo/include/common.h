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