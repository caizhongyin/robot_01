//#include "common.h"

class Class{
public:
    int init();
    int inference(cv::Mat& img);
    int postprocess(cv::Mat& frame, const cv::Mat tmp);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    //SampleUniquePtr<nvinfer1::IExecutionContext> context;
    nvinfer1::IExecutionContext* context;
    //std::vector<void*> bindings;
    //cv::Mat pred(cv::Size(8400, 84), CV_32F, {255,255,255});
private:
    float confThreshold = 0.3;
    float scoreThreshold = 0.3;
    float nmsThreshold = 0.4;
    float inpWidth = 112.f;
    float inpHeight = 112.f;
    int classesSize = 6;
    std::string class_model_path = "../models/s009_cls6_250828.trt";
    //std::vector<float> cls_mean = {0.485, 0.456, 0.406}; //rgb
    //std::vector<float> cls_std = {0.229, 0.224, 0.225}; //rgb
    std::vector<float> cls_mean = {0.406, 0.456, 0.485}; //bgr
    std::vector<float> cls_std = {0.225, 0.224, 0.229}; //bgr
};
