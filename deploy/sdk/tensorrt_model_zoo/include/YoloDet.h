//#include "common.h"

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YoloDet{
public:
    int init();
    int inference(cv::Mat& img, std::vector<BoxInfo>& objects);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    void postprocess(cv::Mat& frame, const cv::Mat outs, std::vector<BoxInfo>& objects);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    //SampleUniquePtr<nvinfer1::IExecutionContext> context;
    nvinfer1::IExecutionContext* context;
    //std::vector<void*> bindings;
    //cv::Mat pred(cv::Size(8400, 84), CV_32F, {255,255,255});
private:
    float confThreshold = 0.3;
    float scoreThreshold = 0.3;
    float nmsThreshold = 0.4;
    float inpWidth = 640.f;
    float inpHeight = 640.f;
    int classesSize = 5;
    std::string yolo_det_path = "../models/subway5_v11l_640_0.4.15_fp16.trt";
};
