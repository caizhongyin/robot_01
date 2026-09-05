#include "common.h"
#include "Class.h"

cv::Mat standardizeImage(const cv::Mat& inputImage, const std::vector<float>& mean, const std::vector<float>& std) {
    // 确保输入图像是浮点类型（通常是预处理后的类型）
    cv::Mat imageFloat;
    inputImage.convertTo(imageFloat, CV_32FC3);
    
    // 确保均值和标准差向量有3个值（对应RGB三通道）
    if (mean.size() != 3 || std.size() != 3) {
        throw std::invalid_argument("Mean and std must have 3 values each for RGB channels");
    }
    
    // 将图像拆分为单独通道
    std::vector<cv::Mat> channels;
    cv::split(imageFloat, channels);
    
    // 对每个通道进行标准化：(x - mean) / std
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] / 255. - mean[i]) / std[i];
    }
    
    // 合并通道
    cv::Mat standardizedImage;
    cv::merge(channels, standardizedImage);
    
    return standardizedImage;
}

int Class::init(){
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    std::ifstream ifs(class_model_path, std::ifstream::binary);
    ifs.seekg(0, std::ios_base::end);
    int size = ifs.tellg();
    ifs.seekg(0, std::ios_base::beg);
    std::unique_ptr<char> pData(new char[size]);
    ifs.read(pData.get(), size);
    ifs.close();
    //std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    SampleUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger.getTRTLogger())};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(pData.get(), size), InferDeleter());
    //auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    context = mEngine->createExecutionContext();
    // 显存分配
    //std::vector<void*> bindings(mEngine->getNbIOTensors());
    //CHECK(cudaMalloc(&bindings[0], sizeof(float) * 1 * 3 * inpHeight * inpWidth)); // type: float32[1,3,640,640]
    //CHECK(cudaMalloc(&bindings[1], sizeof(int) * 1 * classesSize));   // type: float32[1,84,8400]
    return 0;
}

int Class::inference(cv::Mat& img)
{
    std::vector<void*> bindings(mEngine->getNbIOTensors());
    //bindings = bindings(mEngine->getNbIOTensors());
    CHECK(cudaMalloc(&bindings[0], sizeof(float) * 1 * 3 * inpHeight * inpWidth)); // type: float32[1,3,640,640]
    CHECK(cudaMalloc(&bindings[1], sizeof(int) * 1 * classesSize));   // type: float32[1,84,8400]
    // 输入
    //cv::Mat std_img = standardizeImage(img, cls_mean, cls_std);
    //cv::Mat blob = cv::dnn::blobFromImage(std_img, 1., cv::Size(inpWidth,inpHeight), {0,0,0}, true, false);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255., cv::Size(inpWidth,inpHeight), {0,0,0}, true, false);

    cv::Mat pred(cv::Size(1, classesSize), CV_32F, {255,255,255});
    // 推理
    auto t1 = cv::getTickCount();

    CHECK(cudaMemcpy(bindings[0], static_cast<const void*>(blob.data), 1 * 3 * inpHeight * inpWidth* sizeof(float), cudaMemcpyHostToDevice));
    context->executeV2(bindings.data());
    CHECK(cudaMemcpy(static_cast<void*>(pred.data), bindings[1], 1 * classesSize * sizeof(float), cudaMemcpyDeviceToHost)); //sizeof(int)

    auto t2 = cv::getTickCount();

    std::string cost_time = cv::format("class inference time: %.2f ms", (t2 - t1) / cv::getTickFrequency() * 1000);
    std::cout << cost_time << std::endl;

    // 后处理
    int class_id;
    cv::Mat tmp = pred.t();
    class_id = postprocess(img, tmp);

    cudaFree(bindings[0]);
    cudaFree(bindings[1]);

    return class_id;
}

int Class::postprocess(cv::Mat& frame, const cv::Mat tmp)
{
    using namespace cv;
    using namespace cv::dnn;

    std::vector<int> class_ids;

    float* data = (float*)tmp.data;

    for(int i = 0; i < tmp.rows; ++i) {
        float* classes_scores = data;

        cv::Mat scores(1, classesSize, CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;

        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if(max_class_score > scoreThreshold) {
            class_ids.push_back(class_id.x);
        }

        data += tmp.cols;
    }
    return class_ids[0];
}
