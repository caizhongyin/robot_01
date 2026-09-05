#include "common.h"
#include "YoloDet.h"

int YoloDet::init(){
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    std::ifstream ifs(yolo_det_path, std::ifstream::binary);
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
    //CHECK(cudaMalloc(&bindings[0], sizeof(float) * 1 * 3 * 640 * 640)); // type: float32[1,3,640,640]
    //CHECK(cudaMalloc(&bindings[1], sizeof(int) * 1 * 84 * 8400));   // type: float32[1,84,8400]
    return 0;
}

int YoloDet::inference(cv::Mat& img, std::vector<BoxInfo>& objects)
{
    std::vector<void*> bindings(mEngine->getNbIOTensors());
    //bindings = bindings(mEngine->getNbIOTensors());
    CHECK(cudaMalloc(&bindings[0], sizeof(float) * 1 * 3 * inpHeight * inpWidth)); // type: float32[1,3,640,640]
    CHECK(cudaMalloc(&bindings[1], sizeof(int) * 1 * (classesSize+4) * 8400));   // type: float32[1,84,8400]
    // 输入
    cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255., cv::Size(inpWidth,inpHeight), {0,0,0}, true, false);
    //blob = blob * 2 - 1;
    cv::Mat pred(cv::Size(8400, classesSize+4), CV_32F, {255,255,255});
    // 推理
    auto t1 = cv::getTickCount();

    CHECK(cudaMemcpy(bindings[0], static_cast<const void*>(blob.data), 1 * 3 * inpHeight * inpWidth* sizeof(float), cudaMemcpyHostToDevice));
    context->executeV2(bindings.data());
    CHECK(cudaMemcpy(static_cast<void*>(pred.data), bindings[1], 1 * (classesSize+4) * 8400 * sizeof(float), cudaMemcpyDeviceToHost)); //sizeof(int)

    auto t2 = cv::getTickCount();

    std::string cost_time = cv::format("yolo_det inference time: %.2f ms", (t2 - t1) / cv::getTickFrequency() * 1000);
    std::cout << cost_time << std::endl;

    // 后处理
    cv::Mat tmp = pred.t();
    postprocess(img, tmp, objects);
    //cv::imwrite("result.jpg",img);

    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
    return 0;
}

void YoloDet::postprocess(cv::Mat& frame, const cv::Mat tmp, std::vector<BoxInfo>& objects)
{
    using namespace cv;
    using namespace cv::dnn;
    // yolov8-12 has an output of shape (batchSize, class_num+4, 8400) (box[x,y,w,h] + confidence[c])

    auto tt1 = cv::getTickCount();

    auto inputSz = frame.size();

    float x_factor = inputSz.width / inpWidth;
    float y_factor = inputSz.height / inpHeight;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* data = (float*)tmp.data;

    for(int i = 0; i < tmp.rows; ++i) {
        //float confidence = data[4];
        //std::cout<<"confidence:"<<confidence<<std::endl;
        //if(confidence >= confThreshold) {
        float* classes_scores = data + 4;

        cv::Mat scores(1, classesSize, CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;

        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if(max_class_score > scoreThreshold) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        //}

        data += tmp.cols;
    }

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

    auto tt2 = cv::getTickCount();
    std::string label = format("postprocess time: %.2f ms", (tt2 - tt1) / cv::getTickFrequency() * 1000);
    cv::putText(frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    for(size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        //Rect box = boxes[idx];
        BoxInfo box;
        box.x1 = boxes[idx].x;
        box.y1 = boxes[idx].y;
        box.x2 = boxes[idx].x + boxes[idx].width;
        box.y2 = boxes[idx].y + boxes[idx].height;
        box.score = confidences[idx];
        box.label = class_ids[idx];
        objects.push_back(box);
        //drawPred(class_ids[idx], confidences[idx], box.x, box.y,
        //         box.x + box.width, box.y + box.height, frame);
    }
}

void YoloDet::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    using namespace cv;

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%d: %.2f", classId, conf);
    Scalar color(rand(), rand(), rand());

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), color, FILLED);
    cv::putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}
