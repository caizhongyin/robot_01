//#include "common.h"
#include "YoloSeg.h"
using namespace cv;

int YoloSeg::init(){
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    std::ifstream ifs(yolo_seg_path, std::ifstream::binary);
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

int YoloSeg::inference(cv::Mat& img)
{
    float ratio_w = 1.0, ratio_h = 1.0;
    int newh = 640, neww = 640, padh = 0, padw = 0;
    std::vector<void*> bindings(mEngine->getNbIOTensors());
    //bindings = bindings(mEngine->getNbIOTensors());
    CHECK(cudaMalloc(&bindings[0], sizeof(float) * 1 * 3 * inpHeight * inpWidth)); // type: float32[1,3,640,640]
    CHECK(cudaMalloc(&bindings[1], sizeof(float) * 1 * OUTPUT_SIZE));   // type: float32[1,84,8400]
    CHECK(cudaMalloc(&bindings[2], sizeof(float) * 1 * OUTPUT_SIZE1));   // type: float32[1,84,8400]
    // 输入
    cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255., cv::Size(inpWidth,inpHeight), {0,0,0}, true, false);
    //blob = blob * 2 - 1;
    //cv::Mat pred(cv::Size(8400, classesSize+4), CV_32F, {255,255,255});
	static float prob[OUTPUT_SIZE];
	static float prob1[OUTPUT_SIZE1];
    // 推理
    auto t1 = cv::getTickCount();

    CHECK(cudaMemcpy(bindings[0], static_cast<const void*>(blob.data), 1 * 3 * inpHeight * inpWidth* sizeof(float), cudaMemcpyHostToDevice));
    context->executeV2(bindings.data());
    //CHECK(cudaMemcpy(static_cast<void*>(pred.data), bindings[1], 1 * (classesSize+4) * 8400 * sizeof(float), cudaMemcpyDeviceToHost)); //sizeof(int)
	CHECK(cudaMemcpy(prob, bindings[1], 1 * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(prob1, bindings[2], 1 * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost));

    auto t2 = cv::getTickCount();

    std::string cost_time = cv::format("yolo_seg inference time: %.2f ms", (t2 - t1) / cv::getTickFrequency() * 1000);
    std::cout << cost_time << std::endl;

    // 后处理
    //cv::Mat tmp = pred.t();
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<cv::Mat> picked_proposals;  //后续计算mask
 
	// 处理box
	int net_length = CLASSES + 4 + _segChannels;
	cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob);
 
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < Num_box; i++) {
		//输出是1*net_length*Num_box;所以每个box的属性是每隔Num_box取一个值，共net_length个值
		cv::Mat scores = out1(cv::Rect(i, 4, 1, CLASSES)).clone();
		cv::Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= CONF_THRESHOLD) {
			cv::Mat temp_proto = out1(cv::Rect(i, 4 + CLASSES, 1, _segChannels)).clone();
			picked_proposals.push_back(temp_proto.t());
			float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
			float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
			float w = out1.at<float>(2, i) * ratio_w;  //w
			float h = out1.at<float>(3, i) * ratio_h;  //h
			int left = MAX((x - 0.5 * w), 0);
			int top = MAX((y - 0.5 * h), 0);
			int width = (int)w;
			int height = (int)h;
			if (width <= 0 || height <= 0) { continue; }
 
			classIds.push_back(classIdPoint.y);
			confidences.push_back(max_class_socre);
			boxes.push_back(Rect(left, top, width, height));
		}
 
	}
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
	std::vector<cv::Mat> temp_mask_proposals;
	std::vector<OutputSeg> output;
    cv::Rect holeImgRect(0, 0, img.cols, img.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx]&holeImgRect;
		output.push_back(result);
		temp_mask_proposals.push_back(picked_proposals[idx]);
	}
 
	// 处理mask
	cv::Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		maskProposals.push_back(temp_mask_proposals[i]);
 
	cv::Mat protos = cv::Mat(_segChannels, _segWidth * _segHeight, CV_32F, prob1);
	cv::Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时
	cv::Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });//n*160*160
 
	std::vector<Mat> maskChannels;
	cv::split(masks, maskChannels);
	cv::Rect roi(int((float)padw / inpWidth * _segWidth), int((float)padh / inpHeight * _segHeight), int(_segWidth - padw / 2), int(_segHeight - padh / 2));
	for (int i = 0; i < output.size(); ++i) {
		Mat dest, mask;
		cv::exp(-maskChannels[i], dest);//sigmoid
		dest = 1.0 / (1.0 + dest);//160*160
		dest = dest(roi);
		cv::resize(dest, mask, cv::Size(img.cols, img.rows), INTER_NEAREST);
		//crop----截取box中的mask作为该box对应的mask
		cv::Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > MASK_THRESHOLD;
		output[i].boxMask = mask;
	}
	auto end = std::chrono::system_clock::now();
	std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
 
	DrawPred(img, output);
	//cv::imshow("output.jpg", img);
	//char c = cv::waitKey(0);
    cv::imwrite("output.jpg", img);

    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
    cudaFree(bindings[2]);
    return 0;
}

void YoloSeg::DrawPred(cv::Mat& img,std:: vector<OutputSeg> result) {
	//生成随机颜色
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < CLASSES; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
	cv::Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		cv::rectangle(img, result[i].box, color[result[i].id], 2, 8);
		
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		char label[100];
		sprintf(label, "%d:%.2f", result[i].id, result[i].confidence);
 
		//std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		cv::Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	
	cv::addWeighted(img, 0.5, mask, 0.8, 1, img); //将mask加在原图上面
}

int main(int argc, char** argv)
{
    YoloSeg yoloseg_model;
    yoloseg_model.init();
    for(int i = 0; i < 5; ++i){
        cv::Mat img = cv::imread(R"(../images/test_det.jpg)");
        cv::Mat rsz_img;
        cv::resize(img, rsz_img, cv::Size(640, 640));
        yoloseg_model.inference(rsz_img);
    }
}