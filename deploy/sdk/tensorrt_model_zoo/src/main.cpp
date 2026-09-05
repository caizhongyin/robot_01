#include "common.h"
#include "YoloDet.h"
#include "Class.h"

int main(int argc, char** argv)
{
    YoloDet yolodet_model;
    yolodet_model.init();
    Class cls_model;
    cls_model.init();
    for(int i = 0; i < 5; ++i){
        cv::Mat img = cv::imread(R"(../images/test_det.jpg)");
        std::vector<BoxInfo> objects;
        auto t1 = cv::getTickCount();
        yolodet_model.inference(img, objects);
        auto t2 = cv::getTickCount();
        std::string cost_time = cv::format("yolo time: %.2f ms", (t2 - t1) / cv::getTickFrequency() * 1000);
        std::cout << cost_time << std::endl;
        for(int i = 0; i < objects.size(); ++i) {
            if(objects[i].score < 0.45){
                cv::Rect roi;
                roi.x = objects[i].x1;
                roi.y = objects[i].y1;
                roi.width = objects[i].x2 - objects[i].x1;
                roi.height = objects[i].y2 - objects[i].y1;
                cv::Mat crop_img = cv::Mat(img, roi);
                int class_id = cls_model.inference(img);
                if(class_id != objects[i].label){
                    std::cout<< "class_id:" << class_id << " objects[i].label:" << objects[i].label << std::endl;
                    //objects[i].label = class_id;
                }
                if(class_id == 5){
                    objects[i].label = class_id;
                }
            }
            yolodet_model.drawPred(objects[i].label, objects[i].score, objects[i].x1, objects[i].y1, objects[i].x2, objects[i].y2, img);
        }
        auto t3 = cv::getTickCount();
        std::string cost_time_cls = cv::format("class time: %.2f ms", (t3 - t2) / cv::getTickFrequency() * 1000);
        std::cout << cost_time_cls << std::endl;
        cv::imwrite("result.jpg",img);
    }
	return 1;
}