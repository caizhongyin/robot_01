/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <chrono>
#include "SegStixel.h"

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

// get current local time stamp
static int64_t getCurrentLocalTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
    // return std::chrono::duration_cast(std::chrono::system_clock::now().time_since_epoch()).count();
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    SegStixel segStixel;
    bool show_seg = true;
    const char *in_path = argv[1];
    //const char *out_path = argv[2];
    // Load image
    std::ifstream inFile(in_path);
    //std::ofstream  outfile(out_path);
    std::string str;
    int64 t0,t1,t2,t3,t4;
    cv::Rect rect(0,352,640,288);
    if(inFile.is_open())
    {
        while (getline(inFile, str))
        {
            //std::cout<<str<<std::endl;
            cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
            cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
            if (!img.data) {
                cout<<"cv::imread fail:"<<str<<endl;
                continue;
            }
            t0 = getCurrentLocalTimeStamp();
            cv::Mat img2;
            cv::hconcat(img,img,img2);
            cv::resize(img2, img2, cv::Size(640, 640), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
            
            t1 = getCurrentLocalTimeStamp();
            cv::Mat segmented_img;
            cv::Mat roi = img2(rect);
            segStixel.forward(roi, segmented_img);

            t2 = getCurrentLocalTimeStamp();

            if(show_seg){
              std::string str_out = str+"_stixel.jpg";
              cv::cvtColor(segmented_img, segmented_img, cv::COLOR_RGB2BGR);
              bool saved = cv::imwrite(str_out, segmented_img);
            }

            std::cout<<"seg_times: preproccess "<<t1-t0<<" seg_stixel "<<t2-t1<<std::endl;
       }
    }
    return 0;
}
