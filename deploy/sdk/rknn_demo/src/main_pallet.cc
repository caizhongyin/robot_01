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
#include "PalletDet.h"
#include "PalletQlt.h"
#include "PalletOcc.h"
#include "PalletReg.h"

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
std::vector<int> get_square_bdbox(int min_x, int min_y, int max_x, int max_y, float scale, bool is_square)
{
    //float scale = 1.2;
    //bool is_square = True;
    int width = max_x - min_x;
    int height = max_y - min_y;

    int bd_length,bd_min_x,bd_min_y;
    int new_width, new_height;
    if (is_square)
    {
        bd_length = std::max(width, height) * scale;
        bd_min_x = int((min_x + max_x) / 2 - bd_length / 2);
        bd_min_y = int((min_y + max_y) / 2 - bd_length / 2);
        new_width = new_height = int(bd_length);

    }else
    {
        bd_min_x = int(min_x - (scale - 1)*width/2);
        bd_min_y = int(min_y- (scale - 1)*height/2);
        new_width = int(width * scale);
        new_height = int(height * scale); 
    }

    std::vector<int> vepallet_det_ctxy{bd_min_x, bd_min_y, new_width, new_height};
    return vepallet_det_ctxy;
}

std::vector<int> get_bdbox(int min_x, int min_y, int max_x, int max_y, int w, int h, float scale)
{
    //float scale = 1.0;
    int width = max_x - min_x;
    int height = max_y - min_y;

    int bd_length,bd_min_x,bd_min_y;
    int new_width, new_height;
    bd_min_x = int(min_x - (scale - 1)*width/2);
    bd_min_y = int(min_y- (scale - 1)*height/2);
    new_width = int(width * scale);
    new_height = int(height * scale);

    bd_min_x = (bd_min_x > 0) ? bd_min_x : 0;
    bd_min_y = (bd_min_y > 0) ? bd_min_y : 0;
    new_width = (bd_min_x+new_width < w) ? new_width : w-bd_min_x;
    new_height = (bd_min_y+new_height < h) ? new_height : h-bd_min_y;

    std::vector<int> vepallet_det_ctxy{bd_min_x, bd_min_y, new_width, new_height};
    return vepallet_det_ctxy;
}

int to_int(std::string str)
{
  std::istringstream is(str); //构造输入字符串流，流的内容初始化为“12”的字符串
  int i;
  is >> i; //从is流中读入一个int整数存入i中
  return i;
}

int to_float(std::string str)
{
  std::istringstream is(str); //构造输入字符串流，流的内容初始化为“12”的字符串
  float i;
  is >> i; //从is流中读入一个int整数存入i中
  return i;
}

//字符串分割函数
std::vector<std::string> split(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

// get current local time stamp
static int64_t getCurrentLocalTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
    // return std::chrono::duration_cast(std::chrono::system_clock::now().time_since_epoch()).count();
}

const char* class_names[] = {"pallet","food","soup","container","hand","receipt","receipt-holder","receipt-holder-empty"};
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    PalletDet palletDet;
    PalletQlt palletQlt;
    PalletOcc palletOcc;
    PalletReg palletReg;
    bool show_pallet_box = false;
    const char *in_path = argv[1];
    //const char *out_path = argv[2];
    // Load image
    std::ifstream inFile(in_path);
    //std::ofstream  outfile(out_path);
    std::string str;
    int64 t0,t1,t2,t3,t4;
    std::vector<int> default_pallet_box = {30, 110, 370, 460};
    if(inFile.is_open())
    {
        while (getline(inFile, str))
        {
            //std::cout<<str<<std::endl;
            cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
            if (!img.data) {
                cout<<"cv::imread fail:"<<str<<endl;
                continue;
            }
            
            t0 = getCurrentLocalTimeStamp();
            std::vector<BoxInfo> objects;
            palletDet.forward(img, objects);

            t1 = getCurrentLocalTimeStamp();
            //outfile<<str;
            std::vector<int> pallet_box = {9999,9999,0,0};
            std::vector<BoxInfo> reg_box;
            for(int j=0;j<objects.size();j++)
            {
                if(objects[j].label == 0){
                    if(objects[j].x1 < pallet_box[0]){
                        pallet_box[0] = objects[j].x1;
                    }
                    if(objects[j].y1 < pallet_box[1]){
                        pallet_box[1] = objects[j].y1;
                    }
                    if(objects[j].x2 > pallet_box[2]){
                        pallet_box[2] = objects[j].x2;
                    }
                    if(objects[j].y2 > pallet_box[3]){
                        pallet_box[3] = objects[j].y2;
                    }                    
                }
                if(objects[j].label == 1 or objects[j].label == 2){
                    reg_box.push_back(objects[j]);
                }
                if(show_pallet_box){
                    //outfile<<" "<<objects[j].x1<<","<<objects[j].y1<<","<< objects[j].x2 <<"," <<objects[j].y2 << ","<< objects[j].score <<"," << objects[j].label;
                    std::cout<<objects[j].x1<<" "<<objects[j].y1<<" "<< objects[j].x2 <<" " <<objects[j].y2 << " "<< objects[j].score <<" " << objects[j].label<<std::endl;
                    cv::Rect rect(cv::Point(objects[j].x1,objects[j].y1),cv::Point(objects[j].x2,objects[j].y2));
                    rectangle(img,rect,cv::Scalar(255,0,0),1);
                    putText(img, std::string(class_names[objects[j].label])+" "+  std::to_string(objects[j].score), cv::Point(objects[j].x1,objects[j].y1-10),1,1,cv::Scalar(255,0,0));
                    imwrite(str+"_bbox.jpg",img);
                }
            }
            //outfile<<std::endl;
            if(pallet_box[0] == 9999){
                pallet_box = default_pallet_box;
            }
            cv::Rect rect(pallet_box[0],pallet_box[1],pallet_box[2]-pallet_box[0],pallet_box[3]-pallet_box[1]);
            cv::Mat roi = img(rect);
            int qlt_res = palletQlt.forward(roi);

            t2 = getCurrentLocalTimeStamp();
            int occ_res = palletOcc.forward(roi);

            t3 = getCurrentLocalTimeStamp();
            if(occ_res == 1){
                for(int j=0;j<reg_box.size();j++)
                {
                    cv::Rect rect(reg_box[j].x1, reg_box[j].y1, reg_box[j].x2-reg_box[j].x1, reg_box[j].y2-reg_box[j].y1);
                    cv::Mat roi = img(rect);
                    std::vector<float> ftr(512);
                    palletReg.forward(roi, ftr);
                    //std::cout<<"dish_ftr: "<<ftr<<std::endl;
                }
            }
            t4 = getCurrentLocalTimeStamp();
            std::cout<<"pallet_times: det "<<t1-t0<<" qlt "<< t2-t1<<" occ "<<t3-t2<<" reg "<<t4-t3<<std::endl;
            std::cout<<"pallet_res: "<<str<<" qlt "<<qlt_res<<" occ "<<occ_res<<" reg "<<reg_box.size()<<std::endl;
       }
    }
    return 0;
}
