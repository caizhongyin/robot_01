// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

#define MAX_TOP_NUM 20
  if (topNum > MAX_TOP_NUM)
    return 0;

  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j)) {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
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

// std::vector<int> argmax_indices; // 存储 argmax 结果
void softmax(float* data, int channels, int height, int width) {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int max_index = 0;
            // 计算最大值以增加稳定性
            float max_val = data[h * w];
            for (int c = 1; c < channels; ++c) {
              // if(max_val<data[c*width*height+h*width+w]){
              //   max_val = data[c*width*height+h*width+w];
              //   max_index = c;
              // }
                max_val = std::max(max_val, data[c*width*height+h*width+w]);
            }
            // argmax_indices.push_back(max_index);

            // 计算指数并计算 softmax 时的总和
            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                data[c*width*height+h*width+w] -= max_val;
                data[c*width*height+h*width+w] = std::exp(data[c*width*height+h*width+w]);
                sum += data[c*width*height+h*width+w];
            }
            
            // 使用总和进行归一化
            for (int c = 0; c < channels; ++c) {
                data[c*width*height+h*width+w] /= sum;
                // printf("%d,          %f\n",c*width*height+h*width+w,data[c*width*height+h*width+w]);
            }
        }
    }
}

int colors[22][3] = {{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0}, {0, 0, 128}, {128, 0, 128}, {0, 128, 128},
                   {128, 128, 128}, {64, 0, 0}, {192, 0, 0}, {64, 128, 0}, {192, 128, 0}, {64, 0, 128}, {192, 0, 128},
                   {64, 128, 128}, {192, 128, 128}, {0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0}, {0, 64, 128}, {128, 64, 12}};
                   
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    // const int MODEL_IN_WIDTH = 640;
    // const int MODEL_IN_HEIGHT = 288;
    // const int MODEL_IN_WIDTH = 320;
    const int MODEL_IN_WIDTH = 480;
    const int MODEL_IN_HEIGHT = 192;
    const int MODEL_IN_CHANNELS = 3;
    bool show_seg = true;

    /*core_mask: 0: auto, 1: npu core1, 2: npu core2, 4:npu core3,

​                     3: npu core1&2,

​                     7: npu core1&2&3

    仅RK3588支持 core mask。*/
    uint32_t core_mask = 1;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *in_path = argv[2];
    if (argc > 3) {
      core_mask = strtoul(argv[3], NULL, 10);
    }

    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret   = rknn_init(&ctx, model, model_len, 0, NULL);
    if (ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = MODEL_IN_WIDTH * MODEL_IN_HEIGHT * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    // Load image
    std::ifstream inFile(in_path);
    std::string str;
    int nAll=0;
    int nPos=0;
    int pred=0;
    float tm;
    int64 t1;
    if(inFile.is_open())
    {
        while (getline(inFile, str))
        {
            std::cout<<str<<std::endl;
            cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
            int w0 = img.cols;
            int h0 = img.rows;
            if (!img.data) {
                cout<<"cv::imread fail:"<<str<<endl;
                continue;
            }
            cv::Mat rsz_img;
            cv::Mat detMat;
 
            cv::resize(img, rsz_img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(rsz_img, detMat, COLOR_BGR2RGB);


            t1=getCurrentLocalTimeStamp();
            // Get Model Input Output Info
            inputs[0].buf = detMat.data;
            rknn_inputs_set(ctx, 1, inputs);

            ret = rknn_run(ctx, NULL);
            if (ret < 0){
                printf("rknn_run fail! ret=%d", ret);
            }

            //rknn_output outputs[1];
            //memset(outputs, 0, sizeof(outputs));
            //outputs[0].want_float = 1;
            ret = rknn_outputs_get(ctx, 1, outputs, NULL);
            if (ret < 0){
                printf("rknn_outputs_get fail! ret=%d\n", ret);
            }

            std::cout<<"segstixel inference time:"<<getCurrentLocalTimeStamp()-t1<<std::endl;

            // Get perf detail
            rknn_perf_detail perf_detail;
            ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
            if (ret != RKNN_SUCC) {
              printf("rknn_query fail! ret=%d\n", ret);
              return -1;
            }
            printf("rknn run perf detail is:\n%s", perf_detail.perf_data);

            // Get run duration time
            rknn_perf_run perf_run;
            ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
            if (ret != RKNN_SUCC) {
              printf("rknn_query fail! ret=%d\n", ret);
              return -1;
            }
            printf("rknn run perf time is %ldus\n", perf_run.run_duration);

            
            int output_size = outputs[0].size;
            //cout<<"output_size:"<<output_size<<endl;
            //float output_data = *(float*)outputs[0].buf;
            float* output_data = reinterpret_cast<float*>(outputs[0].buf);
            std::vector<float> interpreted_data(output_size);
            for (int i = 0; i < output_size/4; ++i) {
                interpreted_data[i] = output_data[i];
            }
            
            //===================post proccess
            cv::Mat segmented_img = rsz_img.clone(); //cv::Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
            for (int h = 0; h < MODEL_IN_HEIGHT; ++h) {
               for (int w = 0; w < MODEL_IN_WIDTH; ++w) {
                   if(interpreted_data[h*MODEL_IN_WIDTH+w] > 0.5){
                     segmented_img.at<cv::Vec3b>(h, w) = cv::Vec3b(128, 64, 128);
                   }
               }
            }
            
            if(show_seg){
              // cv::resize(segmented_img, segmented_img, cv::Size(w0, h0), (0, 0), (0, 0), cv::INTER_LINEAR);
              std::string str_out = str+"_seg.png";
              cout<<str_out<<endl;
              bool saved = cv::imwrite(str_out, segmented_img);
            }
            img.release();
            detMat.release();
            // Release rknn_outputs
            rknn_outputs_release(ctx, 1, outputs);
       }
    }
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
