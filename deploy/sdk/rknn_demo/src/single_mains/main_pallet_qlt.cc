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

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

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

    std::vector<int> vectxy{bd_min_x, bd_min_y, new_width, new_height};
    return vectxy;
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

    std::vector<int> vectxy{bd_min_x, bd_min_y, new_width, new_height};
    return vectxy;
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

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int MODEL_IN_WIDTH = 112;
    const int MODEL_IN_HEIGHT = 112;
    const int MODEL_IN_CHANNELS = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *in_path = argv[2];
    const char *out_path = argv[3];

    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret   = rknn_init(&ctx, model, model_len, 0, NULL);
    if (ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = MODEL_IN_WIDTH * MODEL_IN_HEIGHT * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // Load image
    std::ifstream inFile(in_path);
    std::ofstream  outfile(out_path);
    std::string str;
    int nAll=0;
    int nPos=0;
    int pred=0;
    float tm;
    if(inFile.is_open())
    {
        while (getline(inFile, str))
        {
            std::vector<std::string> result = split(str, ",");
            std::string strFile = result[0];
            int x1 = to_int(result[1]);
            int y1 = to_int(result[2]);
            int x2 = to_int(result[3]);
            int y2 = to_int(result[4]);
            int trueLabel = to_int(result[5]);
            //std::cout<<strFile<<std::endl;
 
            cv::Mat img = cv::imread(strFile, cv::IMREAD_COLOR);
            if (!img.data) {
                cout<<"cv::imread fail:"<<strFile<<endl;
                continue;
            }
 
            time_t start,end;
 
            start=clock();
            vector<int> bbox{x1, y1, x2, y2};
            //std::vector<int> box = get_square_bdbox(bbox[0], bbox[1], bbox[2], bbox[3], 1.2, True);
            std::vector<int> box = get_bdbox(bbox[0], bbox[1], bbox[2], bbox[3], img.cols, img.rows, 1.0);
 
            int bd_min_x = box[0];
            int bd_min_y = box[1];
            int bd_width = box[2];
            int bd_height = box[3];
 
            cv::Mat img_border;
 
            cv::copyMakeBorder(img, img_border, bd_height, bd_height, bd_width, bd_width, cv::BORDER_CONSTANT);
 
            cv::Mat detMat = img_border(cv::Rect(bd_width + bd_min_x, bd_height + bd_min_y, bd_width, bd_height));
 
            if (detMat.cols != MODEL_IN_WIDTH || detMat.rows != MODEL_IN_HEIGHT) {
                //printf("resize %d %d to %d %d\n", detMat.cols, detMat.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
                cv::resize(detMat, detMat, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            }
 
            cv::cvtColor(detMat, detMat, COLOR_BGR2RGB);
            // Get Model Input Output Info
            inputs[0].buf = detMat.data;

            rknn_inputs_set(ctx, 1, inputs);

            ret = rknn_run(ctx, NULL);
            if (ret < 0){
                printf("rknn_run fail! ret=%d", ret);
            }

            rknn_output outputs[1];
            memset(outputs, 0, sizeof(outputs));
            outputs[0].want_float = 1;
            ret = rknn_outputs_get(ctx, 1, outputs, NULL);
            if (ret < 0){
                printf("rknn_outputs_get fail! ret=%d\n", ret);
            }
            
            float res0 = *(float*)outputs[0].buf;
            float res1 = *((float*)outputs[0].buf+1);

            nAll++;
            if (res0 > res1){
              pred = 0;
            }else{
              pred = 1;
            }

            if(pred == trueLabel){
              nPos++;
            }

            tm = tm + clock() - start;
            
            if (nAll % 10 == 0){
                std::cout<<"tmp_acc:"<<float(nPos)/float(nAll)<<std::endl;
            }
            img.release();
            detMat.release();
            // Release rknn_outputs
            rknn_outputs_release(ctx, 1, outputs);
            outfile << strFile <<","<<result[1]<<","<<result[2]<<","<<result[3]<<","<<result[4]<<","<<trueLabel<<","<<pred<<","<<res0<<","<<res1<<"\n";
       }
    }

    float acc=nPos/float(nAll);
    float avg_tm = tm / float(nAll);
    std::cout<<"acc:"<<acc<<std::endl;
    std::cout<<"cost_time_avg:"<< avg_tm <<std::endl;
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
