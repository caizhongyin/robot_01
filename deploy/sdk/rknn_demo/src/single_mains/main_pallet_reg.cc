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

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    // Load image
    std::ifstream inFile(in_path);
    std::ofstream outfile(out_path);
    std::string str;
    int nAll=0;
    int nPos=0;
    int pred=0;
    float tm;
    int64 t0;
    int64 t1;
    int64 t2;
    int64 t3;
    int64 t_pre = 0;
    int64 t_forward = 0;
    int64 t_post = 0;
    int img_nums = 0;
    int reg_id = 0;
    std::vector<float> last_ftr;
    std::string last_name;
    std::vector<std::string> names;
    std::vector<std::string> labels;
    std::vector<std::vector<float>> ftrs;
    float score = 0;
    if(inFile.is_open())
    {
        while (getline(inFile, str))
        {
            //std::cout<<str<<std::endl;
            cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
            int w0 = img.cols;
            int h0 = img.rows;
            if (!img.data) {
                std::cout<<"cv::imread fail:"<<str<<endl;
                continue;
            }
            cv::Mat rsz_img;
            cv::Mat detMat;
            t0=getCurrentLocalTimeStamp();
 
            cv::resize(img, rsz_img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(rsz_img, detMat, COLOR_BGR2RGB);
            // Get Model Input Output Info
            inputs[0].buf = detMat.data;

            t1=getCurrentLocalTimeStamp();

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

            //std::cout<<"deeplabv3 inference time:"<<getCurrentLocalTimeStamp()-t1<<std::endl;
            t2 = getCurrentLocalTimeStamp();
            img_nums += 1;
            
            int output_size = outputs[0].size;
            //float output_data_mask = *(float*)outputs[0].buf;
            float* output_data = reinterpret_cast<float*>(outputs[0].buf);
            std::vector<float> interpreted_data(output_size/4);
            for (int i = 0; i < output_size/4; ++i) {
                interpreted_data[i] = output_data[i];
            }
            //===================post proccess
            names = split(str, "/");
            if(last_name.size() > 0){
              for (int i = 0; i < output_size/4; ++i) {
                  score += interpreted_data[i] * last_ftr[i];
              }
              if(last_name == names[1]){
                std::cout<<"pos_score:"<<score<<endl;
              }else{
                std::cout<<"neg_score:"<<score<<endl;
                reg_id += 1;
              }
            }
            score = 0;
            last_name = names[1];
            last_ftr = interpreted_data;
            t3 = getCurrentLocalTimeStamp();
            //std::cout<<"output_size:"<<output_size<<" "<<interpreted_data.size()<<endl;
            t_pre += (t1-t0);
            t_forward += (t2-t1);
            t_post += (t3-t2);
            img.release();
            detMat.release();
            if(img_nums % 100 == 0){
              std::cout << "forward nums:" << img_nums << endl;
            }
            outfile<<reg_id<<" "<<str<<" "<<names[1]<<" "<<names[1]+"_desc"<<" ";
            for(int j=0;j<interpreted_data.size();j++)
            {
              if(j==interpreted_data.size() - 1){
                outfile<<interpreted_data[j];
              }else{
                outfile<<interpreted_data[j]<<",";
              }
            }
            outfile<<std::endl;
            labels.push_back(names[1]);
            ftrs.push_back(interpreted_data);
       }
    }
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
    std::cout << "avg cost time(pre forward post): "<< t_pre/img_nums << " " << t_forward / img_nums << " " << t_post / img_nums << endl;
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    std::vector<float> pos_scores;
    std::vector<float> neg_scores;
    for(int i=0; i < labels.size(); i++){
        for(int j=i+1; j < labels.size(); j++){
          score = 0.0;
          for(int k=0; k < 512; k++){
            score += ftrs[i][k] * ftrs[j][k];
          }
          if(labels[i] == labels[j]){
            pos_scores.push_back(score);
          }else{
            neg_scores.push_back(score);
          }
        }
    }
    std::sort(neg_scores.begin(), neg_scores.end());
    float neg_num = neg_scores.size();
    int pos_num = pos_scores.size();
    int ok_num;
    float acc;
    std::vector<float> statistic_scores = {0.9, 0.95, 0.99, 0.999, 0.9999};
    std::cout<<"precession recall score"<<endl;
    for(int i=0; i < statistic_scores.size(); i++){
      score = neg_scores[int(neg_num * statistic_scores[i])];
      ok_num = 0;
      for(int j=0;j<pos_num;j++){
        if(pos_scores[j]>score){
          ok_num +=1;
        }
      }
      acc = float(ok_num) / float(pos_num);
      std::cout<<statistic_scores[i]<<" "<<acc<<" "<<score<<endl;
    }
    return 0;
}
