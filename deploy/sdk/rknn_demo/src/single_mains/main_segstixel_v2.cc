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
  const int MODEL_IN_WIDTH = 640;
  const int MODEL_IN_HEIGHT = 288;
  const int MODEL_IN_CHANNELS = 3;
  bool show_seg = true;
    

  rknn_context ctx = 0;
  int ret;

  char *model_path = argv[1];
  char *in_path = argv[2];

  // Load RKNN Model
  ret   = rknn_init(&ctx, model_path, 0, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&input_attrs[i]);
  }

  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&output_attrs[i]);
  }

  // Get custom string
  rknn_custom_string custom_string;
  ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("custom string: %s\n", custom_string.string);

  unsigned char*     input_data   = NULL;
  rknn_tensor_type   input_type   = RKNN_TENSOR_UINT8;
  rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

  // Create input tensor memory
  rknn_tensor_mem* input_mems[1];
  // default input type is int8 (normalize and quantize need compute in outside)
  // if set uint8, will fuse normalize and quantize to npu
  input_attrs[0].type = input_type;
  // default fmt is NHWC, npu only support NHWC in zero copy mode
  input_attrs[0].fmt = input_layout;

  input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

  // Create output tensor memory
  rknn_tensor_mem* output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this require float32 to compute top5
    // allocate float32 output tensor
    int output_size = output_attrs[i].n_elems * sizeof(float);
    output_mems[i]  = rknn_create_mem(ctx, output_size);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this require float32 to compute top5
    output_attrs[i].type = RKNN_TENSOR_FLOAT32;
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    if (ret < 0) {
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }
    
  // Load image
  std::ifstream inFile(in_path);
  std::string str;

  // Load image
  int req_height  = 0;
  int req_width   = 0;
  int req_channel = 0;

  switch (input_attrs[0].fmt) {
  case RKNN_TENSOR_NHWC:
    req_height  = input_attrs[0].dims[1];
    req_width   = input_attrs[0].dims[2];
    req_channel = input_attrs[0].dims[3];
    break;
  case RKNN_TENSOR_NCHW:
    req_height  = input_attrs[0].dims[2];
    req_width   = input_attrs[0].dims[3];
    req_channel = input_attrs[0].dims[1];
    break;
  default:
    printf("meet unsupported layout\n");
    return -1;
  }
  int height  = 0;
  int width   = 0;
  int channel = 0;
  int64 t1;

  width      = input_attrs[0].dims[2];
  int stride = input_attrs[0].w_stride;
  std::cout << width << std::endl;
  std::cout << stride << std::endl;
  if(inFile.is_open())
  {
        while (getline(inFile, str))
        {
            std::cout<<str<<std::endl;
            cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
            
            if (!img.data) {
                cout<<"cv::imread fail:"<<str<<endl;
                continue;
            }
            cv::Mat rsz_img;
            cv::Mat detMat;
 
            cv::resize(img, rsz_img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(rsz_img, detMat, COLOR_BGR2RGB);
            // Get Model Input Output Info
            input_data = detMat.data;
            input_data = img.data;
            if (!input_data) {
              return -1;
            }

            

            // Copy input data to input tensor memory

            t1=getCurrentLocalTimeStamp();
            if (width == stride) {
              memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
            } else {
              int height  = input_attrs[0].dims[1];
              int channel = input_attrs[0].dims[3];
              // copy from src to dst with stride
              uint8_t* src_ptr = input_data;
              uint8_t* dst_ptr = (uint8_t*)input_mems[0]->virt_addr;
              // width-channel elements
              int src_wc_elems = width * channel;
              int dst_wc_elems = stride * channel;
              for (int h = 0; h < height; ++h) {
                memcpy(dst_ptr, src_ptr, src_wc_elems);
                src_ptr += src_wc_elems;
                dst_ptr += dst_wc_elems;
              }
            }
            
            
            ret = rknn_run(ctx, NULL);
            if (ret < 0){
                printf("rknn_run fail! ret=%d", ret);
            }

            std::cout<<" inference time:"<<getCurrentLocalTimeStamp()-t1<<std::endl;
            
            // int output_size = outputs[0].size;
            
            float* output_data = (float*)output_mems[0]->virt_addr;
            // std::vector<float> interpreted_data(output_size);
            // for (int i = 0; i < output_size/4; ++i) {
            //     interpreted_data[i] = output_data[i];
            // }
            
            //===================post proccess
            cv::Mat segmented_img = rsz_img.clone(); //cv::Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
            for (int h = 0; h < MODEL_IN_HEIGHT; ++h) {
               for (int w = 0; w < MODEL_IN_WIDTH; ++w) {
                   if(output_data[h*MODEL_IN_WIDTH+w] > 0.5){
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
            
       }
  }
  // Destroy rknn memory
  rknn_destroy_mem(ctx, input_mems[0]);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    rknn_destroy_mem(ctx, output_mems[i]);
  }

  // destroy
  rknn_destroy(ctx);
  return 0;
}
