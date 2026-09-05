#include "SegStixel.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>

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

SegStixel::SegStixel()
{
    ini();
}

SegStixel::~SegStixel()
{
    // Release
    if(seg_stixel_ctx >= 0) {
        rknn_destroy(seg_stixel_ctx);
    }
    if(seg_stixel_model) {
        free(seg_stixel_model);
    }

}

int SegStixel::ini()
{
    // Load RKNN Model
    int seg_stixel_model_len = 0;
    seg_stixel_model = load_model(seg_stixel_model_path, &seg_stixel_model_len);
    ret   = rknn_init(&seg_stixel_ctx, seg_stixel_model, seg_stixel_model_len, 0, NULL);
    if (ret < 0) {
      printf("seg_stixel rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    memset(seg_stixel_inputs, 0, sizeof(seg_stixel_inputs));
    seg_stixel_inputs[0].index = 0;
    seg_stixel_inputs[0].type = RKNN_TENSOR_UINT8;
    seg_stixel_inputs[0].size = seg_stixel_width * seg_stixel_height * 3;
    seg_stixel_inputs[0].fmt = RKNN_TENSOR_NHWC;
    seg_stixel_inputs[0].pass_through = 0;
    printf("seg_stixel rknn_init success\n");

    return 1;
}

int SegStixel::forward(const cv::Mat& img, cv::Mat& segmented_img)
{   
    cv::Mat rsz_img = img.clone();
    //cv::resize(rsz_img, rsz_img, cv::Size(seg_stixel_width, seg_stixel_height), (0, 0), (0, 0), cv::INTER_LINEAR);
 
    //cv::cvtColor(rsz_img, detimg, cv::COLOR_BGR2RGB);
    // Get Model Input Output Info
    seg_stixel_inputs[0].buf = rsz_img.data;

    ret = rknn_inputs_set(seg_stixel_ctx, 1, seg_stixel_inputs);
    if(ret < 0) {
        printf("seg_stixel rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(seg_stixel_ctx, NULL);
    if (ret < 0){
        printf("seg_stixel rknn_run fail! ret=%d", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(seg_stixel_ctx, 1, outputs, NULL);
    if (ret < 0){
        printf("seg_stixel rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    int output_size = outputs[0].size;
    float* output_data = reinterpret_cast<float*>(outputs[0].buf);
    std::vector<float> interpreted_data(output_size);
    for (int i = 0; i < output_size/4; ++i) {
        interpreted_data[i] = output_data[i];
    }
    segmented_img = rsz_img.clone(); //cv::Mat(seg_stixel_height, seg_stixel_width, CV_8UC3);
    for (int h = 0; h < seg_stixel_height; ++h) {
       for (int w = 0; w < seg_stixel_width; ++w) {
           if(interpreted_data[h*seg_stixel_width+w] > 0.5){
             //segmented_img.at<cv::Vec3b>(h, w) = 0.5* cv::Vec3b(128, 64, 128) + 0.5 * segmented_img.at<cv::Vec3b>(h, w);
              segmented_img.at<cv::Vec3b>(h, w) = cv::Vec3b(128, 64, 128);
           }
       }
    }

    return 1;
}