#include "PalletReg.h"
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

PalletReg::PalletReg()
{
    ini();
}

PalletReg::~PalletReg()
{
    // Release
    if(pallet_reg_ctx >= 0) {
        rknn_destroy(pallet_reg_ctx);
    }
    if(pallet_reg_model) {
        free(pallet_reg_model);
    }

}

int PalletReg::ini()
{
    // Load RKNN Model
    int pallet_reg_model_len = 0;
    pallet_reg_model = load_model(pallet_reg_model_path, &pallet_reg_model_len);
    ret   = rknn_init(&pallet_reg_ctx, pallet_reg_model, pallet_reg_model_len, 0, NULL);
    if (ret < 0) {
      printf("pallet_reg rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    memset(pallet_reg_inputs, 0, sizeof(pallet_reg_inputs));
    pallet_reg_inputs[0].index = 0;
    pallet_reg_inputs[0].type = RKNN_TENSOR_UINT8;
    pallet_reg_inputs[0].size = pallet_reg_width * pallet_reg_height * 3;
    pallet_reg_inputs[0].fmt = RKNN_TENSOR_NHWC;
    pallet_reg_inputs[0].pass_through = 0;
    printf("pallet_reg rknn_init success\n");

    return 1;
}

int PalletReg::forward(const cv::Mat& img, std::vector<float> ftr)
{   
    cv::Mat detimg = img.clone();
    cv::resize(detimg, detimg, cv::Size(pallet_reg_width, pallet_reg_height), (0, 0), (0, 0), cv::INTER_LINEAR);
 
    cv::cvtColor(detimg, detimg, cv::COLOR_BGR2RGB);
    // Get Model Input Output Info
    pallet_reg_inputs[0].buf = detimg.data;

    ret = rknn_inputs_set(pallet_reg_ctx, 1, pallet_reg_inputs);
    if(ret < 0) {
        printf("pallet_reg rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(pallet_reg_ctx, NULL);
    if (ret < 0){
        printf("pallet_reg rknn_run fail! ret=%d", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(pallet_reg_ctx, 1, outputs, NULL);
    if (ret < 0){
        printf("pallet_reg rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    int output_size = outputs[0].size;
    //float output_data_mask = *(float*)outputs[0].buf;
    float* output_data = reinterpret_cast<float*>(outputs[0].buf);
    for (int i = 0; i < output_size/4; ++i) {
        ftr[i] = output_data[i];
    }

    rknn_outputs_release(pallet_reg_ctx, 1, outputs);
    detimg.release();

    return 1;
}