#include "PalletQlt.h"
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

PalletQlt::PalletQlt()
{
    ini();
}

PalletQlt::~PalletQlt()
{
    // Release
    if(pallet_qlt_ctx >= 0) {
        rknn_destroy(pallet_qlt_ctx);
    }
    if(pallet_qlt_model) {
        free(pallet_qlt_model);
    }

}

int PalletQlt::ini()
{
    // Load RKNN Model
    int pallet_qlt_model_len = 0;
    pallet_qlt_model = load_model(pallet_qlt_model_path, &pallet_qlt_model_len);
    ret   = rknn_init(&pallet_qlt_ctx, pallet_qlt_model, pallet_qlt_model_len, 0, NULL);
    if (ret < 0) {
      printf("pallet_qlt rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    memset(pallet_qlt_inputs, 0, sizeof(pallet_qlt_inputs));
    pallet_qlt_inputs[0].index = 0;
    pallet_qlt_inputs[0].type = RKNN_TENSOR_UINT8;
    pallet_qlt_inputs[0].size = pallet_qlt_width * pallet_qlt_height * 3;
    pallet_qlt_inputs[0].fmt = RKNN_TENSOR_NHWC;
    pallet_qlt_inputs[0].pass_through = 0;
    printf("pallet_qlt rknn_init success\n");

    return 1;
}

int PalletQlt::forward(const cv::Mat& img)
{   
    cv::Mat detimg = img.clone();
    cv::resize(detimg, detimg, cv::Size(pallet_qlt_width, pallet_qlt_height), (0, 0), (0, 0), cv::INTER_LINEAR);
 
    cv::cvtColor(detimg, detimg, cv::COLOR_BGR2RGB);
    // Get Model Input Output Info
    pallet_qlt_inputs[0].buf = detimg.data;

    ret = rknn_inputs_set(pallet_qlt_ctx, 1, pallet_qlt_inputs);
    if(ret < 0) {
        printf("pallet_qlt rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(pallet_qlt_ctx, NULL);
    if (ret < 0){
        printf("pallet_qlt rknn_run fail! ret=%d", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(pallet_qlt_ctx, 1, outputs, NULL);
    if (ret < 0){
        printf("pallet_qlt rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    float res0 = *(float*)outputs[0].buf;
    float res1 = *((float*)outputs[0].buf+1);

    int pred;
    if (res0 > res1){
      pred = 0;
    }else{
      pred = 1;
    }

    rknn_outputs_release(pallet_qlt_ctx, 1, outputs);
    detimg.release();

    return pred;
}