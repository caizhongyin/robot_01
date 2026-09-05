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

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        //std::cout<<feat_h<<" "<<feat_w<<std::endl;
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    int reg_max = 7;
    int input_size[2] = {224, 224}; // input height and width

    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);

    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)input_size[0]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)input_size[1]);
    //std::cout << xmin << "," << ymin << "," << xmax << "," << ymax << ","<<score<<" " << std::endl;

    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void decode_infer(float* feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results, int num_class)
{
    const int num_points = center_priors.size();

    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        int index=idx *  (num_class+32);
        float* scores = feats+index;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++)
        {

            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }

        }

        if (score > threshold)
        {
            const float* bbox_pred = feats+index + num_class;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, (score), ct_x, ct_y, stride));
        }
    }
}

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int MODEL_IN_WIDTH = 224;
    const int MODEL_IN_HEIGHT = 224;
    const int MODEL_IN_CHANNELS = 3;
    int num_class = 8;
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
    float score_threshold = 0.35;
    float nms_threshold = 0.4;
    const char* class_names[] = {"pallet","food","soup","container","hand","receipt","receipt-holder","receipt-holder-empty"};

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
    int64 t1;
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
            int org_width = img.cols;
            int org_height = img.rows;

            t1=getCurrentLocalTimeStamp();
            cv::Mat detimg = img.clone();
            cv::resize(detimg, detimg, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
 
            // cv::cvtColor(img, img, COLOR_BGR2RGB);
            // Get Model Input Output Info
            inputs[0].buf = detimg.data;

            ret = rknn_inputs_set(ctx, 1, inputs);
            if(ret < 0) {
                printf("nanodet rknn_input_set fail! ret=%d\n", ret);
                return -1;
            }

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
            
            std::cout<<"nanodet reference time:"<<getCurrentLocalTimeStamp()-t1<<std::endl;
            
            std::vector<std::vector<BoxInfo>> results;
            results.resize(num_class);

            std::vector<CenterPrior> center_priors;
            generate_grid_center_priors(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, strides, center_priors);

            decode_infer((float*)outputs[0].buf, center_priors, score_threshold, results, num_class);

            std::vector<BoxInfo> dets;

            for (int i = 0; i < (int)results.size(); i++)
            {
                nms(results[i], nms_threshold);

                for (auto box : results[i])
                {
                    dets.push_back(box);
                }
            }

            rknn_outputs_release(ctx, 1, outputs);

            std::vector<BoxInfo> objects;
            for(int i=0;i<dets.size();i++)
            {
                float width_ratio = (float)org_width / (float)MODEL_IN_WIDTH;
                float height_ratio = (float)org_height / (float)MODEL_IN_HEIGHT;
                dets[i].x1=(dets[i].x1) * width_ratio;
                dets[i].y1=(dets[i].y1) * height_ratio;
                dets[i].x2=(dets[i].x2) * width_ratio;
                dets[i].y2=(dets[i].y2) * height_ratio;
            }

            for(int i=0;i<dets.size();i++) {
                int iBorderN=0;
                if (int(dets[i].x1) == 0 )iBorderN++;
                if (int(dets[i].y1) == 0 )iBorderN++;
                if (int(dets[i].x2) == org_width)iBorderN++;
                if (int(dets[i].y2) == org_height)iBorderN++;
                if (iBorderN < 2)
                {
                    objects.push_back(dets[i]);
                }
            }
            std::cout<<"obstacle box num:"<<objects.size()<<std::endl;
            outfile<<str;

            for(int j=0;j<objects.size();j++)
            {
                outfile<<" "<<objects[j].x1<<","<<objects[j].y1<<","<< objects[j].x2 <<"," <<objects[j].y2 << ","<< objects[j].score <<"," << objects[j].label;
                //std::cout<<objects[j].x1<<" "<<objects[j].y1<<" "<< objects[j].x2 <<" " <<objects[j].y2 << " "<< objects[j].score <<" " << objects[j].label<<std::endl;
                cv::Rect rect(cv::Point(objects[j].x1,objects[j].y1),cv::Point(objects[j].x2,objects[j].y2));
                rectangle(img,rect,cv::Scalar(255,0,0),1);
                putText(img, std::string(class_names[objects[j].label])+" "+  std::to_string(objects[j].score), cv::Point(objects[j].x1,objects[j].y1-10),1,1,cv::Scalar(255,0,0));
                imwrite(str+"_bbox.jpg",img);
            }
            outfile<<std::endl;
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
