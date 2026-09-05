#include "PalletDet.h"
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

PalletDet::PalletDet()
{
    ini();
}
PalletDet::~PalletDet()
{
    // Release
    if(pallet_det_ctx >= 0) {
        rknn_destroy(pallet_det_ctx);
    }
    if(pallet_det_model) {
        free(pallet_det_model);
    }

}

int PalletDet::ini()
{
    // Load RKNN Model
    int pallet_det_model_len = 0;
    pallet_det_model = load_model(pallet_det_model_path, &pallet_det_model_len);
    ret   = rknn_init(&pallet_det_ctx, pallet_det_model, pallet_det_model_len, 0, NULL);
    if (ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    memset(pallet_det_inputs, 0, sizeof(pallet_det_inputs));
    pallet_det_inputs[0].index = 0;
    pallet_det_inputs[0].type = RKNN_TENSOR_UINT8;
    pallet_det_inputs[0].size = PALLET_DET_WIDTH * PALLET_DET_HEIGHT * 3;
    pallet_det_inputs[0].fmt = RKNN_TENSOR_NHWC;
    pallet_det_inputs[0].pass_through = 0;
    printf("pallet_rknn_init success\n");

    return 1;
}

int PalletDet::forward(const cv::Mat& img, std::vector<BoxInfo>& objects)
{
    int org_width = img.cols;
    int org_height = img.rows;
    
    cv::Mat detimg = img.clone();
    cv::resize(detimg, detimg, cv::Size(PALLET_DET_WIDTH, PALLET_DET_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
 
    // cv::cvtColor(img, img, COLOR_BGR2RGB);
    // Get Model Input Output Info
    pallet_det_inputs[0].buf = detimg.data;

    ret = rknn_inputs_set(pallet_det_ctx, 1, pallet_det_inputs);
    if(ret < 0) {
        printf("nanodet rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(pallet_det_ctx, NULL);
    if (ret < 0){
        printf("rknn_run fail! ret=%d", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(pallet_det_ctx, 1, outputs, NULL);
    if (ret < 0){
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    std::vector<std::vector<BoxInfo>> results;
    results.resize(num_class);

    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(PALLET_DET_HEIGHT, PALLET_DET_WIDTH, strides, center_priors);

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

    rknn_outputs_release(pallet_det_ctx, 1, outputs);

    for(int i=0;i<dets.size();i++)
    {
        float width_ratio = (float)org_width / (float)PALLET_DET_WIDTH;
        float height_ratio = (float)org_height / (float)PALLET_DET_HEIGHT;
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
    //std::cout<<"obstacle box num:"<<objects.size()<<std::endl;

    return 1;
}