#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
//#include "pose_postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include <dirent.h>

#include <fstream>
#include <iostream>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float x;
    float y;
    float score;
} KeyPoint;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    std::vector<KeyPoint> keyPoints;
} DetectRect;

class DetPose
{
public:
    DetPose();
    ~DetPose();
    const int det_pose_width = 640;
    const int det_pose_height = 640;
    const int det_pose_channel = 3;

    rknn_context det_pose_ctx;
    const char *det_pose_model_path = "model/det_pose.rknn";
    unsigned char *det_pose_model;
    rknn_input det_pose_inputs[1];
    rknn_tensor_attr output_attrs[9];
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    int ret;

    int ini();
    int forward(const cv::Mat& img, cv::Mat& segmented_img);

public:
    int GenerateMeshgrid();
    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints);
    float sigmoid(float x);

private:
    std::vector<float> meshgrid;
    const int class_num = 1;
    int headNum = 3;
    int input_w = 640;
    int input_h = 640;
    int strides[3] = {8, 16, 32};
    int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    int keypoint_num = 17;

    float nmsThresh = 0.45;
    float objectThresh = 0.5;
};
