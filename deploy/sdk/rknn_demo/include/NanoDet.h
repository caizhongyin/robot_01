#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>

#define NANO_DET_CLASS_NUM 8
#define NANO_DET_NET_INPUT_SIZE 224
#define NANO_DET_SCORE_THESH_HOLD 0.35
#define NANO_DET_NMS_THESH_HOLD 0.4

struct BoundingBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    int label;
    float score;
};

typedef struct {
    cv::Rect2f box;
    float confidence;
    int index;
}BBOX;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;


class NanoDet
{
public:

    int input_size[2] = {NANO_DET_NET_INPUT_SIZE, NANO_DET_NET_INPUT_SIZE}; // input height and width
    int num_class = NANO_DET_CLASS_NUM; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
public:
    NanoDet();
    ~NanoDet();
    rknn_context m_ctx;
    unsigned char *m_model;
public:
    int ini();
    void printRKNNTensor(rknn_tensor_attr *attr);
    unsigned char *load_model(const char *filename, int *model_size);
    int nano_det(const cv::Mat& bgr, std::vector<BoxInfo>& objects, float score_threshold, float nms_threshold);


    void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    void decode_infer(float* feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);

    cv::Mat rot_mat;
    cv::Mat m_reverMat;

    rknn_input_output_num m_io_num;
    rknn_input inputs[1];
};
