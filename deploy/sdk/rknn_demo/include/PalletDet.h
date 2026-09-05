#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>


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

class PalletDet
{
public:
    PalletDet();
    ~PalletDet();
    const int PALLET_DET_WIDTH = 224;
    const int PALLET_DET_HEIGHT = 224;
    const int PALLET_DET_CHANNELS = 3;
    int num_class = 8;
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
    float score_threshold = 0.35;
    float nms_threshold = 0.4;

    rknn_context pallet_det_ctx;
    const char *pallet_det_model_path = "model/pallet_det.rknn";
    unsigned char *pallet_det_model;
    rknn_input pallet_det_inputs[1];
    int ret;

    int ini();
    int forward(const cv::Mat& img, std::vector<BoxInfo>& objects);
};
