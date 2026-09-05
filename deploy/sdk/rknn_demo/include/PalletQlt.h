#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>

class PalletQlt
{
public:
    PalletQlt();
    ~PalletQlt();
    const int pallet_qlt_width = 112;
    const int pallet_qlt_height = 112;
    const int pallet_qlt_channel = 3;

    rknn_context pallet_qlt_ctx;
    const char *pallet_qlt_model_path = "model/pallet_qlt.rknn";
    unsigned char *pallet_qlt_model;
    rknn_input pallet_qlt_inputs[1];
    int ret;

    int ini();
    int forward(const cv::Mat& img);
};
