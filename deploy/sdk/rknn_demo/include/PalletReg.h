#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>

class PalletReg
{
public:
    PalletReg();
    ~PalletReg();
    const int pallet_reg_width = 112;
    const int pallet_reg_height = 112;
    const int pallet_reg_channel = 3;

    rknn_context pallet_reg_ctx;
    const char *pallet_reg_model_path = "model/pallet_reg.rknn";
    unsigned char *pallet_reg_model;
    rknn_input pallet_reg_inputs[1];
    int ret;

    int ini();
    int forward(const cv::Mat& img, std::vector<float> ftr);
};
