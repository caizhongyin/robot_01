#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>

class PalletOcc
{
public:
    PalletOcc();
    ~PalletOcc();
    const int pallet_occ_width = 112;
    const int pallet_occ_height = 112;
    const int pallet_occ_channel = 3;

    rknn_context pallet_occ_ctx;
    const char *pallet_occ_model_path = "model/pallet_occ.rknn";
    unsigned char *pallet_occ_model;
    rknn_input pallet_occ_inputs[1];
    int ret;

    int ini();
    int forward(const cv::Mat& img);
};
