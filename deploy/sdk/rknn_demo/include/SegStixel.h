#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>
#include <math.h>

class SegStixel
{
public:
    SegStixel();
    ~SegStixel();
    const int seg_stixel_width = 640;
    const int seg_stixel_height = 288;
    const int seg_stixel_channel = 3;

    rknn_context seg_stixel_ctx;
    const char *seg_stixel_model_path = "model/seg_stixel.rknn";
    unsigned char *seg_stixel_model;
    rknn_input seg_stixel_inputs[1];
    int ret;

    int ini();
    int forward(const cv::Mat& img, cv::Mat& segmented_img);
};
