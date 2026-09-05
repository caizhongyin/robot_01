#include "DetPose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
//#include "pose_postprocess.h"

#include <algorithm>
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/****** yolov8 ****/

float DetPose::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

int DetPose::GenerateMeshgrid()
{
    ret = 0;
    if (headNum == 0)
    {
        printf("=== yolov8 Meshgrid  Generate failed! \n");
    }

    for (int index = 0; index < headNum; index++)
    {
        for (int i = 0; i < mapSize[index][0]; i++)
        {
            for (int j = 0; j < mapSize[index][1]; j++)
            {
                meshgrid.push_back(float(j + 0.5));
                meshgrid.push_back(float(i + 0.5));
            }
        }
    }

    // printf("=== yolov8 Meshgrid  Generate success! \n");

    return ret;
}

int DetPose::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints)
{
    ret = 0;
    if (meshgrid.empty())
    {
        ret = GenerateMeshgrid();
    }

    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_pose = 0;
    float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_pose = 0;
    KeyPoint Point;

    std::vector<DetectRect> detectRects;

    for (int index = 0; index < headNum; index++)
    {
        int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
        int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
        int8_t *pose = (int8_t *)pBlob[index + headNum * 2];

        quant_zp_reg = qnt_zp[index * 2 + 0];
        quant_zp_cls = qnt_zp[index * 2 + 1];
        quant_zp_pose = qnt_zp[index + headNum * 2];

        quant_scale_reg = qnt_scale[index * 2 + 0];
        quant_scale_cls = qnt_scale[index * 2 + 1];
        quant_scale_pose = qnt_scale[index + headNum * 2];

        for (int h = 0; h < mapSize[index][0]; h++)
        {
            for (int w = 0; w < mapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == class_num)
                {
                    cls_max = sigmoid(DeQnt2F32(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                    cls_index = 0;
                }
		else
		{
                    for (int cl = 0; cl < class_num; cl++)
                    {
			cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w];

			if (0 == cl)
			{
                            cls_max = cls_val;
                            cls_index = cl;
			}
			else
			{
                            if (cls_val > cls_max)
                            {
                            	cls_max = cls_val;
                            	cls_index = cl;
                            }
			}
                    }
                    cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
		}

                if (cls_max > objectThresh)
                {
                    xmin = (meshgrid[gridIndex + 0] - DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg)) * strides[index];
                    ymin = (meshgrid[gridIndex + 1] - DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg)) * strides[index];
                    xmax = (meshgrid[gridIndex + 0] + DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg)) * strides[index];
                    ymax = (meshgrid[gridIndex + 1] + DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg)) * strides[index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < input_w ? xmax : input_w;
                    ymax = ymax < input_h ? ymax : input_h;

                    if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                    {
                        DetectRect temp;
                        temp.xmin = xmin / input_w;
                        temp.ymin = ymin / input_h;
                        temp.xmax = xmax / input_w;
                        temp.ymax = ymax / input_h;
                        temp.classId = cls_index;
                        temp.score = cls_max;
                        

                        for(int kc = 0; kc < keypoint_num; kc ++)
                        {
                            Point.x = (DeQnt2F32(pose[(kc * 3 + 0) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose) * 2 + (meshgrid[gridIndex + 0] - 0.5)) * strides[index] / input_w;
                            Point.y = (DeQnt2F32(pose[(kc * 3 + 1) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose) * 2 + (meshgrid[gridIndex + 1] - 0.5)) * strides[index] / input_h;
                            Point.score = sigmoid(DeQnt2F32(pose[(kc * 3 + 2) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose));

                            temp.keyPoints.push_back(Point);
                        }

                        detectRects.push_back(temp);
                    }
                }
            }
        }
    }

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
              { return (Rect1.score > Rect2.score); });

    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1的格式存放在vector<float>中
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            // 每个检测框对应的17个关键点按照（score, x, y）格式存在vector<float>中
            for(int kn = 0; kn < keypoint_num; kn ++)
            {
                DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].score));
                DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].x));
                DetectKeyPoints.push_back(float(detectRects[i].keyPoints[kn].y));
            }


            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > nmsThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }

    return ret;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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

DetPose::DetPose()
{
    ini();
}

DetPose::~DetPose()
{
    // Release
    if(det_pose_ctx >= 0) {
        rknn_destroy(det_pose_ctx);
    }
    if(det_pose_model) {
        free(det_pose_model);
    }

}

int DetPose::ini()
{
    // Load RKNN Model
    int det_pose_model_len = 0;
    det_pose_model = load_model(det_pose_model_path, &det_pose_model_len);
    ret   = rknn_init(&det_pose_ctx, det_pose_model, det_pose_model_len, 0, NULL);
    if (ret < 0) {
      printf("det_pose rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    memset(det_pose_inputs, 0, sizeof(det_pose_inputs));
    det_pose_inputs[0].index = 0;
    det_pose_inputs[0].type = RKNN_TENSOR_UINT8;
    det_pose_inputs[0].size = det_pose_width * det_pose_height * 3;
    det_pose_inputs[0].fmt = RKNN_TENSOR_NHWC;
    det_pose_inputs[0].pass_through = 0;

    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < 9; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(det_pose_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        //dump_tensor_attr(&(output_attrs[i]));
    }
    for (int i = 0; i < 9; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    printf("det_pose rknn_init success\n");

    return 1;
}

int DetPose::forward(const cv::Mat& img, cv::Mat& pose_image)
{   
    pose_image = img.clone();
    //cv::resize(pose_image, pose_image, cv::Size(det_pose_width, det_pose_height), (0, 0), (0, 0), cv::INTER_LINEAR);
 
    //cv::cvtColor(rsz_img, detimg, cv::COLOR_BGR2RGB);
    // Get Model Input Output Info
    det_pose_inputs[0].buf = pose_image.data;

    ret = rknn_inputs_set(det_pose_ctx, 1, det_pose_inputs);
    if(ret < 0) {
        printf("det_pose rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(det_pose_ctx, NULL);
    if (ret < 0){
        printf("det_pose rknn_run fail! ret=%d", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(det_pose_ctx, 1, outputs, NULL);
    if (ret < 0){
        printf("det_pose rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    

    // 后处理部分
    int img_width = pose_image.cols;
    int img_height = pose_image.rows;
    std::cout<<"det_pose: postproccess img_width_height "<< img_width << " " << img_height <<std::endl;
    int8_t *pblob[9];
    for (int i = 0; i < 9; ++i)
    {
        pblob[i] = (int8_t *)outputs[i].buf;
    }

    // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
    //GetResultRectYolov8 PostProcess;
    std::vector<float> DetectiontRects;

    // 将17个关键点按照每个点（score, x, y）的顺序存入
    std::vector<float> DetectKeyPoints;
    /*
    ret = GetConvDetectionResult(pblob, out_zps, out_scales, DetectiontRects, DetectKeyPoints); //PostProcess.

    std::cout<<"det_pose: postproccess out_zps out_scales DetectKeyPoints"<< out_zps.size() << " " << out_scales.size() << " " << DetectKeyPoints.size() <<std::endl;

    int KeyPointsNum = 17;
    float pose_score = 0;
    int pose_x = 0, pose_y = 0;
    int NumIndex = 0, Temp = 0;
    
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(pose_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(pose_image, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    
        Temp = 0; 
        // 关键点
        for (int k = NumIndex * KeyPointsNum * 3; k < (NumIndex + 1)* KeyPointsNum * 3 ; k += 3)
        {
            pose_score = DetectKeyPoints[k + 0];
            
            if(pose_score > 0.5)
            {
                // pose_x = int(DetectKeyPoints[k + 1] * float(img_width) + 0.5);
                // pose_y = int(DetectKeyPoints[k + 2] * float(img_height) + 0.5);

                pose_x = int(DetectKeyPoints[k + 1] * float(img_width));
                pose_y = int(DetectKeyPoints[k + 2] * float(img_height));

                if(Temp < 5)
                {
                    cv::circle(pose_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(0, 0, 255), 5);
                }
                else if(5 <= Temp && Temp < 12)
                {
                    cv::circle(pose_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(0, 255, 0), 5);
                }
                else
                {
                    cv::circle(pose_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(255, 0, 0), 5);
                }

                // if(k -3  >= 0)
                // {
                //     int pose1_x = int(DetectKeyPoints[k - 3 + 1] * float(det_pose_width) + 0.5);
                //     int pose1_y = int(DetectKeyPoints[k -3  + 2] * float(det_pose_height) + 0.5);
                //     cv::line(pose_image, {pose_x, pose_y}, {pose1_x, pose1_y}, cv::Scalar(255, 0, 255), 2);
                // }
            }
            Temp += 1;
        }
        NumIndex += 1;
    }
    std::cout<<"det_pose: postproccess DetectiontRects.size() "<< DetectiontRects.size() <<std::endl;
    */

    return 1;
}