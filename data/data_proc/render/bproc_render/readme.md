# bproc render test
## start
```
# 1. find 生成obj列表
bash gen_obj_list.sh                                      

# 2. 渲染
## 单目
blenderproc run render_multi_obj.py --num-images 5 --min-objects 1 --max-objects 4 --gpu_id 1 --no-stereo
## 双目
blenderproc run render_multi_obj.py --num-images 5 --min-objects 1 --max-objects 4 --baseline 0.05 --output-dir output_stereo --gpu_id 1

# 3. 解析
## 单目
python extract_h5.py output output_extracted  
## 双目
python extract_h5.py output_stereo output_extracted_stereo

# 4. 生成yolo格式训练集
## 单目
python make_yolo_dataset.py \
      --input output_extracted \
      --det-out yolo_det --seg-out yolo_seg \
      --val-ratio 0.1 --poly-epsilon 1.0

## 单双目
python make_yolo_dataset.py \
    --input output_extracted output_extracted_stereo \
    --det-out yolo_det --seg-out yolo_seg \
    --val-ratio 0.1 --poly-epsilon 1.0

## 可视化
python yolo_viz.py --data-dir yolo_det yolo_seg
```
