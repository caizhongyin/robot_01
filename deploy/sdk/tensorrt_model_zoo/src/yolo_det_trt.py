import os
from datetime import datetime

import cv2
import numpy as np
import pickle
import tensorrt as trt
from cuda import cudart
import math


CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
OBB_NUM = 1

INPUT_W = 640
INPUT_H = 640

def cuda_call(call):
    """CUDA调用错误检查"""
    err, res = call[0], call[1:]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA call failed: {err}")
    if len(res) == 1:
        res = res[0]
    return res

def memcpy_host_to_device(device_ptr, host_arr):
    """从主机内存拷贝到设备内存"""
    nbytes = host_arr.nbytes
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr.ctypes.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

def memcpy_device_to_host(host_arr, device_ptr):
    """从设备内存拷贝到主机内存"""
    nbytes = host_arr.nbytes
    cuda_call(cudart.cudaMemcpy(host_arr.ctypes.data, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

def compute_iou(rec1, rec2):
    # computing area of each rectangles
    rec1 = np.array(rec1[:4],dtype='float')
    rec2 = np.array(rec2[:4],dtype='float')
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), swap=(2, 0, 1)):
    """图像letterbox预处理"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(swap)
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.
    return im, r, (dw, dh)

class Detection:
    def __init__(self, bbox, score, class_id, angle):
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.angle = angle
        
class TensorRTEngine:
    """TensorRT 推理引擎封装 - 使用 cudart"""
    def __init__(self, engine_path):
        self.engine_path = engine_path
        
        # 设置TensorRT日志
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        # 获取输入形状
        self.imgsz = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))[2:]
        self.context = self.engine.create_execution_context()
        
        # 设置I/O绑定
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            if is_input:
                self.batch_size = shape[0]
            
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def output_spec(self):
        """获取输出张量的规格"""
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, img):
        """执行推理"""
        # 准备输出数据
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # 处理I/O并执行网络
        memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(img))
        self.context.execute_v2(self.allocations)
        
        for o in range(len(outputs)):
            memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
        
        return outputs[0] if len(outputs) == 1 else outputs

class YoloObbTrt:
    def __init__(self, model_path):
        self.trt_engine = TensorRTEngine(model_path)
        self.name_map  = { 0: 'package', 1: 'barcode'}
    
    def predict(self, img, conf=None):
        """使用TensorRT进行推理"""
        if conf is None:
            conf = CONF_THRESH
            
        # 预处理
        processed_img, ratio, dwdh = letterbox(img, self.trt_engine.imgsz)
        
        # 推理
        data = self.trt_engine.infer(processed_img)

        print(f"Input image shape: {img.shape}") # (h, w, 3)
        print(f"TRT output shape: {data.shape}") # (1, 4+class_num, 8400)

        data = data.transpose(0, 2, 1)
        predictions = data[0]  # 去掉batch维度 (8400, 4+class_num)
        print(predictions.shape)
        #print("predictions[0]:",predictions[0])
        
        # 后处理
        dets = self.postprocess(predictions, ratio, dwdh)
        #dets = self.post_process(predictions)
        
        detections = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            
            for i in range(len(dets)):
                if final_scores[i] >= conf:
                    x1,y1,x2,y2 = final_boxes[i]
                    #cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                    cls_id = int(final_cls_inds[i])
                    score = final_scores[i]
                    
                    detections.append({
                        "cls": cls_id,
                        "bbox": np.array([x1,y1,x2,y2]),
                        "mask": None,
                        "det_conf": score,
                        "instance_id": i,
                        "name": self.name_map.get(cls_id, f"class_{cls_id}"),
                        "qlt_id": 0
                    })
        
        return detections
    
    def postprocess(self, predictions, ratio, dwdh):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # 转换坐标格式
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        
        # 去除padding并缩放到原图尺寸
        #print('ratio:',ratio,'dwdh:',dwdh)
        dwdh = np.asarray(dwdh * 2, dtype=np.float32)
        boxes_xyxy -= dwdh
        boxes_xyxy /= ratio
        
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.4, score_thr=0.25)
        
        return dets


if __name__ == '__main__':
    yolo_obb_trt = YoloObbTrt('../models/yolo11n.trt')
    img = cv2.imread('../test_det.jpg')
    dets = yolo_obb_trt.predict(img)
    for det in dets:
        x1,y1,x2,y2 = det["bbox"]
        color = (0,255,0)
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(img, str(det['cls']), (int(x1)-2, int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2) #0.5 1
        
    cv2.imwrite('test_det_box.jpg',img)
    