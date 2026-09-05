import os
from datetime import datetime

import cv2
import numpy as np
import pickle
import tensorrt as trt
from cuda import cudart
import math
import time

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

class YoloPoseTrt:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.4, kpt_thresh=0.3):
        self.trt_engine = TensorRTEngine(model_path)
        self.name_map  = { 0: 'person'}
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.kpt_thresh = kpt_thresh
        self.num_kpts = 17  # COCO人体关键点数量

        self.kpt_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def predict(self, img):
        """使用TensorRT进行推理"""

        # 预处理
        processed_img, ratio, dwdh = letterbox(img, self.trt_engine.imgsz)
        
        # 推理
        data = self.trt_engine.infer(processed_img)

        # print(f"Input image shape: {img.shape}") # (h, w, 3)
        # print(f"TRT output shape: {data.shape}") # (1, 56, 8400)

        data = data.transpose(0, 2, 1)
        predictions = data[0]  # 去掉batch维度 (8400, 56)
        #print(predictions.shape)
        
        # 后处理
        dets = self.postprocess(predictions, ratio, dwdh)
        
        detections = []
        if dets is not None:
            for i, det in enumerate(dets):
                if det[4] >= self.conf_thresh:  # 检查置信度
                    bbox = det[:4]  # x1, y1, x2, y2
                    conf = det[4]   # 置信度
                    cls_id = 0      # 人体类别
                    
                    # 提取关键点 (从第6个元素开始，每3个为一组: x, y, conf)
                    keypoints = []
                    for kpt_idx in range(self.num_kpts):
                        start_idx = 5 + kpt_idx * 3
                        if start_idx + 2 < len(det):
                            x = det[start_idx]
                            y = det[start_idx + 1] 
                            kpt_conf = det[start_idx + 2]
                            keypoints.append([x, y, kpt_conf])
                        else:
                            keypoints.append([0, 0, 0])  # 无效关键点
                    
                    detections.append({
                        "cls": cls_id,
                        "bbox": bbox,
                        "keypoints": np.array(keypoints),
                        "det_conf": conf,
                        "instance_id": i,
                        "name": self.name_map.get(cls_id, f"class_{cls_id}"),
                        "qlt_id": 0
                    })
        
        return detections
    
    def postprocess(self, predictions, ratio, dwdh):
        """YOLO-Pose后处理"""
        # 分离边界框、置信度和关键点
        boxes = predictions[:, :4]  # x, y, w, h (center format)
        scores = predictions[:, 4:5]  # 置信度
        kpts = predictions[:, 5:]  # 关键点坐标和置信度
        
        # 过滤低置信度预测
        valid_mask = scores[:, 0] > self.conf_thresh
        if not valid_mask.any():
            return None
            
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        kpts = kpts[valid_mask]
        
        # 转换边界框格式 (center -> xyxy)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.  # y1  
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.  # y2
        
        # 还原坐标到原图尺寸
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        
        # 还原边界框坐标
        boxes_xyxy -= dwdh
        boxes_xyxy /= ratio
        
        # 还原关键点坐标
        for i in range(0, kpts.shape[1], 3):
            kpts[:, i] = (kpts[:, i] - dwdh[0]) / ratio      # x坐标
            if i + 1 < kpts.shape[1]:
                kpts[:, i + 1] = (kpts[:, i + 1] - dwdh[1]) / ratio  # y坐标
        
        # NMS处理
        keep = nms(boxes_xyxy, scores[:, 0], self.iou_thresh)
        
        if len(keep) == 0:
            return None
        
        # 组合结果
        final_dets = []
        for idx in keep:
            det = np.concatenate([
                boxes_xyxy[idx],     # bbox (4)
                [scores[idx, 0]],    # conf (1) 
                kpts[idx]            # keypoints (51 = 17*3)
            ])
            final_dets.append(det)
        
        return np.array(final_dets)


    def vis_results(self, img, detections, show_keypoints=True):
        """可视化人体姿态检测结果"""
        vis_img = img.copy() 
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
            cls_id = det["cls"]
            score = det["det_conf"]
            name = det["name"]
            keypoints = det["keypoints"]
            
            bbox_color = (0, 255, 0)  # 绿色
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), bbox_color, 2)
            
            label = f"{name}: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
            
            if show_keypoints:
                kpts = keypoints.reshape(-1, 3)  # (17, 3)
                kpt_color = (0, 255, 255)  # 黄色关键点
                kpt_radius = 3
                
                for kpt in kpts:
                    x, y, conf = kpt
                    # 只绘制置信度高且坐标有效的关键点
                    if conf > self.kpt_thresh and x > 0 and y > 0:
                        cv2.circle(vis_img, (int(x), int(y)), kpt_radius, kpt_color, -1)
        
        return vis_img

if __name__ == '__main__':
    t0 = time.time()
    yolo_pose_trt = YoloPoseTrt('../../checkpoints/s010/pose/yolo11n-pose.trt')
    # yolo_pose_trt = YoloPoseTrt('/media/pxn/data/wzt/visual/checkpoints/pose/yolo11n-pose.engine') #Orin
    t1 = time.time()
    img = cv2.imread('../data/images/devforhead_video_2355.jpg')
    # img = cv2.imread('/media/pxn/data/wzt/visual/data/test_data/s009_subway/test_det.jpg')# Orin
    t2 = time.time()
    print('load_model:',t1-t0)
    print('read_img:',t2-t1)
    for i in range(5):
        t0 = time.time()
        dets = yolo_pose_trt.predict(img)
        print('cost_time:',time.time()-t0)

    result_img = yolo_pose_trt.vis_results(img, dets)
    cv2.imwrite('test_pose_result.jpg', result_img)

