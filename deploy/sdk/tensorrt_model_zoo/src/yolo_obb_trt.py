
import os
import math
import time
from datetime import datetime

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart


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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """图像letterbox预处理"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose((2, 0, 1))
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

class YoloObbTrt:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.4, num_classes=2):
        self.trt_engine = TensorRTEngine(model_path)
        self.name_map  = { 0: 'package', 1: 'barcode'}
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
    
    
    def predict(self, img):
        """使用TensorRT进行推理"""
            
        # 预处理
        processed_img, ratio, dwdh = letterbox(img, self.trt_engine.imgsz)
        
        # 推理
        data = self.trt_engine.infer(processed_img)
        # print(f"TRT output shape: {data.shape}") # (1, 7, 8400)

        data = data.transpose(0, 2, 1) # (1, 8400, 7)
        predictions = data[0]  # (8400, 7)
        # print(f"Predictions shape: {predictions.shape}")

        # 后处理
        dets = self.postprocess_obb(predictions, ratio, dwdh)
        
        if dets is None or len(dets) == 0:
            return []
        
        detections = []
        for i, box in enumerate(dets):
            cx, cy, w, h, angle, confidence, label = box
            
            rbox = np.array([cx, cy, w, h, angle])
            corners = self.xywhr2xyxyxyxy(rbox)[0]  # (4, 2)
            
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            detection = {
                "cls": int(label),
                "bbox": np.array([x1, y1, x2, y2]),
                "center": np.array([cx, cy]),
                "size": np.array([w, h]),
                "angle": angle,
                "angle_deg": math.degrees(angle),
                "corners": corners,
                "mask": None,
                "det_conf": confidence,
                "instance_id": i,
                "name": self.name_map.get(int(label), f"class_{int(label)}"),
                "qlt_id": 0
            }
            
            detections.append(detection)
        
        return detections
    
    def postprocess_obb(self, predictions, ratio, dwdh):
        """
        使用ProbIoU的OBB后处理
        
        Args:
            predictions: (N, 7) [cx, cy, w, h, cls1_conf, cls2_conf, angle]
            ratio: 缩放比例
            dwdh: 填充大小
            
        Returns:
            dets: [cx, cy, w, h, score, class_id, angle]
        """
        if len(predictions) == 0:
            return None

        boxes = predictions[:, :4]
        cls_confs = predictions[:, 4:6] 
        angles = predictions[:, 6]
        
        max_conf_indices = np.argmax(cls_confs, axis=1)
        max_confs = np.max(cls_confs, axis=1)
        
        valid_mask = max_confs >= self.conf_thresh
        if not np.any(valid_mask):
            return None
        
        boxes = boxes[valid_mask].copy()
        angles = angles[valid_mask].copy() 
        confs = max_confs[valid_mask]
        labels = max_conf_indices[valid_mask]
        

        dwdh_array = np.array(dwdh * 2)

        # letterbox逆变换
        boxes[:, 0] = (boxes[:, 0] - dwdh_array[0]) / ratio  # cx
        boxes[:, 1] = (boxes[:, 1] - dwdh_array[1]) / ratio  # cy  
        boxes[:, 2:4] = boxes[:, 2:4] / ratio                # w, h
        

        # [cx, cy, w, h, angle, confidence, label]
        result_boxes = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            result_boxes.append([
                cx, cy, w, h, 
                angles[i], 
                confs[i], 
                labels[i]
            ])
        
        # 按置信度降序排序
        result_boxes.sort(key=lambda x: x[5], reverse=True)
        
        keep_boxes = self.NMS(result_boxes, self.iou_thresh)
        
        return keep_boxes
    
    def NMS(self, boxes, iou_thres):
        """
        NMS函数 - 使用numpy版本的probiou
        """
        remove_flags = [False] * len(boxes)
        keep_boxes = []
        
        for i, ibox in enumerate(boxes):
            if remove_flags[i]:
                continue
            
            keep_boxes.append(ibox)
            
            for j in range(i + 1, len(boxes)):
                if remove_flags[j]:
                    continue
                    
                jbox = boxes[j]
                
                # 只有相同类别才进行NMS
                if ibox[6] != jbox[6]:  # ibox[6]和jbox[6]是label
                    continue
                
                iou = self.probiou(ibox, jbox)
                if iou > iou_thres:
                    remove_flags[j] = True
                    
        return keep_boxes
    
    def probiou(self, obb1, obb2, eps=1e-7):
        """
        numpy版本的probiou
        输入格式: [cx, cy, w, h, angle, confidence, label]
        """
        def covariance_matrix(obb):
            # Extract elements: w, h, r (angle)
            w, h, r = obb[2], obb[3], obb[4]  # 从 [cx,cy,w,h,angle,conf,label] 提取
            a = (w ** 2) / 12
            b = (h ** 2) / 12

            cos_r = np.cos(r)
            sin_r = np.sin(r)
            
            # Calculate covariance matrix elements
            a_val = a * cos_r ** 2 + b * sin_r ** 2
            b_val = a * sin_r ** 2 + b * cos_r ** 2
            c_val = (a - b) * sin_r * cos_r

            return a_val, b_val, c_val

        a1, b1, c1 = covariance_matrix(obb1)
        a2, b2, c2 = covariance_matrix(obb2)

        x1, y1 = obb1[0], obb1[1]  # 中心点
        x2, y2 = obb2[0], obb2[1]

        denominator = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
        t1 = ((a1 + a2) * ((y1 - y2) ** 2) + (b1 + b2) * ((x1 - x2) ** 2)) / denominator
        t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator
        
        # 计算对数项，注意数值稳定性
        det_sum = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
        det1 = a1 * b1 - c1 ** 2
        det2 = a2 * b2 - c2 ** 2
        
        det_sum = max(det_sum, eps)
        det1 = max(det1, eps) 
        det2 = max(det2, eps)
        
        t3 = np.log(det_sum / (4 * np.sqrt(det1) * np.sqrt(det2) + eps) + eps)

        bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3
        bd = np.clip(bd, eps, 100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps)
        
        return 1 - hd

    def xywhr2xyxyxyxy(self, center):
        """
        角点转换函数
        输入: [cx, cy, w, h, angle] 或 [[cx, cy, w, h, angle], ...]
        """
        center = np.array(center)
        if center.ndim == 1:
            center = center.reshape(1, -1)
        
        ctr = center[..., :2]  # 中心点
        w, h, angle = center[..., 2], center[..., 3], center[..., 4]
        
        cos_value, sin_value = np.cos(angle), np.sin(angle)
        
        # 计算向量
        vec1_x = w / 2 * cos_value
        vec1_y = w / 2 * sin_value
        vec2_x = -h / 2 * sin_value  
        vec2_y = h / 2 * cos_value
        
        # 四个角点
        pt1 = ctr + np.stack([vec1_x + vec2_x, vec1_y + vec2_y], axis=-1)
        pt2 = ctr + np.stack([vec1_x - vec2_x, vec1_y - vec2_y], axis=-1)
        pt3 = ctr + np.stack([-vec1_x - vec2_x, -vec1_y - vec2_y], axis=-1)
        pt4 = ctr + np.stack([-vec1_x + vec2_x, -vec1_y + vec2_y], axis=-1)
        
        return np.stack([pt1, pt2, pt3, pt4], axis=-2)


    def vis_results(self, img, detections, thickness=2):
        """
        可视化旋转边界框检测结果
        """
        vis_img = img.copy()  # 复制图像，避免修改原图
        
        # 定义不同类别的颜色
        colors = {
            0: (0, 255, 0),    # package - 绿色
            1: (255, 0, 0),    # barcode - 蓝色  
        }
        
        # 遍历所有检测结果进行绘制
        for det in detections:
            # print(f"Detection: {det['name']}, conf: {det['det_conf']:.3f}, angle: {det.get('angle_deg', 0):.1f}°")
            
            cls_id = det['cls']
            color = colors.get(cls_id, (128, 128, 128))
            
            if 'corners' in det and det['corners'] is not None:
                # 绘制旋转框
                corners = det['corners'].astype(np.int32)
                cv2.drawContours(vis_img, [corners], -1, color, thickness)
                
                # 绘制中心点
                # center = det['center'].astype(np.int32)
                # cv2.circle(vis_img, tuple(center), 3, color, -1)
                
                # # 绘制方向线（从中心到第一个角点，表示方向）
                # direction_point = corners[0].astype(np.int32)
                # cv2.line(vis_img, tuple(center), tuple(direction_point), color, thickness)
                
                # # 添加角度信息到标签
                # angle_deg = det.get('angle_deg', 0)
                # label = f"{det['name']} {det['det_conf']:.2f} {angle_deg:.1f}"
                label = f"{det['name']} {det['det_conf']:.2f}"
                
                # 计算标签位置
                label_pos = tuple(corners[0])
            else:
                # 如果没有角点信息或角度不显著，绘制普通矩形
                x1, y1, x2, y2 = det["bbox"].astype(int)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                
                angle_deg = det.get('angle_deg', 0)
                if abs(angle_deg) > 5:
                    label = f"{det['name']} {det['det_conf']:.2f} {angle_deg:.1f}"
                else:
                    label = f"{det['name']} {det['det_conf']:.2f}"
                label_pos = (x1, y1 - 5)
            
            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_img, 
                        (label_pos[0], label_pos[1] - label_h - 5),
                        (label_pos[0] + label_w, label_pos[1]), 
                        color, -1)
            
            # 绘制标签文字
            cv2.putText(vis_img, label, label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        
        return vis_img


if __name__ == '__main__':
    t0 = time.time()
    yolo_obb_trt = YoloObbTrt('../../checkpoints/s010/obb/s010_obb_0.0.1.trt')
    # yolo_obb_trt = YoloObbTrt('/media/pxn/data/wzt/visual/checkpoints/obb/s010_obb_0.0.1_orin.engine') # Orin
    t1 = time.time()
    img = cv2.imread('../data/images/zbar_test11.jpg')
    # img = cv2.imread('/media/pxn/data/wzt/visual/data/test_data/s010/0_2_20250909_195733_938_200.jpg') # Orin
    t2 = time.time()
    print('load_model:', t1-t0)
    print('read_img:', t2-t1)
    
    # 运行推理测试
    for i in range(5):
        t0 = time.time()
        dets = yolo_obb_trt.predict(img)
        print('cost_time:', time.time()-t0)
    
    # 可视化结果
    result_img = yolo_obb_trt.vis_results(img, dets)
    cv2.imwrite('test_det_box_obb.jpg', result_img)