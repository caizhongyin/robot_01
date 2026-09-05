
import os
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart
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


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """图像letterbox预处理"""
    # 获取原始图像尺寸
    h, w = img.shape[:2]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 计算缩放比例
    r = min(new_shape[0] / h, new_shape[1] / w)
    
    # 计算新尺寸和填充
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    
    # 调整图像大小
    if (w, h) != new_unpad:
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img
    
    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=color)

    img_processed = img_padded[..., [2, 1, 0]].transpose(2, 0, 1).astype(np.float32)
    img_processed /= 255.0
    
    # 添加batch维度
    img_batch = np.ascontiguousarray(img_processed[np.newaxis])
    return img_batch, r, (dw, dh)

def xywh2xyxy(x):
    """Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Scale boxes from img1_shape to img0_shape
    """
    # Calculate ratio and padding
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    
    # Clip boxes to image bounds
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, img0_shape[0])  # y1, y2
    
    return boxes

def nms(boxes, scores, iou_threshold):
    """
    Numpy implementation of Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    
    # 按置信度排序
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
            
        current_box = boxes[current]
        other_boxes = boxes[sorted_indices[1:]]
        
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        union = current_area + other_areas - intersection
        iou = intersection / union
        
        keep_mask = iou <= iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return keep


def non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.2, classes=None, max_det=300,
                        num_classes=3, mask_dim=None):
    """优化的NMS实现"""
    if predictions.ndim == 3:
        predictions = predictions[0]

    N, D = predictions.shape
    if mask_dim is None:
        raise ValueError("non_max_suppression requires mask_dim")

    # 提取数据
    boxes_cxcywh = predictions[:, :4]
    cls_scores = predictions[:, 4:4 + num_classes]
    mask_coeffs = predictions[:, 4 + num_classes: 4 + num_classes + mask_dim]
    
    # 置信度筛选
    max_class_scores = np.max(cls_scores, axis=1)
    class_indices = np.argmax(cls_scores, axis=1)
    
    conf_mask = max_class_scores > conf_thres
    if not np.any(conf_mask):
        return np.empty((0, 6 + mask_dim), dtype=np.float32)
    
    boxes = boxes_cxcywh[conf_mask]
    max_class_scores = max_class_scores[conf_mask]
    class_indices = class_indices[conf_mask]
    mask_coeffs = mask_coeffs[conf_mask]
    
    # 转换坐标
    boxes = xywh2xyxy(boxes)
    
    # 分类别NMS
    final_detections = []
    for class_id in range(num_classes):
        class_mask = (class_indices == class_id)
        if not np.any(class_mask):
            continue
            
        class_boxes = boxes[class_mask]
        class_scores = max_class_scores[class_mask]
        class_mask_coeffs = mask_coeffs[class_mask]
        
        keep_indices = nms(class_boxes, class_scores, iou_thres)
        if len(keep_indices) == 0:
            continue
            
        det = np.column_stack([
            class_boxes[keep_indices],
            class_scores[keep_indices],
            np.full(len(keep_indices), class_id),
            class_mask_coeffs[keep_indices]
        ])
        final_detections.append(det)
    
    if len(final_detections) == 0:
        return np.empty((0, 6 + mask_dim), dtype=np.float32)
    
    final_detections = np.vstack(final_detections)
    if len(final_detections) > max_det:
        final_detections = final_detections[np.argsort(final_detections[:, 4])[::-1][:max_det]]
    
    return final_detections

def process_mask(protos, masks_in, bboxes, orig_shape, input_shape, ratio, dwdh, thresh=0.5):
    """
    Process prototype masks with predicted mask coefficients to generate instance segmentation masks.
    
    Args:
    - protos: [c, mh, mw]
    - masks_in: [n, c]
    - bboxes: [n, 4]，注意这里传"原图坐标"的 bbox
    - orig_shape: (H0, W0) 原图尺寸
    - input_shape: (Hi, Wi) 网络输入尺寸（letterbox 后）
    - ratio, dwdh: 预处理时的缩放比例与 padding
        
    Returns:
    - binary_masks: [n, H0, W0]，bool
    """

    c, mh, mw = protos.shape
    n = masks_in.shape[0]
    if n == 0:
        return np.zeros((0, orig_shape[0], orig_shape[1]), dtype=bool)
    
    # print(f"process_mask: protos {protos.shape}, masks_in {masks_in.shape}, bboxes {bboxes.shape}")
    
    # 确保掩码系数维度匹配(masks_in 的列数 == c)
    if masks_in.shape[1] != c:
        raise ValueError(f"mask coeff dim mismatch: got {masks_in.shape[1]}, expect {c}")

    # 低分辨率掩码: [n, mh, mw]
    protos_flat = protos.reshape(c, -1)  # [c, mh*mw]
    masks = (masks_in @ protos_flat).reshape(n, mh, mw)
    masks = 1.0 / (1.0 + np.exp(-masks))  # sigmoid

    # 将原图 bbox -> letterbox(输入)坐标
    # x_in = x_orig * ratio + dw, y_in = y_orig * ratio + dh
    dw, dh = dwdh
    Hi, Wi = input_shape
    hx, wx = mh / Hi, mw / Wi  # 从输入坐标缩放到掩码网格坐标的比例

    boxes_in = bboxes.copy()
    boxes_in[:, [0, 2]] = boxes_in[:, [0, 2]] * ratio + dw
    boxes_in[:, [1, 3]] = boxes_in[:, [1, 3]] * ratio + dh

    # 输入坐标 -> 掩码网格坐标
    boxes_m = boxes_in.copy()
    boxes_m[:, [0, 2]] *= wx
    boxes_m[:, [1, 3]] *= hx

    # 在低分辨率掩码上裁剪
    masks = crop_mask(masks, boxes_m)

    # 上采样到输入大小(Hi, Wi)
    masks_up = np.stack([cv2.resize(m, (Wi, Hi), interpolation=cv2.INTER_LINEAR) for m in masks], axis=0)
    # masks_up = np.stack([cv2.resize(m, (Wi, Hi), interpolation=cv2.INTER_NEAREST) for m in masks], axis=0)

    # unletterbox
    top = int(round(dh - 0.1))
    bottom = int(round(Hi - (dh + 0.1)))
    left = int(round(dw - 0.1))
    right = int(round(Wi - (dw + 0.1)))

    masks_unpad = masks_up[:, top:bottom, left:right]

    # 缩放回原图大小
    H0, W0 = orig_shape
    masks_final = np.stack([cv2.resize(m, (W0, H0), interpolation=cv2.INTER_LINEAR) for m in masks_unpad], axis=0)

    # 二值化
    binary_masks = masks_final > thresh
    return binary_masks

# def crop_mask(masks, boxes):
#     """
#     Crop masks to bounding boxes
#     Args:
#         masks: [n, h, w] tensor of masks
#         boxes: [n, 4] tensor of bbox coords [x1, y1, x2, y2]
#     Returns:
#         masks: cropped masks
#     """
#     n, h, w = masks.shape
#     x1, y1, x2, y2 = np.split(boxes, 4, axis=1)  # [n, 1] each
    
#     # Create coordinate grids
#     r = np.arange(w)[None, None, :]  # [1, 1, w]
#     c = np.arange(h)[None, :, None]  # [1, h, 1]
    
#     # Create mask for region inside bounding boxes
#     crop_mask_array = ((r >= x1[:, :, None]) & (r < x2[:, :, None]) & 
#                        (c >= y1[:, None, :]) & (c < y2[:, None, :]))  # [n, h, w]
    
#     return masks * crop_mask_array

def crop_mask(masks, boxes):
#     """
#     Crop masks to bounding boxes
#     Args:
#         masks: [n, h, w] tensor of masks
#         boxes: [n, 4] tensor of bbox coords [x1, y1, x2, y2]
#     Returns:
#         masks: cropped masks
#     """
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Create coordinate grids
    r = np.arange(w, dtype=np.float32)
    c = np.arange(h, dtype=np.float32)
    
    # Create mask for region inside bounding boxes
    crop_masks = []
    for i in range(n):
        x_mask = (r >= x1[i]) & (r < x2[i])
        y_mask = (c >= y1[i]) & (c < y2[i])
        mask_region = y_mask[:, None] & x_mask[None, :]
        crop_masks.append(masks[i] * mask_region)
    
    return np.array(crop_masks)


def mask_to_polygon(mask, epsilon_factor=0.005, min_area=100):
    """
    将二值掩码转换为多边形轮廓
    """
    if mask is None or not np.any(mask):
        return []
    
    # 确保是二值掩码
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    polygons = []
    for contour in contours:
        # 过滤小轮廓
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # 轮廓简化
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为坐标点列表
        if len(approx) >= 3:  # 至少3个点才能构成多边形
            polygon = approx.reshape(-1, 2).tolist()
            polygons.append(polygon)
    
    return polygons

        
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
    
    def __del__(self):
        """清理GPU内存"""
        for allocation in self.allocations:
            cuda_call(cudart.cudaFree(allocation))

class YoloSegTrt:
    def __init__(self, model_path, conf_thresh=0.4, iou_thresh=0.2, num_classes=3):
        self.trt_engine = TensorRTEngine(model_path)
        self.name_map  = { 0: 'package', 1: 'barcode', 2: 'clothing_box'}
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
    
    def predict(self, img, return_polygons=True):
        """
        使用TensorRT进行推理，返回polygon结果
        Args:
            img: 输入图像
            conf: 置信度阈值
            return_polygons: 是否返回多边形（为True时返回polygon，为False时返回bbox）
        Returns:
            list: 检测结果列表，每个元素包含bbox和polygon信息
        """
            
        # 预处理
        prep_img, ratio, dwdh = letterbox(img, self.trt_engine.imgsz)
        
        # 推理
        outs = self.trt_engine.infer(prep_img)

        # print(outs[0].shape, outs[1].shape)
        
        # 后处理
        return self.postprocess(img, prep_img, outs, return_polygons, ratio, dwdh)

    def postprocess(self, img, prep_img, outs, return_polygons=True, ratio=None, dwdh=None):
        preds, protos = outs  # [1, D, N], [1, C, mh, mw] 或 [C, mh, mw]
        # print(f"Preds shape: {preds.shape}, Protos shape: {protos.shape}")

        preds = preds.transpose(0, 2, 1)[0]  # -> (N, D)

        # 原型通道数（掩码系数维度）
        proto_data = protos[0] if len(protos.shape) == 4 else protos  # [C, mh, mw]
        mask_dim = int(proto_data.shape[0])

        # NMS
        preds_nms = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, 
                                       num_classes=self.num_classes, mask_dim=mask_dim)
        if len(preds_nms) == 0:
            return []

        # 缩放 bbox 回原图
        pred_scaled = preds_nms.copy()
        pred_scaled[:, :4] = scale_boxes(prep_img.shape[2:], pred_scaled[:, :4], img.shape)

        # 处理mask
        masks = None
        if return_polygons and protos is not None:
            mask_coeffs = pred_scaled[:, 6: 6 + mask_dim]
            bboxes = pred_scaled[:, :4]

            masks = process_mask(
                proto_data, mask_coeffs, bboxes,
                orig_shape=img.shape[:2],
                input_shape=prep_img.shape[2:],  # (Hi, Wi)
                ratio=ratio, dwdh=dwdh, thresh=0.5
            )

        results = []
        for i in range(len(pred_scaled)):
            x1, y1, x2, y2, conf, cls = pred_scaled[i, :6]
            cls_id = int(cls)

            polygons = []
            if return_polygons and masks is not None and i < len(masks):
                mask = masks[i].astype(np.uint8)  # 0/1
                polygons = mask_to_polygon(mask, epsilon_factor=0.003, min_area=100)

            results.append({
                "cls": cls_id,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "polygons": polygons,
                "det_conf": float(conf),
                "instance_id": i,
                "name": self.name_map.get(cls_id, f"class_{cls_id}"),
                "qlt_id": 0
            })

            # print(f"Detection {i+1}: {results[-1]['name']} ({conf}), "
            #     f"bbox: ({x1},{y1},{x2},{y2}), polygons: {len(polygons)}")

        return results
    
    def vis_results(self, img, detections, show_polygons=True):
        """可视化检测和多边形结果"""
        vis_img = img.copy()
        
        colors = {
            0: (0, 255, 0),    # package - 绿色
            1: (255, 0, 0),    # barcode - 蓝色
            2: (0, 0, 255),    # clothing_box - 红色
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
            cls_id = det["cls"]
            score = det["det_conf"]
            name = det["name"]
            polygons = det["polygons"]
            
            color = colors.get(cls_id, (128, 128, 128))  # 默认灰色
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name}: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if show_polygons and polygons:
                for polygon in polygons:
                    if len(polygon) >= 3:
                        pts = np.array(polygon, dtype=np.int32)
                        cv2.fillPoly(vis_img, [pts], color=(color[0]//4, color[1]//4, color[2]//4))
                        cv2.polylines(vis_img, [pts], True, color, 2)
        
        return vis_img
    

if __name__ == '__main__':
    t0 = time.time()
    yolo_seg_trt = YoloSegTrt('../../checkpoints/s010/segemetation/s010_seg_0.0.1.trt')
    # yolo_seg_trt = YoloSegTrt('/media/pxn/data/wzt/visual/checkpoints/segemetation/s010_seg_0.0.1_orin.engine') #Orin
    t1 = time.time()
    img = cv2.imread('../data/images/20250909_115218_018_2_20250909_195722_595_100.jpg')
    # img = cv2.imread('/media/pxn/data/wzt/visual/data/test_data/s010/20250909_120603_156_2_20250909_195822_759_260.jpg') # Orin
    t2 = time.time()
    print('load_model:',t1-t0)
    print('read_img:',t2-t1)
    # 性能测试
    for i in range(5):
        t0 = time.time()
        dets = yolo_seg_trt.predict(img, return_polygons=True)
        print('cost_time:',time.time()-t0)
    # 保存可视化结果
    vis_img = yolo_seg_trt.vis_results(img, dets, show_polygons=True)
    cv2.imwrite('test_seg_result.jpg', vis_img)
