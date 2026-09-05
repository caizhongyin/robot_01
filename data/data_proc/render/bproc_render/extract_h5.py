"""
解析 BlenderProc 生成的 HDF5 文件，输出 RGB 图像、深度图、分割 mask、边缘点，
并把 HDF5 中保存的物体 6D 位姿解析为每个场景目录下的 pose.json。

用法:
    python extract_h5.py [input_dir] [output_root]

默认:
    input_dir  = ./output
    output_root = ./output_extracted
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np
from PIL import Image


# 深度图中背景像素的远平面值（BlenderProc 惯例），超过该值视为无效
DEPTH_INVALID_THRESHOLD = 1e6


def _decode_json_value(raw):
    """把 HDF5 里以 JSON 字节串/数组保存的数据还原为 Python 对象"""
    if isinstance(raw, np.ndarray):
        raw = raw[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def parse_instance_attribute_maps(raw) -> list:
    """解析 instance_attribute_maps（JSON bytes/str），返回 [{idx, category_id, instance_id}, ...]"""
    return _decode_json_value(raw)


def save_depth(depth: np.ndarray, scene_dir: str, bg_depth: float = None, suffix: str = ""):
    """保存深度图：16bit 毫米图 + 伪彩色可视化图

    :param bg_depth: 背景深度（米）。为 None 时背景置 0；给定数值时背景像素使用该深度
    """
    invalid = (depth >= DEPTH_INVALID_THRESHOLD) | (depth <= 0)
    if bg_depth is not None:
        depth = depth.copy()
        depth[invalid] = bg_depth
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = ~invalid

    # 16bit 毫米深度图（背景为 0）
    depth_mm = np.zeros_like(depth, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(os.path.join(scene_dir, f"depth_16bit{suffix}.png"))

    # 伪彩色可视化：按有效深度范围归一化
    norm_full = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
        norm = np.clip((depth[valid] - dmin) / max(dmax - dmin, 1e-9), 0, 1)
        norm_full[valid] = (norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(norm_full, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    Image.fromarray(depth_vis).save(os.path.join(scene_dir, f"depth_vis{suffix}.png"))


def extract_edges(seg: np.ndarray, attrs: list) -> list:
    """从实例分割 mask 中提取每个实例的最外层边缘点（密集轮廓点）"""
    id_to_attrs = {int(a["idx"]): a for a in attrs}
    edges_data = []
    for inst_id in np.unique(seg):
        inst_id = int(inst_id)
        if inst_id == 0:
            continue
        mask = (seg == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        points = largest.reshape(-1, 2).astype(int).tolist()
        info = id_to_attrs.get(inst_id, {})
        edges_data.append({
            "instance_id": inst_id,
            "category_id": info.get("category_id"),
            "category_name": info.get("category_name"),
            "num_points": len(points),
            "edge_points": points,
        })
    return edges_data


def extract_bboxes(seg: np.ndarray, attrs: list) -> list:
    """从实例分割 mask 中提取每个实例的矩形框"""
    id_to_attrs = {int(a["idx"]): a for a in attrs}
    bboxes = []
    for inst_id in np.unique(seg):
        inst_id = int(inst_id)
        if inst_id == 0:
            continue
        mask = seg == inst_id
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        info = id_to_attrs.get(inst_id, {})
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bboxes.append({
            "instance_id": inst_id,
            "category_id": info.get("category_id"),
            "category_name": info.get("category_name"),
            "bbox": [x_min, y_min, x_max, y_max],
            "width": x_max - x_min,
            "height": y_max - y_min,
        })
    return bboxes


def save_segmentation(seg: np.ndarray, scene_dir: str, suffix: str = ""):
    """保存分割 mask：16bit 实例 ID 图 + 每个实例随机颜色的可视化图"""
    Image.fromarray(seg.astype(np.uint16)).save(os.path.join(scene_dir, f"segmentation_ids{suffix}.png"))

    rng = np.random.RandomState(42)
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for inst_id in np.unique(seg):
        if int(inst_id) == 0:
            continue
        seg_color[seg == inst_id] = rng.randint(0, 255, size=3)
    Image.fromarray(seg_color).save(os.path.join(scene_dir, f"segmentation_color{suffix}.png"))


def draw_edges(rgb: np.ndarray, edges_data: list, scene_dir: str, suffix: str = ""):
    """在 RGB 图上绘制边缘点并保存"""
    vis = rgb.copy()
    for edge in edges_data:
        pts = np.array(edge["edge_points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(vis, [pts], -1, (0, 255, 0), 2)
    Image.fromarray(vis).save(os.path.join(scene_dir, f"rgb_with_edges{suffix}.png"))


def draw_bboxes(rgb: np.ndarray, bboxes: list, scene_dir: str, suffix: str = ""):
    """在 RGB 图上绘制矩形框（含类别名）并保存"""
    vis = rgb.copy()
    for box in bboxes:
        x_min, y_min, x_max, y_max = box["bbox"]
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{box.get('category_name', 'unknown')}"
        cv2.putText(vis, label, (x_min, max(y_min - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(scene_dir, f"rgb_with_boxes{suffix}.png"))


def load_labels(hdf5_path: str, attrs: list) -> dict:
    """优先读取渲染时保存的 labels.json，否则用 instance_attribute_maps 兜底"""
    labels_path = os.path.join(os.path.dirname(hdf5_path), "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        for attr in attrs:
            cid = str(attr.get("category_id"))
            if cid not in labels and "category_name" in attr:
                labels[cid] = attr["category_name"]
        return labels
    return {str(a.get("category_id")): a.get("category_name", f"category_{a.get('category_id')}")
            for a in attrs}


def extract_pose_json(f: h5py.File, labels: dict, scene_name: str, width: int, height: int):
    """从 HDF5 的 object_pose_* 数据集解析该帧物体的 6D 位姿，返回 JSON 结构。

    位姿为世界坐标系（单位米）：translation 是 3D 平移，rotation_matrix 是 3x3
    旋转矩阵，rotation_euler_xyz 是弧度欧拉角；scale 是归一化缩放（不属于 6D
    位姿，但保留用于还原实际几何）。旧版无位姿数据的 HDF5 返回 None。
    """
    if "object_pose_translations" not in f:
        return None

    instance_ids = np.asarray(f["object_pose_instance_ids"][()], dtype=np.int32).reshape(-1)
    category_ids = np.asarray(f["object_pose_category_ids"][()], dtype=np.int32).reshape(-1)
    num_objects = len(instance_ids)
    translations = np.asarray(f["object_pose_translations"][()], dtype=float).reshape(num_objects, 3)
    rotations = np.asarray(f["object_pose_rotations"][()], dtype=float).reshape(num_objects, 3, 3)
    eulers = np.asarray(f["object_pose_eulers"][()], dtype=float).reshape(num_objects, 3)
    scales = np.asarray(f["object_pose_scales"][()], dtype=float).reshape(num_objects, 3)
    obj_paths = _decode_json_value(f["object_pose_paths"][()])
    if len(obj_paths) != num_objects:
        raise ValueError(f"object_pose_paths 数量 {len(obj_paths)} 与物体数量 {num_objects} 不一致")

    camera_matrices = np.asarray(f["camera_pose_matrices"][()], dtype=float)
    camera_names = _decode_json_value(f["camera_names"][()])
    intrinsics = np.asarray(f["camera_intrinsics"][()], dtype=float)
    if camera_matrices.ndim != 3 or camera_matrices.shape[1:] != (4, 4):
        raise ValueError(f"camera_pose_matrices 形状异常: {camera_matrices.shape}")
    if len(camera_names) != len(camera_matrices):
        raise ValueError("camera_names 与 camera_pose_matrices 数量不一致")

    objects = []
    for idx in range(num_objects):
        category_id = int(category_ids[idx])
        objects.append({
            "instance_id": int(instance_ids[idx]),
            "category_id": category_id,
            "category_name": labels.get(str(category_id), f"category_{category_id}"),
            "obj_path": obj_paths[idx],
            "frame": "world",
            "translation": translations[idx].tolist(),
            "rotation_matrix": rotations[idx].tolist(),
            "rotation_euler_xyz": eulers[idx].tolist(),
            "scale": scales[idx].tolist(),
        })

    cameras = []
    for idx, mat in enumerate(camera_matrices):
        cameras.append({
            "name": camera_names[idx],
            "frame": "world",
            "rotation_matrix": mat[:3, :3].tolist(),
            "translation": mat[:3, 3].tolist(),
        })

    try:
        scene_id = int(scene_name)
    except ValueError:
        scene_id = scene_name
    return {
        "scene_id": scene_id,
        "stereo": len(camera_names) > 1,
        "width": int(width),
        "height": int(height),
        "intrinsics": intrinsics.tolist(),
        "cameras": cameras,
        "objects": objects,
    }


def process_eye(rgb: np.ndarray, depth: np.ndarray, seg: np.ndarray, attrs: list,
                scene_dir: str, tag: str = "", bg_depth: float = None):
    """处理单个视图（单目 tag=""，双目 tag="L"/"R"），返回 (边缘数, 框数)"""
    suffix = f"_{tag}" if tag else ""
    Image.fromarray(rgb).save(os.path.join(scene_dir, f"rgb{suffix}.png"))
    save_depth(depth, scene_dir, bg_depth=bg_depth, suffix=suffix)
    save_segmentation(seg, scene_dir, suffix=suffix)

    edges_data = extract_edges(seg, attrs)
    with open(os.path.join(scene_dir, f"edges{suffix}.json"), "w", encoding="utf-8") as jf:
        json.dump(edges_data, jf, indent=2, ensure_ascii=False)
    draw_edges(rgb, edges_data, scene_dir, suffix=suffix)

    bboxes = extract_bboxes(seg, attrs)
    with open(os.path.join(scene_dir, f"bboxes{suffix}.json"), "w", encoding="utf-8") as jf:
        json.dump(bboxes, jf, indent=2, ensure_ascii=False)
    draw_bboxes(rgb, bboxes, scene_dir, suffix=suffix)
    return len(edges_data), len(bboxes)


def extract_hdf5(hdf5_path: str, output_root: str, bg_depth: float = None):
    scene_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        attrs = parse_instance_attribute_maps(f["instance_attribute_maps"][()])
        labels = load_labels(hdf5_path, attrs)
        for attr in attrs:
            attr["category_name"] = labels.get(str(attr.get("category_id")), "background")

        colors_data = np.array(f["colors"])
        depth_data = np.array(f["depth"])
        seg_data = np.array(f["instance_id_segmaps"])
        is_stereo = colors_data.ndim == 4  # 双目形状 [2, H, W, 3]
        height, width = colors_data.shape[-3], colors_data.shape[-2]
        pose_data = extract_pose_json(f, labels, scene_name, width, height)
        results = []
        if is_stereo:
            # 双目：第一维 [0]=左目 [1]=右目
            for tag, eye_idx in [("L", 0), ("R", 1)]:
                rgb = colors_data[eye_idx, ..., :3].astype(np.uint8)
                depth = depth_data[eye_idx]
                seg = seg_data[eye_idx]
                n_edges, n_boxes = process_eye(rgb, depth, seg, attrs, scene_dir,
                                               tag=tag, bg_depth=bg_depth)
                results.append((tag, n_edges, n_boxes))
        else:
            rgb = colors_data[..., :3].astype(np.uint8)
            depth = depth_data
            seg = seg_data
            n_edges, n_boxes = process_eye(rgb, depth, seg, attrs, scene_dir,
                                           tag="", bg_depth=bg_depth)
            results.append(("单目", n_edges, n_boxes))

    summary = ", ".join(f"{tag}: {ne}边缘/{nb}框" for tag, ne, nb in results)
    if pose_data is not None:
        pose_path = os.path.join(scene_dir, "pose.json")
        with open(pose_path, "w", encoding="utf-8") as jf:
            json.dump(pose_data, jf, indent=2, ensure_ascii=False)
        summary += f", 6D位姿 {len(pose_data['objects'])} 个物体"
    else:
        summary += ", HDF5 中无 object_pose_* 位姿数据（旧版渲染文件？）"
    print(f"{scene_name}: {summary}, 结果已保存到 {scene_dir}")


def main():
    parser = argparse.ArgumentParser(description="解析 BlenderProc HDF5 输出")
    parser.add_argument("input_dir", nargs="?", default="output", help="HDF5 文件所在目录")
    parser.add_argument("output_root", nargs="?", default="output_extracted", help="输出根目录")
    parser.add_argument("--bg-depth", type=float, default=None,
                        help="背景深度（米）；不指定则背景置 0，指定后背景像素使用该深度")
    args = parser.parse_args()

    hdf5_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".hdf5")
    )
    if not hdf5_files:
        print(f"在 {args.input_dir} 下未找到 .hdf5 文件")
        return
    for path in hdf5_files:
        extract_hdf5(path, args.output_root, bg_depth=args.bg_depth)


if __name__ == "__main__":
    main()
