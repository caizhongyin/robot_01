"""
把单目(output_extracted)和双目(output_extracted_stereo)的解析结果一起转换为 YOLO 格式训练集：
  - 检测训练集: yolo_det/   (labels 为归一化 bbox: class cx cy w h)
  - 实例分割训练集: yolo_seg/ (labels 为归一化多边形: class x1 y1 x2 y2 ...)

用法:
    python make_yolo_dataset.py
    python make_yolo_dataset.py --input output_extracted output_extracted_stereo
"""

import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image


def find_labels_jsons(input_roots: list) -> dict:
    """尝试自动定位各渲染输出目录中的 labels.json 并合并"""
    merged = {}
    for input_root in input_roots:
        candidates = [
            os.path.join(os.path.dirname(input_root),
                         os.path.basename(input_root).replace("_extracted", ""), "labels.json"),
            os.path.join(os.path.dirname(input_root), "labels.json"),
            os.path.join(input_root, "..", "labels.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                with open(c, "r", encoding="utf-8") as f:
                    merged.update(json.load(f))
                break
    return merged


def build_class_map(input_roots: list, scenes: list) -> dict:
    """返回 {category_id: {yolo_id, name}}，yolo_id 按 category_id 升序从 0 开始"""
    labels = find_labels_jsons(input_roots)
    class_map = {int(cid): name for cid, name in labels.items()} if labels else {}

    # labels.json 缺失时从标注 JSON 中收集类别
    if not class_map:
        for scene in scenes:
            for tag in ("", "L", "R"):
                bbox_path = os.path.join(scene, f"bboxes{'_' + tag if tag else ''}.json")
                if not os.path.exists(bbox_path):
                    continue
                for box in json.load(open(bbox_path, encoding="utf-8")):
                    class_map[int(box["category_id"])] = box["category_name"]

    sorted_ids = sorted(class_map)
    return {cid: {"yolo_id": idx, "name": class_map[cid]} for idx, cid in enumerate(sorted_ids)}


def bbox_to_yolo(box: dict, img_w: int, img_h: int) -> str:
    """bbox [xmin,ymin,xmax,ymax] -> 归一化 'cx cy w h'"""
    x_min, y_min, x_max, y_max = box["bbox"]
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def polygon_to_yolo(edge: dict, img_w: int, img_h: int, epsilon: float) -> str:
    """密集边缘点 -> 归一化多边形点串（可选 approxPolyDP 简化）"""
    pts = np.array(edge["edge_points"], dtype=np.float32).reshape(-1, 1, 2)
    if epsilon > 0:
        peri = cv2.arcLength(pts, True)
        pts = cv2.approxPolyDP(pts, epsilon, True)
    pts = pts.reshape(-1, 2)
    if len(pts) < 3:
        return None
    norm = pts / np.array([img_w, img_h], dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    return " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)


def collect_images_and_labels(input_roots: list) -> list:
    """收集所有输入目录的 (stem, scene_key, rgb路径, bbox路径, edges路径)

    - 双目目录: 场景下有 rgb_L/R.png + bboxes_L/R.json + edges_L/R.json
    - 单目目录: 场景下有 rgb.png + bboxes.json + edges.json
    scene_key 带来源目录前缀，保证不同目录的同名场景在 train/val 切分时互不影响
    """
    items = []
    for input_root in input_roots:
        if not os.path.isdir(input_root):
            print(f"警告: 输入目录不存在，跳过: {input_root}")
            continue
        root_name = os.path.basename(os.path.normpath(input_root))
        for scene in sorted(os.listdir(input_root)):
            scene_dir = os.path.join(input_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            scene_key = f"{root_name}|{scene}"

            # 先检测双目
            stereo_ok = all(os.path.exists(os.path.join(scene_dir, f"rgb_{tag}.png"))
                            and os.path.exists(os.path.join(scene_dir, f"bboxes_{tag}.json"))
                            and os.path.exists(os.path.join(scene_dir, f"edges_{tag}.json"))
                            for tag in ("L", "R"))
            if stereo_ok:
                for tag in ("L", "R"):
                    items.append((f"{scene}_{tag}", scene_key,
                                  os.path.join(scene_dir, f"rgb_{tag}.png"),
                                  os.path.join(scene_dir, f"bboxes_{tag}.json"),
                                  os.path.join(scene_dir, f"edges_{tag}.json")))
                continue

            # 否则检测单目
            mono_ok = all(os.path.exists(os.path.join(scene_dir, name)) for name in
                          ("rgb.png", "bboxes.json", "edges.json"))
            if mono_ok:
                items.append((scene, scene_key,
                              os.path.join(scene_dir, "rgb.png"),
                              os.path.join(scene_dir, "bboxes.json"),
                              os.path.join(scene_dir, "edges.json")))
    return items


def write_dataset(items, class_map, img_w, img_h, out_root, val_ratio, poly_epsilon, mode):
    """mode: 'det' 或 'seg'，写出 images/labels 的 train/val 结构"""
    train_dir = os.path.join(out_root, "images", "train")
    val_dir = os.path.join(out_root, "images", "val")
    train_label_dir = os.path.join(out_root, "labels", "train")
    val_label_dir = os.path.join(out_root, "labels", "val")
    for d in (train_dir, val_dir, train_label_dir, val_label_dir):
        os.makedirs(d, exist_ok=True)

    random.seed(42)
    # 按场景切分（同一场景的左右目必须一起进 train 或一起进 val）
    scenes = sorted({item[1] for item in items})
    n_val_scenes = max(1, int(round(len(scenes) * val_ratio))) if val_ratio > 0 else 0
    val_scenes = set(random.sample(scenes, n_val_scenes)) if n_val_scenes > 0 else set()

    for stem, scene_key, rgb_path, bbox_path, edge_path in items:
        is_val = scene_key in val_scenes
        img_dir, label_dir = (val_dir, val_label_dir) if is_val else (train_dir, train_label_dir)

        shutil.copy(rgb_path, os.path.join(img_dir, f"{stem}.png"))
        boxes = json.load(open(bbox_path, encoding="utf-8"))
        edges = json.load(open(edge_path, encoding="utf-8"))
        yolo_id_of = {cid: m["yolo_id"] for cid, m in class_map.items()}
        edge_by_instance = {e["instance_id"]: e for e in edges}

        lines = []
        for box in boxes:
            cid = int(box["category_id"])
            yolo_id = yolo_id_of[cid]
            if mode == "det":
                lines.append(f"{yolo_id} {bbox_to_yolo(box, img_w, img_h)}")
            else:
                edge = edge_by_instance.get(box["instance_id"])
                if edge is None:
                    continue
                poly = polygon_to_yolo(edge, img_w, img_h, poly_epsilon)
                if poly is not None:
                    lines.append(f"{yolo_id} {poly}")

        with open(os.path.join(label_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # classes.txt 与 data.yaml
    sorted_entries = sorted(class_map.items(), key=lambda kv: kv[1]["yolo_id"])
    with open(os.path.join(out_root, "classes.txt"), "w", encoding="utf-8") as f:
        for cid, m in sorted_entries:
            f.write(f"{m['name']}\n")
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_root)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for cid, m in sorted_entries:
            f.write(f"  {m['yolo_id']}: {m['name']}\n")

    return len(items), sum(1 for item in items if item[1] in val_scenes)


def main():
    parser = argparse.ArgumentParser(description="转换为 YOLO 检测/实例分割训练集")
    parser.add_argument("--input", nargs="+",
                        default=["output_extracted", "output_extracted_stereo"],
                        help="解析结果目录（单目和双目可同时传入）")
    parser.add_argument("--det-out", default="yolo_det", help="检测数据集输出目录")
    parser.add_argument("--seg-out", default="yolo_seg", help="实例分割数据集输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--poly-epsilon", type=float, default=1.0,
                        help="多边形简化阈值(像素)，0 表示保留全部边缘点")
    args = parser.parse_args()

    items = collect_images_and_labels(args.input)
    if not items:
        raise RuntimeError(f"在 {args.input} 下没有找到 rgb + bboxes + edges 数据")

    # 读取第一张图确定尺寸
    img = Image.open(items[0][2])
    img_w, img_h = img.size
    scenes = [os.path.join(root, s)
              for root in args.input
              for s in os.listdir(root)
              if os.path.isdir(os.path.join(root, s))]
    class_map = build_class_map(args.input, scenes)

    n_det, n_val = write_dataset(items, class_map, img_w, img_h, args.det_out,
                                 args.val_ratio, 0, "det")
    n_seg, _ = write_dataset(items, class_map, img_w, img_h, args.seg_out,
                             args.val_ratio, args.poly_epsilon, "seg")

    print(f"共 {n_det} 张图 (train {n_det - n_val} / val {n_val}), 类别:")
    for cid, m in sorted(class_map.items(), key=lambda kv: kv[1]["yolo_id"]):
        print(f"  [{m['yolo_id']}] {m['name']}")
    print(f"检测数据集: {args.det_out}")
    print(f"实例分割数据集: {args.seg_out}")


if __name__ == "__main__":
    main()
