#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6D 位姿可视化
=============

读取 extract_h5.py 输出的场景目录（每个场景需含 pose.json、rgb*.png、
segmentation_ids*.png、depth_16bit*.png），把每个物体的 6D 位姿绘制到 RGB 图上：

  - 红/绿/蓝三色箭头分别表示物体局部坐标系的 X/Y/Z 轴（3D 投影到图像平面）；
  - 白圈标记物体的 6D 位姿原点（世界坐标 translation 的投影）；
  - 默认锚点策略为 auto：原点投影落在物体 mask 内时坐标轴画在原点，
    否则画在物体 mask 的 3D 质心处（很多模型文件自身原点不在几何中心，
    此时画在质心更容易观察朝向；真正的原点仍用白圈标出）。

用法::

    # 默认处理 output_extracted 与 output_extracted_stereo
    python pose_viz.py

    # 指定输入目录与输出目录
    python pose_viz.py --input output_extracted output_extracted_stereo \
                       --output-dir output_pose_viz

    # 调整坐标轴长度 / 强制锚点
    python pose_viz.py --axis-length 0.25 --anchor origin

输出结构（镜像输入目录结构，不会改动原目录）::

    output_pose_viz/
    ├── output_extracted/
    │   └── 0/pose.png
    └── output_extracted_stereo/
        └── 0/pose_L.png
        └── 0/pose_R.png
"""

import argparse
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 中文标签绘制（类别名可能含中文；找不到字体时退回 cv2.putText）
# ---------------------------------------------------------------------------

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_pil_font_path = None
_pil_font_searched = False


def _load_pil_font(size: int):
    global _pil_font_path, _pil_font_searched
    if not _pil_font_searched:
        _pil_font_searched = True
        for path in _CJK_FONT_CANDIDATES:
            if os.path.exists(path):
                _pil_font_path = path
                break
    if _pil_font_path is None:
        return None
    from PIL import ImageFont
    return ImageFont.truetype(_pil_font_path, size)


def _draw_label(img, text: str, bottom_left, bg_color):
    """在 bottom_left 处绘制带背景的标签（ASCII 用 OpenCV，中文用 PIL 兜底）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    x, y = int(bottom_left[0]), int(bottom_left[1])

    if text.isascii():
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return

    from PIL import Image, ImageDraw
    fnt = _load_pil_font(18)
    if fnt is None:
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 2
    draw.rectangle([x - pad, y - bh - pad, x + bw + pad, y + pad],
                   fill=(int(bg_color[2]), int(bg_color[1]), int(bg_color[0])))
    draw.text((x - bbox[0], y - bbox[1] - bh), text, font=fnt, fill=(255, 255, 255))
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 几何：世界系 -> 相机系 -> 像素
# ---------------------------------------------------------------------------

# 物体局部坐标轴 -> 绘制颜色（BGR）
AXIS_COLORS = {
    "X": (0, 0, 255),     # 红
    "Y": (0, 255, 0),     # 绿
    "Z": (255, 0, 0),     # 蓝
}
AXIS_DIRS = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def project_point(world_pt, camera: dict, K: np.ndarray):
    """把世界坐标点投影到像素坐标；点在相机后方时返回 None。

    与 render_multi_obj.py 的相机约定一致：Blender 相机沿局部 -Z 观察，
    深度值是视轴方向的 Z 值；相机局部 +Y 向上，而图像 v 向下（左上角原点），
    因此 v = cy - fy * y_cam / z_forward。
    """
    R = np.asarray(camera["rotation_matrix"], dtype=float)
    t = np.asarray(camera["translation"], dtype=float)
    p_cam = R.T @ (np.asarray(world_pt, dtype=float) - t)
    z_forward = -p_cam[2]
    if z_forward <= 1e-6:
        return None
    u = K[0, 0] * p_cam[0] / z_forward + K[0, 2]
    v = K[1, 2] - K[1, 1] * p_cam[1] / z_forward
    return np.array([u, v], dtype=float)


def point_inside(pixel, width: int, height: int) -> bool:
    if pixel is None:
        return False
    return 0 <= pixel[0] < width and 0 <= pixel[1] < height


def mask_centroid(seg: np.ndarray, instance_id: int, depth_mm: np.ndarray = None,
                  camera: dict = None, K: np.ndarray = None):
    """返回 (3D 质心世界坐标 or None, 2D 质心像素)。

    优先用深度图对 mask 像素做加权 3D 平均再投影；没有有效深度时退化为
    像素坐标平均（此时 3D 质心为 None）。
    """
    ys, xs = np.where(seg == int(instance_id))
    if len(xs) == 0:
        return None, None

    if depth_mm is not None and camera is not None and K is not None:
        z_mm = depth_mm[ys, xs].astype(np.float32)
        valid = z_mm > 0
        if valid.any():
            z_m = z_mm[valid] / 1000.0
            px = xs[valid].astype(np.float32)
            py = ys[valid].astype(np.float32)
            p_cam = np.stack([
                (px - K[0, 2]) * z_m / K[0, 0],
                (K[1, 2] - py) * z_m / K[1, 1],
                -z_m,
            ], axis=-1)  # (M, 3) 相机坐标
            R = np.asarray(camera["rotation_matrix"], dtype=float)
            t = np.asarray(camera["translation"], dtype=float)
            p_world = t + p_cam @ R.T       # (M, 3)
            center_world = p_world.mean(axis=0)
            center_px = project_point(center_world, camera, K)
            if center_px is not None:
                return center_world, center_px

    return None, np.array([xs.mean(), ys.mean()])


def draw_object_pose(img: np.ndarray, obj: dict, camera: dict, K: np.ndarray,
                     seg: np.ndarray, depth_mm: np.ndarray,
                     axis_length: float = 0.2, line_width: int = 2,
                     anchor_mode: str = "auto"):
    """在单张 RGB 图上绘制一个物体的 6D 位姿。"""
    height, width = img.shape[:2]
    instance_id = int(obj["instance_id"])
    R_obj = np.asarray(obj["rotation_matrix"], dtype=float)
    origin_world = np.asarray(obj["translation"], dtype=float)
    origin_px = project_point(origin_world, camera, K)

    mask = seg == instance_id
    ys, xs = np.where(mask)
    has_mask = len(xs) > 0

    # 物体 mask 白边，便于确认坐标轴属于哪个实例
    if has_mask:
        contour_img = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 1)

    # 判断 6D 原点是否投影在物体 mask 内部
    origin_on_object = (point_inside(origin_px, width, height)
                        and has_mask and mask[int(origin_px[1]), int(origin_px[0])])
    use_origin = anchor_mode == "origin" or (anchor_mode == "auto" and origin_on_object)

    anchor_px = origin_px
    anchor_world = origin_world if use_origin else None
    if not use_origin:
        center_world, center_px = mask_centroid(seg, instance_id, depth_mm, camera, K)
        if center_px is not None:
            anchor_px = center_px
            anchor_world = center_world

    # 画局部坐标轴：优先用锚点 3D 坐标计算端点；深度不可用时用原点投影的
    # 2D 方向矢量平移得到相同朝向的箭头。
    if anchor_px is not None:
        for axis_name, axis_dir in AXIS_DIRS.items():
            end_px = None
            if anchor_world is not None:
                end_world = anchor_world + R_obj @ (axis_dir * axis_length)
                end_px = project_point(end_world, camera, K)
            elif origin_px is not None:
                end_world = origin_world + R_obj @ (axis_dir * axis_length)
                end_px = project_point(end_world, camera, K)
                if end_px is not None:
                    end_px = anchor_px + (end_px - origin_px)
            if end_px is None:
                continue
            start_pt = tuple(int(round(v)) for v in anchor_px)
            end_pt = tuple(int(round(v)) for v in end_px)
            cv2.arrowedLine(img, start_pt, end_pt, AXIS_COLORS[axis_name],
                            line_width, cv2.LINE_AA, tipLength=0.15)

    # 真实位姿原点标记（白底圈 + 黑描边）
    if origin_px is not None and point_inside(origin_px, width, height):
        ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
        cv2.circle(img, (ox, oy), 7, (0, 0, 0), 2)
        cv2.circle(img, (ox, oy), 7, (255, 255, 255), 1)

    # 类别 + 实例号标签
    category_name = obj.get("category_name") or f"category_{obj.get('category_id')}"
    label = f"{category_name} #{instance_id}"
    if anchor_px is not None:
        label_pt = (anchor_px[0] + 10, anchor_px[1] + 12)
        _draw_label(img, label, label_pt, (40, 40, 40))


# ---------------------------------------------------------------------------
# 目录处理
# ---------------------------------------------------------------------------

def _camera_view_paths(scene_dir: str, camera_name: str):
    """由 pose.json 中 camera.name 得到 RGB / 分割 / 深度文件名。"""
    if camera_name in ("mono", "", "single"):
        return {
            "rgb": os.path.join(scene_dir, "rgb.png"),
            "seg": os.path.join(scene_dir, "segmentation_ids.png"),
            "depth": os.path.join(scene_dir, "depth_16bit.png"),
            "suffix": "",
        }
    side = "left" if camera_name in ("left", "L") else "right"
    suffix = "_L" if side == "left" else "_R"
    return {
        "rgb": os.path.join(scene_dir, f"rgb{suffix}.png"),
        "seg": os.path.join(scene_dir, f"segmentation_ids{suffix}.png"),
        "depth": os.path.join(scene_dir, f"depth_16bit{suffix}.png"),
        "suffix": suffix,
    }


def process_scene(scene_dir: str, pose_path: str, output_dir: str, args) -> int:
    """处理单个场景目录，返回成功绘制的视图数。"""
    with open(pose_path, "r", encoding="utf-8") as f:
        pose = json.load(f)
    K = np.asarray(pose["intrinsics"], dtype=float)
    rendered = 0

    for camera in pose.get("cameras", []):
        paths = _camera_view_paths(scene_dir, camera.get("name", "mono"))
        if not all(os.path.exists(p) for p in (paths["rgb"], paths["seg"])):
            continue
        img = cv2.imread(paths["rgb"], cv2.IMREAD_COLOR)
        seg = cv2.imread(paths["seg"], cv2.IMREAD_UNCHANGED)
        if img is None or seg is None:
            continue
        depth = None
        if os.path.exists(paths["depth"]):
            depth = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)

        for obj in pose.get("objects", []):
            draw_object_pose(img, obj, camera, K, seg, depth,
                             axis_length=args.axis_length,
                             line_width=args.line_width,
                             anchor_mode=args.anchor)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"pose{paths['suffix']}.png")
        cv2.imwrite(out_path, img)
        rendered += 1
    return rendered


def process_input_root(input_root: str, output_root: str, args) -> tuple:
    """处理一个输入根目录，返回 (场景数, 视图数)。"""
    if not os.path.isdir(input_root):
        print(f"警告: 输入目录不存在，跳过: {input_root}")
        return 0, 0
    root_name = os.path.basename(os.path.normpath(input_root))
    out_root = os.path.join(output_root, root_name)
    scenes = sorted(
        name for name in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, name))
    )
    scene_count = view_count = 0
    for scene in scenes:
        scene_dir = os.path.join(input_root, scene)
        pose_path = os.path.join(scene_dir, "pose.json")
        if not os.path.exists(pose_path):
            continue
        n = process_scene(scene_dir, pose_path,
                          os.path.join(out_root, scene), args)
        if n:
            scene_count += 1
            view_count += n
    return scene_count, view_count


def main():
    parser = argparse.ArgumentParser(
        description="读取 extract_h5.py 输出的场景目录，可视化物体 6D 位姿")
    parser.add_argument("--input", nargs="+",
                        default=["output_extracted", "output_extracted_stereo"],
                        help="输入场景根目录（需含各场景的 pose.json），可多个")
    parser.add_argument("--output-dir", default="output_pose_viz",
                        help="可视化结果输出根目录，默认 output_pose_viz")
    parser.add_argument("--axis-length", type=float, default=0.2,
                        help="坐标轴长度（米），默认 0.2")
    parser.add_argument("--line-width", type=int, default=2,
                        help="坐标轴线宽（像素），默认 2")
    parser.add_argument("--anchor", choices=["auto", "origin", "centroid"],
                        default="auto",
                        help="坐标轴绘制锚点：auto=原点在 mask 内用原点否则用 mask 质心；"
                             "origin=始终画在 6D 原点；centroid=始终画在 mask 质心")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_scenes = total_views = 0
    for input_root in args.input:
        n_scenes, n_views = process_input_root(input_root, args.output_dir, args)
        print(f"{input_root}: {n_scenes} 个场景 / {n_views} 个视图")
        total_scenes += n_scenes
        total_views += n_views

    if total_views == 0:
        print("未找到可用的 pose.json / rgb 数据，请先运行 extract_h5.py")
    else:
        print(f"6D 位姿可视化完成: 共 {total_scenes} 个场景 / {total_views} 个视图，"
              f"结果保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
