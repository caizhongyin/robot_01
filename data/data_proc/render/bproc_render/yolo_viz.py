#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测 / 分割数据加载与可视化工具
====================================

支持两种主流 YOLO 标注格式（所有坐标均为 0~1 的归一化值）：

  1) 检测 (detection)     每行:  cls cx cy w h
                             cls   : 类别 id（整数，从 0 开始）
                             cx, cy: 目标中心点归一化坐标
                             w, h  : 目标宽、高（归一化）

  2) 分割 (segmentation)  每行:  cls x1 y1 x2 y2 ... xn yn
                             cls 后是闭合多边形顶点的归一化坐标（至少 3 个点）

典型数据集目录结构::

    dataset/
    ├── images/                  # 图片（支持 train/val 等子目录）
    │   ├── train/
    │   │   ├── 0001.jpg
    │   │   └── ...
    │   └── val/
    │       └── ...
    ├── labels/                  # 与图片同相对路径的 .txt 标签
    │   ├── train/0001.txt
    │   └── val/...
    ├── data.yaml                # 可选：yaml 中的 names 作为类别名
    └── classes.txt              # 可选：类别名文件（每行一个，行号即类别 id）

用法示例::

    # 1) 单张图片 + 标签（自动查找 labels/ 下同名 .txt，自动读取 data.yaml 类别名）
    python yolo_viz.py --image dataset/images/train/0001.jpg

    # 2) 批量可视化整个数据集（结果保存到 dataset/viz_output/，保留 train/val 结构）
    python yolo_viz.py --data-dir /path/to/dataset

    # 3) 一次处理检测与分割两个数据集（按 --mode auto 自动区分两种格式）
    python yolo_viz.py --data-dir yolo_det yolo_seg

    # 4) 自动生成演示数据（检测 + 分割各一张）并可视化，快速体验
    python yolo_viz.py --demo demo_data

模式说明::

    --mode auto  自动判断：标签每行都是 5 个数字 → 检测，否则 → 分割（默认）
    --mode det   强制按检测格式解析
    --mode seg   强制按分割格式解析

返回值/产物：所有可视化结果保存为图片文件；配合 --show 可弹窗预览（需图形界面）。
"""

import argparse
import colorsys
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 常见中文字体路径（用于在图上绘制中文类别名），按优先级排列
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


# ---------------------------------------------------------------------------
# 工具：颜色
# ---------------------------------------------------------------------------

def get_color(class_id: int):
    """按类别 id 生成确定性、可区分的 BGR 颜色（黄金角散开色相）。"""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# 工具：文本绘制（ASCII 用 OpenCV，中文用 PIL 兜底）
# ---------------------------------------------------------------------------

_pil_font_path = None
_pil_font_searched = False


def _load_pil_font(size: int):
    """加载第一个可用的中文字体；找不到返回 None（此时退化为 cv2.putText）。"""
    global _pil_font_path, _pil_font_searched
    if not _pil_font_searched:
        _pil_font_searched = True
        for p in _CJK_FONT_CANDIDATES:
            if os.path.exists(p):
                _pil_font_path = p
                break
    if _pil_font_path is None:
        return None
    from PIL import ImageFont
    return ImageFont.truetype(_pil_font_path, size)


def _draw_label(img, text, top_left, color, center_x=None):
    """在图片上绘制带背景的标签文本。top_left 为标签框右上角锚点。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    x, y = top_left

    if text.isascii():
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        if center_x is not None:
            x = center_x - tw // 2
        cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), color, -1)
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return

    # 中文路径：使用 PIL 渲染
    from PIL import Image, ImageDraw
    fnt = _load_pil_font(20)
    if fnt is None:
        # 无中文字体，退化为 OpenCV（会显示乱码，但流程不中断）
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    pad = 3
    bbox = draw.textbbox((0, 0), text, font=fnt)          # (left, top, right, bottom)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if center_x is not None:
        x = center_x - bw // 2
    # 先画背景矩形，再画文字
    draw.rectangle([x - pad, y - bh - pad, x + bw + pad, y + pad],
                   fill=(color[2], color[1], color[0]))
    draw.text((x - bbox[0], y - bbox[1] - bh), text, font=fnt, fill=(255, 255, 255))
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 数据加载：YOLO 检测 / 分割标签解析
# ---------------------------------------------------------------------------

def load_det_annotations(label_path):
    """读取 YOLO 检测标签。

    返回: [(class_id, (cx, cy, w, h)), ...]  全部为归一化浮点坐标
    """
    anns = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path} 第 {ln} 行不是检测格式：期望 'cls cx cy w h' 共 5 个字段，"
                    f"实际 {len(parts)} 个 ({line!r})")
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            if w <= 0 or h <= 0:
                raise ValueError(f"{label_path} 第 {ln} 行宽高必须为正，实际 {w=} {h=}")
            anns.append((cls, (cx, cy, w, h)))
    return anns


def load_seg_annotations(label_path):
    """读取 YOLO 分割标签。

    返回: [(class_id, [(x1, y1), (x2, y2), ...]), ...]  全部为归一化多边形顶点
    """
    anns = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                raise ValueError(
                    f"{label_path} 第 {ln} 行分割多边形顶点数非法：期望 'cls x1 y1 x2 y2 ...' "
                    f"（至少 3 个点，偶数个坐标），实际 {line!r}")
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            anns.append((cls, poly))
    return anns


def auto_detect_mode(label_path):
    """根据标签内容自动判断格式：所有非空行都是 5 个字段 → 检测；否则 → 分割。"""
    field_counts = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                field_counts.add(len(line.split()))
    if not field_counts:
        return "det"                       # 空标签：无内容可画，按检测处理
    return "det" if field_counts == {5} else "seg"


# ---------------------------------------------------------------------------
# 可视化：绘制检测框 / 分割掩膜
# ---------------------------------------------------------------------------

def box_to_xyxy(box, img_w, img_h):
    """归一化 (cx, cy, w, h) -> 像素 (x1, y1, x2, y2) 整数坐标。"""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def get_name(class_names, cls):
    """类别 id -> 名称；未提供名称文件时自动生成 class_{id}。"""
    if class_names is not None and 0 <= cls < len(class_names) and class_names[cls]:
        return class_names[cls]
    return f"class_{cls}"


def draw_detections(img, anns, class_names=None, thickness=2, show_label=True):
    """在图片上绘制检测框。anns: [(cls, (cx, cy, w, h)), ...]"""
    out = img.copy()
    h, w = out.shape[:2]
    for cls, box in anns:
        x1, y1, x2, y2 = box_to_xyxy(box, w, h)
        color = get_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_label:
            _draw_label(out, f"{get_name(class_names, cls)} [{cls}]", (x1, y1), color)
    return out


def draw_segmentations(img, anns, class_names=None, alpha=0.45,
                       contour_thickness=2, show_label=True):
    """在图片上绘制分割掩膜（半透明填充 + 轮廓 + 质心标签）。

    anns: [(cls, [(x, y), ...]), ...]  归一化多边形顶点
    """
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    polys = []                            # (像素多边形, 颜色, 类别 id)
    for cls, poly_norm in anns:
        if len(poly_norm) < 3:
            print(f"[warn] 类别 {cls} 多边形顶点数 < 3，跳过")
            continue
        pts = np.array([(x * w, y * h) for x, y in poly_norm], dtype=np.int32)
        color = get_color(cls)
        polys.append((pts, color, cls))
        cv2.fillPoly(overlay, [pts], color)

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)   # 半透明填充

    for pts, color, cls in polys:
        cv2.polylines(out, [pts], True, color, contour_thickness, cv2.LINE_AA)
        if show_label:
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            _draw_label(out, f"{get_name(class_names, cls)} [{cls}]", (cx, cy), color,
                        center_x=cx)
    return out


# ---------------------------------------------------------------------------
# 主流程：单张 / 批量 / 演示
# ---------------------------------------------------------------------------

def visualize(image_path, label_path, mode="auto", class_names=None,
              output=None, show=False, alpha=0.45, thickness=2):
    """加载一张图片及其标签并可视化，返回 (标注图, 解析出的标注, 实际模式)。"""
    image_path, label_path = Path(image_path), Path(label_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"标签不存在: {label_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片（格式可能不受支持）: {image_path}")

    if mode == "auto":
        mode = auto_detect_mode(label_path)

    if mode == "det":
        anns = load_det_annotations(label_path)
        out = draw_detections(img, anns, class_names, thickness=thickness)
    elif mode == "seg":
        anns = load_seg_annotations(label_path)
        out = draw_segmentations(img, anns, class_names, alpha=alpha,
                                 contour_thickness=thickness)
    else:
        raise ValueError(f"未知模式: {mode}")

    if output is None:
        output = image_path.with_name(image_path.stem + "_viz" + image_path.suffix)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), out)

    kind = {"det": "检测框", "seg": "分割掩膜"}[mode]
    print(f"[ok] {image_path.name}: {kind} x{len(anns)} -> {output}")

    if show:
        cv2.imshow(image_path.name, out)
        print("[show] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out, anns, mode


def read_data_yaml(data_dir):
    """从 data.yaml（或 dataset.yaml）读取 names 类别名列表；找不到/解析失败返回 None。

    支持常见的几种写法:
      names: {0: a, 1: b}        内联字典
      names: [a, b]              内联列表
      names:                     块状字典（0: a 换行 1: b）
        - a                     或块状列表（- a 换行 - b）
    """
    for p in (Path(data_dir) / "data.yaml", Path(data_dir) / "dataset.yaml"):
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.search(r"(?m)^names\s*:\s*(.*?)\s*$", text)
        if not m:
            return None
        rest = m.group(1)
        if rest.startswith("{"):                       # 内联字典
            return [v.strip().strip("'\"")
                    for _, v in re.findall(r"(\d+)\s*:\s*([^,}]+)", rest)] or None
        if rest.startswith("["):                       # 内联列表
            return [v.strip().strip("'\"")
                    for v in rest.strip("[]").split(",") if v.strip()] or None
        if rest in ("", "#"):                          # 块状写法
            lines = text.splitlines()
            idx = next(i for i, ln in enumerate(lines)
                       if ln.strip().startswith("names:"))
            names = []
            for ln in lines[idx + 1:]:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("- "):                 # 列表项
                    names.append(s[2:].strip().strip("'\""))
                elif re.match(r"^\d+\s*:", s):         # 字典项 "0: name"
                    names.append(s.split(":", 1)[1].strip().strip("'\""))
                else:
                    break                              # names 块结束
            return names or None
        return None
    return None


def resolve_class_names(data_dir, explicit=None):
    """解析类别名，优先级：显式指定 > data.yaml > classes.txt > None（动态 class_{id}）。"""
    if explicit is not None:
        return explicit
    yaml_names = read_data_yaml(data_dir)
    if yaml_names:
        return yaml_names
    cp = Path(data_dir) / "classes.txt"
    if cp.exists():
        return [ln.strip() for ln in cp.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    return None


def visualize_dir(data_dir, mode="auto", class_names=None, output_dir=None,
                  show=False, alpha=0.45, thickness=2):
    """批量可视化数据集。

    支持两种布局：
      * 标准 YOLO 布局：images/{train,val,...}/xxx.png + labels/ 同名 .txt
      * 扁平布局：图片与 .txt 标签在同一目录
    未显式提供类别名时自动读取 data.yaml / classes.txt。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if class_names is None:
        class_names = resolve_class_names(data_dir)
    if class_names:
        print(f"[info] {data_dir.name}: 类别名 {len(class_names)} 个: {class_names}")

    images_root = data_dir / "images"
    labels_root = data_dir / "labels"
    if not images_root.is_dir():
        images_root = data_dir                        # 扁平布局

    # 递归收集图片；跳过 labels/ 与 viz_output/ 目录，避免把输出当输入
    imgs = []
    for p in sorted(images_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        parts = p.relative_to(data_dir).parts[:-1]
        if any(part in ("labels", "viz_output") for part in parts):
            continue
        imgs.append(p)

    if not imgs:
        print(f"[warn] {data_dir} 下未找到图片（支持 {sorted(IMG_EXTS)}）")
        return

    if output_dir is None:
        output_dir = data_dir / "viz_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for ip in imgs:
        # 标签查找优先级：labels/ 下同相对路径 > 图片同目录同名 > labels/ 树下按文件名
        lp = None
        if labels_root.is_dir() and ip.is_relative_to(images_root):
            rel = ip.relative_to(images_root)
            cand = labels_root / rel.with_suffix(".txt")
            if cand.exists():
                lp = cand
        if lp is None:
            cand = ip.with_suffix(".txt")
            if cand.exists():
                lp = cand
        if lp is None and labels_root.is_dir():
            matches = sorted(labels_root.rglob(ip.stem + ".txt"))
            if matches:
                lp = matches[0]
        if lp is None:
            print(f"[skip] 缺少标签: {ip.name}（图片 {ip} 跳过）")
            stats["skip"] += 1
            continue
        try:
            rel_out = ip.relative_to(images_root) if ip.is_relative_to(images_root) \
                else ip.name
            out_path = output_dir / rel_out.with_name(rel_out.stem + "_viz" + rel_out.suffix)
            visualize(ip, lp, mode, class_names, out_path, show,
                      alpha=alpha, thickness=thickness)
            stats["ok"] += 1
        except Exception as e:
            print(f"[fail] {ip.name}: {e}")
            stats["fail"] += 1

    print("-" * 50)
    print(f"完成: 成功 {stats['ok']}，跳过 {stats['skip']}，失败 {stats['fail']}"
          f"，输出目录: {output_dir}")


def make_demo(output_dir, size=(640, 480), seed=7):
    """生成演示数据（检测 + 分割各一张图片及标签），返回数据目录。"""
    out = Path(output_dir)
    images_dir, labels_dir = out / "images", out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_w, img_h = size

    def random_bg():
        """随机噪点背景，模拟简单场景。"""
        img = np.full((img_h, img_w, 3), int(rng.integers(35, 70)), dtype=np.uint8)
        for _ in range(rng.integers(18, 35)):
            c = tuple(int(v) for v in rng.integers(0, 90, 3))
            cv2.circle(img, (int(rng.integers(0, img_w)), int(rng.integers(0, img_h))),
                       int(rng.integers(4, 20)), c, -1)
        return img

    # ---- 检测示例：3 个彩色矩形目标 ----
    img = random_bg()
    det_lines = []
    for cls, (cx, cy, bw, bh) in enumerate([(0.30, 0.35, 0.22, 0.28),
                                            (0.68, 0.30, 0.26, 0.20),
                                            (0.50, 0.72, 0.32, 0.22)]):
        color = get_color(cls)
        x1, y1 = int((cx - bw / 2) * img_w), int((cy - bh / 2) * img_h)
        x2, y2 = int((cx + bw / 2) * img_w), int((cy + bh / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)          # 实心色块
        cv2.putText(img, f"obj{cls}", (x1 + 8, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        det_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    cv2.imwrite(str(images_dir / "img_001_det.jpg"), img)
    (labels_dir / "img_001_det.txt").write_text("\n".join(det_lines) + "\n", "utf-8")

    # ---- 分割示例：3 个椭圆目标（多边形近似） ----
    img = random_bg()
    seg_lines = []
    for cls, (cx, cy, rx, ry) in enumerate([(0.32, 0.38, 0.14, 0.11),
                                            (0.66, 0.32, 0.12, 0.15),
                                            (0.50, 0.70, 0.20, 0.09)]):
        color = get_color(cls)
        cv2.ellipse(img, (int(cx * img_w), int(cy * img_h)),
                    (int(rx * img_w), int(ry * img_h)), 0, 0, 360, color, -1)
        t = np.linspace(0, 2 * np.pi, 9)[:-1]                     # 8 顶点多边形
        xs = cx + rx * np.cos(t)
        ys = cy + ry * np.sin(t)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        seg_lines.append(f"{cls} {coords}")
    cv2.imwrite(str(images_dir / "img_002_seg.jpg"), img)
    (labels_dir / "img_002_seg.txt").write_text("\n".join(seg_lines) + "\n", "utf-8")

    # ---- 类别名文件（含中文，演示中文字体绘制） ----
    (out / "classes.txt").write_text("汽车\n行人\n自行车\n", "utf-8")

    print(f"[demo] 演示数据已生成: {out}")
    print(f"       {images_dir / 'img_001_det.jpg'}   (检测标注)")
    print(f"       {images_dir / 'img_002_seg.jpg'}   (分割标注)")
    return out


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="YOLO 检测/分割标注加载与可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", metavar="IMG", help="单张图片路径")
    src.add_argument("--data-dir", metavar="DIR", nargs="+",
                     help="数据集目录，可传多个（如 --data-dir yolo_det yolo_seg）；"
                          "支持 images/labels 下的 train/val 等子目录")
    src.add_argument("--demo", metavar="DIR",
                     help="生成演示数据到 DIR 并立即可视化（快速上手）")
    ap.add_argument("--label", metavar="TXT",
                    help="单张标签路径（配合 --image；缺省时自动查找）")
    ap.add_argument("--mode", choices=["auto", "det", "seg"], default="auto",
                    help="解析模式（默认 auto 自动判断）")
    ap.add_argument("--class-names", metavar="FILE",
                    help="类别名文件，每行一个名字，行号即类别 id")
    ap.add_argument("--classes", type=int, metavar="N",
                    help="类别数量（自动生成 class_0 ~ class_{N-1} 名称）")
    ap.add_argument("--output", metavar="FILE", help="输出图片路径（单张模式）")
    ap.add_argument("--output-dir", metavar="DIR",
                    help="输出目录（批量模式，默认 <data_dir>/viz_output）")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="分割掩膜透明度 0~1（默认 0.45）")
    ap.add_argument("--thickness", type=int, default=2, help="框/轮廓线宽（默认 2）")
    ap.add_argument("--show", action="store_true", help="弹窗预览（需图形界面）")
    args = ap.parse_args(argv)

    # 类别名解析
    class_names = None
    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]
        print(f"[info] 加载类别名 {len(class_names)} 个: {class_names}")
    elif args.classes is not None:
        class_names = [f"class_{i}" for i in range(args.classes)]

    try:
        if args.demo:
            data_dir = make_demo(args.demo)
            visualize_dir(data_dir, mode="auto", class_names=class_names,
                          show=args.show, alpha=args.alpha, thickness=args.thickness)
        elif args.data_dir:
            for d in args.data_dir:
                out_dir = None
                if args.output_dir:                     # 多数据集时按数据集名分目录存放
                    out_dir = Path(args.output_dir) / Path(d).name
                visualize_dir(d, mode=args.mode, class_names=class_names,
                              output_dir=out_dir, show=args.show,
                              alpha=args.alpha, thickness=args.thickness)
        elif args.image:
            img_path = Path(args.image)
            # 自动查找标签：显式指定 > 图片同目录同名 > 数据集 labels/ 下
            # （保留 split 子目录 > 按文件名）
            candidates = []
            if args.label:
                candidates.append(Path(args.label))
            else:
                candidates.append(img_path.with_suffix(".txt"))
                # 向上找到名为 images 的目录，视为数据集 images/ 根
                parent = img_path.parent
                while parent != parent.parent:
                    if parent.name == "images":
                        rel = img_path.relative_to(parent)
                        labels_root = parent.parent / "labels"
                        candidates += [labels_root / rel.with_suffix(".txt"),
                                       labels_root / (img_path.stem + ".txt")]
                        break
                    parent = parent.parent
                candidates.append(img_path.parent.parent / "labels" / (img_path.stem + ".txt"))
            label_path = next((p for p in candidates if p.exists()), None)
            if label_path is None:
                ap.error(f"未找到标签文件（已尝试: {', '.join(map(str, candidates))}，"
                         f"可用 --label 指定）")
            # 未显式提供类别名时，从图片所在数据集目录自动读取
            if class_names is None:
                class_names = resolve_class_names(img_path.parent.parent)
            visualize(img_path, label_path, mode=args.mode, class_names=class_names,
                      output=args.output, show=args.show,
                      alpha=args.alpha, thickness=args.thickness)
        else:
            ap.error("请提供 --image、--data-dir 或 --demo 之一（-h 查看帮助）")
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
