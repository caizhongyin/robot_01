import blenderproc as bproc

# 多物体随机组合渲染：加载 obj_list.txt，每张图随机挑选 1~N 个物体（可重复），
# 生成 RGB / 深度 / 分割 / 边缘点 / 物体框 / 类别名称。
# 用法:
#   bash gen_obj_list.sh                 # 第一步：find 生成 obj_list.txt
#   blenderproc run render_multi_obj.py  # 第二步：渲染
# 可选参数: --num-images/--min-objects/--max-objects/--output-dir/--samples/--resolution

import argparse
import json
import os
import random
import subprocess

import numpy as np

# 环境修复：OpenCV 的 OpenEXR 解码 + EXR 浮点精度（与 bproc_test.py 相同）
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import imageio

_orig_imageio_imread = imageio.v2.imread


def _imageio_imread_exr_float(uri, *args, **kwargs):
    if str(uri).lower().endswith(".exr"):
        img = cv2.imread(str(uri), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read EXR file: {uri}")
        if img.ndim == 2:  # 单通道 EXR（深度/分割）补齐为 3 通道
            img = cv2.merge([img] * 3)
        return img
    return _orig_imageio_imread(uri, *args, **kwargs)


imageio.imread = _imageio_imread_exr_float


def parse_args():
    parser = argparse.ArgumentParser(description="多物体随机组合渲染")
    parser.add_argument("--obj-list", default="obj_list.txt", help="find 生成的 OBJ 列表")
    parser.add_argument("--num-images", type=int, default=3, help="生成图像数量")
    parser.add_argument("--min-objects", type=int, default=1, help="每张图最少物体数")
    parser.add_argument("--max-objects", type=int, default=3, help="每张图最多物体数")
    parser.add_argument("--output-dir", default="output", help="HDF5 输出目录")
    parser.add_argument("--samples", type=int, default=256, help="渲染采样数")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480], help="图像宽高")
    parser.add_argument("--hdri-dir", default="/data/px_data/data_render/HDRI", help="HDRI 目录")
    return parser.parse_args()


def load_obj_list(path: str) -> list:
    """加载 obj 列表；不存在时自动执行 find 命令生成"""
    if not os.path.exists(path):
        print(f"{path} 不存在，自动执行: find test_objs -name '*.obj' > {path}")
        with open(path, "w") as f:
            subprocess.run(["find", "test_objs", "-name", "*.obj"], stdout=f, check=True)
    objs = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    if not objs:
        raise RuntimeError(f"OBJ 列表为空: {path}，请先运行 bash gen_obj_list.sh")
    return objs


def collect_hdri_files(hdri_dir: str) -> list:
    """递归收集 HDRI 文件"""
    files = []
    if os.path.exists(hdri_dir):
        for root, _, fs in os.walk(hdri_dir):
            for f in fs:
                if f.endswith((".hdr", ".exr")):
                    files.append(os.path.join(root, f))
    return files


def normalize_and_place(loaded_objects, category_id, instance_id, scale_range=(0.6, 1.0)):
    """归一化物体尺寸、落地并随机摆放，返回合并后的父物体"""
    merged = bproc.object.merge_objects(loaded_objects, merged_object_name=f"obj_{instance_id}")

    bb_min = np.min([o.get_bound_box().min(axis=0) for o in loaded_objects], axis=0)
    bb_max = np.max([o.get_bound_box().max(axis=0) for o in loaded_objects], axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))
    if diag > 0:
        scale = (1.0 / diag) * np.random.uniform(*scale_range)
        merged.set_scale([scale] * 3)
    else:
        print("Warning: zero bbox, skip scaling")

    for sub in loaded_objects:
        sub.set_cp("category_id", category_id)
        sub.set_cp("instance_id", instance_id)
        # 关键：分割渲染依赖 pass_index 把像素映射到物体，
        # enable_segmentation_output 只会在调用时给已存在的物体编号，
        # 循环里后加载的物体必须手动设置（取值与 instance_id 对应）
        sub.blender_obj.pass_index = instance_id

    # 随机绕 Z 旋转 + 轻微倾斜（先旋转，再按旋转后的包围盒落地）
    merged.set_rotation_euler([
        np.random.uniform(-0.15, 0.15),
        np.random.uniform(-0.15, 0.15),
        np.random.uniform(0, 2 * np.pi),
    ])
    z_min = min(o.get_bound_box()[:, 2].min() for o in loaded_objects)
    merged.set_location([
        np.random.uniform(-0.8, 0.8),
        np.random.uniform(-0.8, 0.8),
        -z_min,
    ])
    return merged


def main():
    args = parse_args()
    if args.min_objects < 1 or args.max_objects < args.min_objects:
        raise ValueError("min-objects 必须 >=1 且 <= max-objects")

    bproc.init()

    # ---------- 1. 加载 OBJ 列表，建立类别映射 ----------
    obj_paths = load_obj_list(args.obj_list)
    unique_objs = sorted(set(obj_paths))
    category_id_of = {p: i + 1 for i, p in enumerate(unique_objs)}   # 0 留给背景
    category_name_of = {p: os.path.splitext(os.path.basename(p))[0] for p in unique_objs}
    print(f"共 {len(obj_paths)} 个 obj 条目 / {len(unique_objs)} 个类别:")
    for p in unique_objs:
        print(f"  [{category_id_of[p]}] {category_name_of[p]} -> {p}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({str(category_id_of[p]): category_name_of[p] for p in unique_objs},
                  f, indent=2, ensure_ascii=False)

    # ---------- 2. 相机 / 渲染输出设置（一次即可，逐场景复用） ----------
    width, height = args.resolution
    bproc.camera.set_resolution(width, height)
    K = np.array([[616, 0, width / 2],
                  [0, 616, height / 2],
                  [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance_id"],
        default_values={"category_id": 0, "instance_id": 0},
    )
    bproc.renderer.set_max_amount_of_samples(args.samples)

    hdri_files = collect_hdri_files(args.hdri_dir)
    print(f"HDRI 文件数: {len(hdri_files)}")

    # ---------- 3. 逐场景渲染 ----------
    for scene_idx in range(args.num_images):
        if scene_idx > 0:
            # 清空上一场景的物体与相机位姿（渲染设置、合成节点保留）
            bproc.object.delete_multiple(scene_objects, remove_all_offspring=True)
            bproc.utility.reset_keyframes()

        # 随机背景
        if hdri_files:
            bproc.world.set_world_background_hdr_img(random.choice(hdri_files))

        # 补充点光源
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([3.0, -3.0, 5.0])
        light.set_energy(300)

        # 随机组合物体（允许重复）
        count = random.randint(args.min_objects, args.max_objects)
        sampled = random.choices(obj_paths, k=count)
        scene_objects = [light]
        print(f"--- 场景 {scene_idx}: {count} 个物体: {[os.path.basename(p) for p in sampled]}")
        for i, path in enumerate(sampled):
            loaded = bproc.loader.load_obj(path)
            if not loaded:
                print(f"跳过无法加载的模型: {path}")
                continue
            merged = normalize_and_place(loaded, category_id_of[path], i + 1)
            scene_objects.append(merged)

        # 相机：绕场景中心随机采样，朝向原点
        radius = np.random.uniform(3.5, 5.5)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0.6, 1.25)
        cam_loc = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ])
        rotation = bproc.camera.rotation_from_forward_vec(-cam_loc)
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, rotation))

        data = bproc.renderer.render()
        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

    print(f"渲染完成! 结果保存在 {args.output_dir}，"
          f"下一步运行: python extract_h5.py {args.output_dir} output_extracted")


if __name__ == "__main__":
    main()
