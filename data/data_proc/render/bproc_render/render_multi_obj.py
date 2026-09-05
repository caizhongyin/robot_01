import blenderproc as bproc

# 多物体随机组合渲染：加载 obj_list.txt，每张图随机挑选 1~N 个物体（可重复），
# 生成 RGB / 深度 / 分割 / 边缘点 / 物体框 / 类别名称。
# 每帧把该帧每个物体的 6D 位姿（世界坐标系，单位米）和相机位姿直接写入对应
# 的 .hdf5（object_pose_* / camera_pose_matrices 等键），不单独输出 JSON；
# 解析成 JSON 的工作由 extract_h5.py 完成，便于后续做 6D 位姿估计/重建。
# 用法:
#   bash gen_obj_list.sh                 # 第一步：find 生成 obj_list.txt
#   blenderproc run render_multi_obj.py  # 第二步：渲染
# 可选参数: --num-images/--min-objects/--max-objects/--output-dir/--samples/--resolution/--gpu-id
# GPU 示例: blenderproc run render_multi_obj.py --gpu-id 0

# 深度图中背景像素的远平面值（BlenderProc 惯例），超过该值视为背景
BG_DEPTH_INVALID_THRESHOLD = 1e6

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
    parser.add_argument("--bg-radius", type=float, default=20.0,
                        help="HDRI 环境球半径（米），默认 20.0；背景深度按像素光线到该球面的距离计算")
    parser.add_argument("--baseline", type=float, default=0.05,
                        help="双目基线（米），默认 0.05 即 5cm")
    parser.add_argument("--no-stereo", action="store_true",
                        help="关闭双目立体渲染（默认开启双目）")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=-1,
                        help="Cycles 使用的 GPU 序号（0 起，按 Blender/CUDA 可见设备顺序）；"
                             "默认 -1 表示不指定，按 BlenderProc 默认（通常为 CPU）")
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


def compute_background_depth(depth_img: np.ndarray, cam_pose: np.ndarray, K: np.ndarray,
                             bg_radius: float, threshold: float = BG_DEPTH_INVALID_THRESHOLD) -> np.ndarray:
    """把 HDRI 环境当作半径为 bg_radius 的背景球体（圆心在场景原点），
    对每个背景像素按相机光线与球面的交点计算深度，得到随像素方向平滑变化的背景深度。

    说明：HDRI 文件本身只有颜色通道、没有深度数据，这里用“环境球”假设推导背景深度，
    深度约定与物体 Z-buffer 一致（相机视轴方向的 Z 值）。
    """
    bg = depth_img >= threshold
    if not bg.any():
        return depth_img

    pose = np.array(cam_pose, dtype=float)
    cam_loc = pose[:3, 3]
    rot = pose[:3, :3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    height, width = depth_img.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # 相机坐标中的光线方向（Blender 相机沿局部 -Z 观察）
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, -np.ones_like(u)], axis=-1)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    # 转到世界坐标
    dirs_world = dirs_cam @ rot.T

    # 光线 o + t*d 与球面 |x|^2 = R^2 求交（相机在球内，取正向交点）
    oc_dot_d = dirs_world @ cam_loc  # (H, W)，每个像素的光线方向与相机位置点积
    disc = oc_dot_d ** 2 - (cam_loc @ cam_loc - bg_radius ** 2)
    t = -oc_dot_d + np.sqrt(disc)
    hit_world = cam_loc[None, None, :] + t[..., None] * dirs_world

    # 交点转回相机坐标，取视轴方向 Z（与物体 Z-buffer 一致）
    hit_cam = (hit_world - cam_loc) @ rot
    z_value = -hit_cam[..., 2]

    out = depth_img.copy()
    out[bg] = z_value[bg]
    return out


def collect_object_pose(merged_obj, obj_path: str, category_id: int, instance_id: int) -> dict:
    """读取合并物体在世界坐标系中的 6D 位姿（平移 + 旋转），返回 numpy 数组字典。

    归一化时父物体会带缩放，缩放不改变物体原点与朝向，但为还原实际几何，
    这里把 scale 一并记录（不是 6D 位姿的一部分）。
    """
    return {
        "instance_id": int(instance_id),
        "category_id": int(category_id),
        "obj_path": obj_path,
        "translation": np.asarray(merged_obj.get_location(), dtype=float),          # 3D 平移 [x, y, z] (m)
        "rotation": np.asarray(merged_obj.get_rotation_mat(), dtype=float),         # 3x3 旋转矩阵（世界系）
        "euler_xyz": np.asarray(merged_obj.get_rotation_euler(), dtype=float),      # 弧度，Blender XYZ 顺序
        "scale": np.asarray(merged_obj.get_scale(), dtype=float),                   # 归一化缩放
    }


def _records_to_array(records: list, key: str, shape_tail: tuple) -> np.ndarray:
    """把每条位姿记录中的某个向量/矩阵字段堆叠成 (N, *shape_tail) 数组"""
    if not records:
        return np.empty((0, *shape_tail), dtype=float)
    return np.stack([rec[key] for rec in records])


def attach_pose_datasets(data: dict, pose_records: list, camera_pose_mats: list,
                         camera_names: list, intrinsics: np.ndarray) -> None:
    """把本帧物体的 6D 位姿与相机位姿作为 HDF5 数据集附加到待写入的 data 字典。

    BlenderProc 的 write_hdf5 会把 data 中每个键都写成一个 HDF5 数据集，
    因此这里每个键的值都包成单帧的 list。物体位姿以纯数值数组存放，
    模型路径 / 相机名等字符串以 UTF-8 编码的 JSON 字节串存放。
    """
    data["object_pose_instance_ids"] = [
        np.array([rec["instance_id"] for rec in pose_records], dtype=np.int32)]
    data["object_pose_category_ids"] = [
        np.array([rec["category_id"] for rec in pose_records], dtype=np.int32)]
    data["object_pose_translations"] = [_records_to_array(pose_records, "translation", (3,))]
    data["object_pose_rotations"] = [_records_to_array(pose_records, "rotation", (3, 3))]
    data["object_pose_eulers"] = [_records_to_array(pose_records, "euler_xyz", (3,))]
    data["object_pose_scales"] = [_records_to_array(pose_records, "scale", (3,))]
    data["object_pose_paths"] = [
        np.bytes_(json.dumps([rec["obj_path"] for rec in pose_records], ensure_ascii=False).encode("utf-8"))]

    data["camera_pose_matrices"] = [np.stack(camera_pose_mats)] if camera_pose_mats else [np.empty((0, 4, 4))]
    data["camera_names"] = [np.bytes_(json.dumps(camera_names).encode("utf-8"))]
    data["camera_intrinsics"] = [np.asarray(intrinsics, dtype=float)]


def main():
    args = parse_args()
    if args.min_objects < 1 or args.max_objects < args.min_objects:
        raise ValueError("min-objects 必须 >=1 且 <= max-objects")

    width, height = args.resolution
    bproc.init()
    if args.gpu_id >= 0:
        print(f"指定渲染 GPU 序号: {args.gpu_id}")
        bproc.renderer.set_render_devices(desired_gpu_ids=[args.gpu_id])
    stereo_enabled = not args.no_stereo
    if stereo_enabled:
        bproc.renderer.toggle_stereo(True)
        bproc.camera.set_stereo_parameters(convergence_mode="PARALLEL",
                                           convergence_distance=0.00001,
                                           interocular_distance=args.baseline)
        print(f"双目立体渲染已启用: 基线 {args.baseline * 100:.1f} cm, 分辨率 {width}x{height}")

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
        pose_records = []
        for i, path in enumerate(sampled):
            loaded = bproc.loader.load_obj(path)
            if not loaded:
                print(f"跳过无法加载的模型: {path}")
                continue
            merged = normalize_and_place(loaded, category_id_of[path], i + 1)
            scene_objects.append(merged)
            pose_records.append(collect_object_pose(
                merged, path, category_id_of[path], i + 1))

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
        cam_pose = bproc.math.build_transformation_mat(cam_loc, rotation)
        bproc.camera.add_camera_pose(cam_pose)

        # 平行双目的左右目位姿：沿相机局部 X 轴左右各偏移基线一半
        left_pose = np.array(cam_pose, dtype=float).copy()
        right_pose = np.array(cam_pose, dtype=float).copy()
        left_pose[:3, 3] = cam_loc + rotation @ np.array([-args.baseline / 2, 0, 0])
        right_pose[:3, 3] = cam_loc + rotation @ np.array([args.baseline / 2, 0, 0])

        # 本帧需要写入 HDF5 的相机位姿/名称（物体位姿在渲染后统一附加）
        camera_pose_mats = ([left_pose, right_pose] if stereo_enabled else [cam_pose])
        camera_names = ["left", "right"] if stereo_enabled else ["mono"]

        data = bproc.renderer.render()

        # 深度后处理：HDRI 环境球背景深度（逐像素，随方向平滑变化）
        depth_frames = data["depth"]
        for frame_idx, depth_img in enumerate(depth_frames):
            if stereo_enabled:
                # 双目：depth 形状为 (2, H, W)，[0]=左目 [1]=右目
                bg_count_l = int((depth_img[0] >= BG_DEPTH_INVALID_THRESHOLD).sum())
                bg_count_r = int((depth_img[1] >= BG_DEPTH_INVALID_THRESHOLD).sum())
                depth_img[0] = compute_background_depth(depth_img[0], left_pose, K, args.bg_radius)
                depth_img[1] = compute_background_depth(depth_img[1], right_pose, K, args.bg_radius)
                print(f"  场景{scene_idx} 双目背景深度已计算: 环境球半径 {args.bg_radius:.1f} m "
                      f"(左目背景 {bg_count_l}, 右目背景 {bg_count_r}/{depth_img[0].size})")
            else:
                bg_count = int((depth_img >= BG_DEPTH_INVALID_THRESHOLD).sum())
                depth_frames[frame_idx] = compute_background_depth(depth_img, cam_pose, K, args.bg_radius)
                print(f"  场景{scene_idx} 背景深度已计算: 环境球半径 {args.bg_radius:.1f} m "
                      f"(背景像素 {bg_count}/{depth_img.size})")

        # 把本帧物体的 6D 位姿与相机位姿写入同一个 HDF5（不作为独立 JSON 输出）
        attach_pose_datasets(data, pose_records, camera_pose_mats, camera_names, K)

        # 注意：不使用 stereo_separate_keys，否则 instance_attribute_maps 会被按元素错误拆分；
        # 双目图像以 [2, H, W, ...] 保存，由 extract_h5.py 按左/右目拆分
        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)

    print(f"渲染完成! 结果保存在 {args.output_dir}，"
          f"下一步运行: python extract_h5.py {args.output_dir} output_extracted")


if __name__ == "__main__":
    main()
