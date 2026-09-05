import blenderproc as bproc
import numpy as np
import os
import random

# 必须放在最前面：该环境下的 OpenCV 默认禁用 OpenEXR 解码，
# 不打开会导致深度 EXR 读取失败（imgcodecs: OpenEXR codec is disabled）
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 修复深度 EXR 精度丢失：该环境的 imageio->OpenCV 默认把 EXR 按 8 位读取，
# 深度会被截断成 0~255 的整数。这里让 .exr 一律用 IMREAD_ANYDEPTH 读取（float32 米制）。
import cv2
import imageio

_orig_imageio_imread = imageio.v2.imread


def _imageio_imread_exr_float(uri, *args, **kwargs):
    if str(uri).lower().endswith(".exr"):
        img = cv2.imread(str(uri), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read EXR file: {uri}")
        # 单通道 EXR（深度/分割）补齐为 3 通道，兼容 bproc 的读取逻辑
        if img.ndim == 2:
            img = cv2.merge([img] * 3)
        return img
    return _orig_imageio_imread(uri, *args, **kwargs)


imageio.imread = _imageio_imread_exr_float

'''
# 1. 安装 BlenderProc
pip install blenderproc

# 2. 下载资源
blenderproc download cc_textures /path/to/cc_textures/
blenderproc download haven /path/to/hdri/

# 3. 运行脚本
blenderproc run generate_dataset.py
'''

# ============================================================
# 1. 初始化
# ============================================================
bproc.init()

# ============================================================
# 2. 加载 3D 模型 (OBJ)
# ============================================================
obj_path = "./test_objs/obj_000728/mopiluoxingruangao.obj"          # 替换为你的模型路径
loaded_objects = bproc.loader.load_obj(obj_path)

if len(loaded_objects) == 0:
    raise ValueError(f"No objects loaded from {obj_path}")

# 将所有加载的对象合并为一个组合对象（便于统一控制位姿）
# 注意：本版本 BlenderProc 的接口是 merge_objects，返回一个父空物体 (Entity)
obj = bproc.object.merge_objects(loaded_objects, merged_object_name="merged_object")

# 分割掩码的自定义属性必须设置在真正的网格物体上（父空物体不会被渲染）
for sub_obj in loaded_objects:
    sub_obj.set_cp("category_id", 1)        # 类别 ID
    sub_obj.set_cp("instance_id", 1)        # 实例 ID

# 归一化物体尺寸：很多模型的包围盒非常大（本例对角约 99 米），
# 不缩放的话后面相机按半径 3~8 采样会钻进物体内部，导致渲染画面全黑
bb_min = np.min([o.get_bound_box().min(axis=0) for o in loaded_objects], axis=0)
bb_max = np.max([o.get_bound_box().max(axis=0) for o in loaded_objects], axis=0)
obj_diag = float(np.linalg.norm(bb_max - bb_min))
if obj_diag > 0:
    obj.set_scale([1.0 / obj_diag] * 3)
    print(f"Object normalized: bbox diag={obj_diag:.2f} -> scale={1.0 / obj_diag:.4f}")
else:
    print("Warning: object has zero bounding box, skip normalization")

# ============================================================
# 3. 物体表面纹理
# ============================================================
# 默认使用模型自带的纹理（OBJ 的 MTL 会自动加载 jpg，UV 映射正确）。
# 之前直接替换成 CC0 纹理会导致整张贴图被拉伸糊在整个物体表面（物体 UV 只有 [0,1] 一屏），
# 看起来纹理不正确。如需随机替换成 CC0 材质，把下面的开关设为 True。
USE_CC_TEXTURE_ON_OBJECT = False

if USE_CC_TEXTURE_ON_OBJECT:
    texture_dir = "/data/px_data/data_render/cc_textures"          # 替换为你的 CC0 纹理目录
    if os.path.exists(texture_dir):
        # 直接加载 CC0 纹理材质库（自动读取 diff/rough/norm 等 PBR 贴图）
        cc_materials = bproc.loader.load_ccmaterials(texture_dir)
        if cc_materials:
            selected_material = random.choice(cc_materials)
            # CC 纹理是四方连续的材质贴图，需要沿 UV 重复多次才能自然贴合物体表面；
            # 把映射节点的 Scale 调大，让贴图重复铺满，而不是一张拉伸糊在表面
            for node in selected_material.blender_obj.node_tree.nodes:
                if node.type == "MAPPING":
                    node.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
            for sub_obj in loaded_objects:
                sub_obj.replace_materials(selected_material)
            print(f"Applied CC texture material: {selected_material.get_name()}")
        else:
            print("No CC materials loaded, keeping original texture")
    else:
        print(f"Texture directory not found: {texture_dir}, keeping original texture")
else:
    print("Using original model texture (loaded from OBJ/MTL)")

# ============================================================
# 4. 随机加载 HDRI 背景
# ============================================================
# 下载命令（在终端执行）：blenderproc download haven <output_dir>
# 例如：blenderproc download haven /path/to/hdri/

hdri_dir = "/data/px_data/data_render/HDRI"                    # 替换为你的 HDRI 目录
if os.path.exists(hdri_dir):
    # 递归获取所有 HDRI 文件（该目录下文件分散在多层子目录中）
    hdri_files = []
    for root, dirs, files in os.walk(hdri_dir):
        for f in files:
            if f.endswith(('.hdr', '.exr')):
                hdri_files.append(os.path.join(root, f))
    if hdri_files:
        selected_hdri = random.choice(hdri_files)
        
        # 设置 HDRI 环境背景
        bproc.world.set_world_background_hdr_img(selected_hdri)
        print(f"Applied HDRI: {selected_hdri}")
    else:
        print("No HDRI files found, using default background")
else:
    print(f"HDRI directory not found: {hdri_dir}, using default background")

# 添加一盏点光源作为补充照明（HDRI 较弱时保证物体可见）
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([3.0, -3.0, 5.0])
light.set_energy(300)

# 可选：启用透明背景（如需分离渲染背景）
# bproc.renderer.set_output_format(enable_transparency=True)

# ============================================================
# 5. 随机旋转和平移物体
# ============================================================
def sample_object_pose(obj: bproc.types.Entity):
    """随机设置物体的位置和旋转"""
    # 随机平移：在 xyz 各方向上 [-2, 2] 范围内随机
    rand_location = np.random.uniform([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
    obj.set_location(rand_location)
    
    # 随机旋转：使用均匀 SO(3) 采样
    rand_rotation = bproc.sampler.uniformSO3()
    obj.set_rotation_euler(rand_rotation)
    
    return rand_location, rand_rotation

# 执行随机位姿采样
curr_location, curr_rotation = sample_object_pose(obj)
print(f"Object pose: location={curr_location}, rotation={curr_rotation}")

# ============================================================
# 6. 相机设置
# ============================================================
# 设置相机内参
resolution = (640, 480)                    # 图像分辨率
bproc.camera.set_resolution(resolution[0], resolution[1])

# 设置相机内参矩阵（示例：fx=800, fy=800, cx=512, cy=512）
K = np.array([[616, 0, resolution[0] / 2],
              [0, 616, resolution[1] / 2],
              [0, 0, 1]])
bproc.camera.set_intrinsics_from_K_matrix(K, resolution[0], resolution[1])

# 随机采样相机位姿（在物体周围）
def sample_camera_pose(obj: bproc.types.Entity):
    """在物体周围随机采样相机位姿"""
    # 获取物体位置
    obj_location = obj.get_location()
    
    # 在球面上随机采样相机位置（物体已归一化到约 1 米，半径 2.5~5 可让物体在画面中更醒目）
    radius = np.random.uniform(2.5, 5.0)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    
    cam_location = np.array([
        obj_location[0] + radius * np.sin(phi) * np.cos(theta),
        obj_location[1] + radius * np.sin(phi) * np.sin(theta),
        obj_location[2] + radius * np.cos(phi)
    ])
    
    # 计算相机朝向物体的旋转矩阵
    rotation_matrix = bproc.camera.rotation_from_forward_vec(obj_location - cam_location)
    cam_pose = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    
    return cam_pose

# 添加相机位姿（可添加多个，渲染多视角）
cam_pose = sample_camera_pose(obj)
bproc.camera.add_camera_pose(cam_pose)

# ============================================================
# 7. 启用多通道渲染输出
# ============================================================
# 启用深度输出（本版本接口必须传入 activate_antialiasing 参数）
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# 启用分割输出（按类别 ID 和实例 ID）
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance_id"],
                                          default_values={"category_id": 0, "instance_id": 0})

# 可选：启用法线输出
# bproc.renderer.enable_normals_output()

# 设置渲染采样数（较高的采样数可提高 RGB 质量，但会增加渲染时间）
bproc.renderer.set_max_amount_of_samples(256)

# ============================================================
# 8. 执行渲染
# ============================================================
data = bproc.renderer.render()

# ============================================================
# 9. 保存渲染结果到 HDF5 文件
# ============================================================
output_dir = "output"
bproc.writer.write_hdf5(output_dir, data)

print(f"Rendering completed! Results saved to {output_dir}")
