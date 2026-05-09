import torch
import torch.distributed as dist
import numpy as np
import cv2
import warnings
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree
import torch.nn.functional as F
# 自定义模块导入
from utils.common_utils import visualize_tensor
from utils.kitti_utils import KITTI_post_process, depth_completion
from utils.common_utils import load_flist, img2label,crop_valid_regions,process_kitti_tensors, resize_if_needed
from utils import misc
# ======================== 全局常量 & 分布式兼容设备配置 ========================

def gaussian_blur_2d(x, kernel_size=5, sigma=1.5):
    """
    修复版高斯模糊：padding='same' 强制输入输出H×W一致，支持任意奇数kernel_size
    输入：x [B, C, H, W]，任意尺寸特征图
    输出：x_blur [B, C, H, W]，尺寸与输入完全一致
    """
    if kernel_size == 7:
        # 7x7高斯核（归一化，保证亮度不变）
        kernel = torch.tensor([[[[1, 6, 15, 20, 15, 6, 1],
                                 [6, 36, 90, 120, 90, 36, 6],
                                 [15, 90, 225, 300, 225, 90, 15],
                                 [20, 120, 300, 400, 300, 120, 20],
                                 [15, 90, 225, 300, 225, 90, 15],
                                 [6, 36, 90, 120, 90, 36, 6],
                                 [1, 6, 15, 20, 15, 6, 1]]]],
                              dtype=x.dtype, device=x.device) / 4096.0
    elif kernel_size == 5:
        # 5x5高斯核（原核，保留）
        kernel = torch.tensor([[[[1,4,6,4,1],
                                 [4,16,24,16,4],
                                 [6,24,36,24,6],
                                 [4,16,24,16,4],
                                 [1,4,6,4,1]]]],
                              dtype=x.dtype, device=x.device) / 256.0
    else:
        # 3x3高斯核（默认，轻量模糊）
        kernel = torch.tensor([[[[1,2,1],[2,4,2],[1,2,1]]]],
                              dtype=x.dtype, device=x.device) / 16.0
    # 核心修复：padding='same' 强制输入输出H×W一致，替代手动计算padding
    return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1),
                    padding='same', groups=x.shape[1])

def underwater_generate(image, depth, mask=None):
    """
    生成水下图像效果（分布式适配+真实度优化+噪声增强+随机模糊开关，复用自定义高斯模糊）
    Args:
        image: 输入图像 (B,3,H,W)，torch.Tensor，取值[0,1]
        depth: 深度图 (B,1,H,W)，torch.Tensor
        mask: 掩码（可选）(B,H,W)，1表示保留原图，0表示生成水下效果
    Returns:
        水下效果图像 (B,3,H,W)，torch.float32，取值[0,1]
    """
    # 1. 动态获取设备
    device = image.device
    img_num = image.shape[0]
    # 2. 生成水质参数eta（优化后，覆盖清水→高浊）
    eta = choose_eta(img_num, device=device)
    # 3. 深度值裁剪（优化范围：0.3~8，贴合真实水下深度）
    depth_max = torch.empty((img_num, 1, 1, 1), device=device).uniform_(2, 8)
    depth_min = torch.tensor(0.3, device=device)
    depth = torch.clamp(depth, min=depth_min, max=depth_max)
    # 4. 透射率计算
    t = torch.exp(-(depth) * eta)
    # 5. 光照生成（优化后，更自然的光照衰减+随机亮度）
    L_t = generate_lighting(image, depth, eta, device=device)
    # 6. 核心水下成像公式（红通道轻微衰减，贴合真实色偏）
    color_bias = torch.rand(img_num, 3, 1, 1, device=device) * 0.2 + 0.9
    color_bias[:, 0:1, :, :] = color_bias[:, 0:1, :, :] * 0.95
    I = (image * t * L_t * color_bias) + (L_t * (1 - t))
    I = torch.clamp(I, 0.0, 1.0)
    # 7. 掩码应用（正确逻辑：mask=1保留原图，mask=0生成水下效果）
    if mask is not None:
        mask = mask.float().unsqueeze(1)
        I = mask * image + (1 - mask) * I
    # 8. 加入轻量水下噪声（高斯+悬浮颗粒，匹配真实数据）
    # ---------------------- 复用你的gaussian_blur_2d：随机模糊开关 ----------------------
    blur_prob = 0.1  # 40%概率生成模糊，兼顾去模糊训练与样本多样性
    if torch.rand(1, device=device) < blur_prob:
        B, C, H, W = I.shape
        gauss_noise = torch.randn_like(I) * 0.005
        particle_noise = (torch.rand(B, 1, H, W, device=device) < 0.001).float()
        particle_noise = particle_noise * torch.rand(B, 1, H, W, device=device) * 0.1
        particle_noise = particle_noise.repeat(1, C, 1, 1)
        I = I + gauss_noise + particle_noise
        I = torch.clamp(I, 0.0, 1.0)

        # 中等模糊适配：3核（轻微）/5核（中等）随机，不使用7核（避免过模糊）
        kernel_size = 3 if torch.rand(1) < 0.5 else 5
        I = gaussian_blur_2d(I, kernel_size=kernel_size)  # 直接调用你的模糊函数
        I = torch.clamp(I, 0.0, 1.0)  # 裁剪防止数值溢出
    # -----------------------------------------------------------------------------------

    return I.float()


def generate_lighting(
    image,
    depth,
    eta,
    sigma=10,
    art_light_prob=0.5,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    生成水下光照效果（优化版：参数范围调整+亮度随机偏移+修复逻辑）
    参数:
        image: 输入图像 (B,C,H,W)
        depth: 深度图 (B,1,H,W)
        eta: 水质参数 (B,3,1,1)
        sigma: 光源扩散系数（基础值）
        art_light_prob: 人工光源使用概率
        device: 计算设备 (CPU/GPU)
    """
    B, C, H, W = image.shape
    # 1. 水体深度（优化范围：0.2~5，避免衰减过度）核心修改
    water_depth = torch.empty((B, 1, 1, 1), device=device).uniform_(0.2, 5)
    # 2. 基础光照衰减（自然光照）
    L_t1 = torch.exp(-depth * eta)  # 场景深度衰减
    L_t2 = torch.exp(-water_depth * eta)  # 水体整体衰减
    # 3. 生成坐标网格（简化代码，无需重复to(device)）
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device),
                          indexing='ij')
    x = x.float().view(1, 1, H, W)
    y = y.float().view(1, 1, H, W)
    # 4. 随机参数生成（优化范围，提升多样性）核心修改
    scale = 0.3 + 2.7 * torch.rand(B, 1, 1, 1).to(device)  # 0.3~3.0（原0.5~2.5）
    sigma = torch.rand(B, 1, 1, 1).to(device) * H * scale
    z_l = torch.randn(B, 1, 1, 1).to(device)
    r_l = torch.rand(B, 1, 1, 1).to(device)
    # 5. 人工光源中心坐标
    x_c = torch.randint(0, W, (B, 1, 1, 1), device=device).float()
    y_c = torch.randint(0, H, (B, 1, 1, 1), device=device).float()
    # 6. 人工光源强度（优化范围：0.5~0.95，避免过曝）核心修改
    L_art = torch.rand(B, 1, 1, 1, device=device).clamp(0.5, 0.95)
    # 7. 人工光源计算（修复后，有空间衰减）
    L_t3, _ = _generate_artificial_light(
        x, y, eta, depth, sigma, x_c, y_c, L_art, z_l, r_l, device=device
    )
    # 8. 光照权重计算（向量化，保持原有逻辑）
    use_art = torch.rand(B, 1, 1, 1).to(device) < art_light_prob
    w_all = torch.rand(B, 3, device=device)
    w_all[~use_art.squeeze(-1).squeeze(-1).squeeze(-1), 2] = 0  # 关闭人工光源则权重置0
    w_sum = w_all.sum(dim=1, keepdim=True).clamp(min=1e-8)  # 避免除0
    w_norm = w_all / w_sum
    weights = w_norm.unsqueeze(2).unsqueeze(3)
    # 9. 最终光照（加入随机亮度偏移，模拟光照不均匀）核心新增
    final_light = (weights[:, 0:1] * L_t1 + weights[:, 1:2] * L_t2 + weights[:, 2:3] * L_t3)
    light_brightness = torch.rand(B, 1, 1, 1, device=device) * 0.4 + 0.8  # 0.8~1.2倍亮度
    final_light = final_light * light_brightness
    final_light = torch.clamp(final_light, 0.0, 1.0)  # 防止过曝
    return final_light


def _generate_artificial_light(x, y, eta, depth, sigma, x_c, y_c, L_art, Z_l, r_l, device=None):
    """
    生成人工光源效果（修复版：空间衰减+快速高斯衰减，避免过曝）
    参数:
        x, y: 网格坐标 (1,1,H,W)
        eta: 水质参数 (B,3,1,1)
        depth: 深度图 (B,1,H,W)
        sigma: 光源扩散系数 (B,1,1,1)
        x_c, y_c: 光源中心坐标 (B,1,1,1)
        L_art: 光源强度 (B,1,1,1)
        Z_l: 光源高度参数 (B,1,1,1)
        r_l: 散射参数 (B,1,1,1)
    返回:
        light: 光照强度 (B,3,H,W)
        v: 光源衰减系数 (B,3,H,W)
    """
    # 1. 光源距离计算（修复核心：加入平面距离，实现空间衰减）核心修改
    D = torch.sqrt(Z_l ** 2 + (x - x_c) ** 2 + (y - y_c) ** 2)
    # 2. 高斯衰减（优化：系数从1→2，加快衰减，避免光源中心过曝）核心修改
    sigma_sq = torch.clamp(sigma ** 2, min=1e-8)  # 避免除0
    v = L_art * torch.exp(-2.0 / (2 * sigma_sq) * ((x - x_c) ** 2 + (y - y_c) ** 2))
    # 3. 综合衰减（水质+深度+光源距离）
    art_light = v * torch.exp(-(D + depth) * eta)
    art_light = torch.clamp(art_light, 0.0, 1.0)  # 裁剪到合理范围
    return art_light, v


def choose_eta(num_pic, device: torch.device = None):
    """
    生成混合的eta参数（优化版：新增5组水质，覆盖清水→高浊，全程GPU执行）
    Args:
        num_pic: 水下图像数量
        device: 目标设备（如cuda:0/cuda:1，分布式场景由调用方传递）
    Returns:
        mixed: (num_pic, 3, 1, 1)，与device同设备，对应RGB三通道的衰减系数
    """
    # 兜底：未指定device时，用当前默认设备（兼容单卡/分布式）
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 预定义参数：原有18组 + 新增5组（清水→高浊梯度，贴合真实水下场景）核心修改
    params = torch.tensor([
        # 原有参数（保留不变）
        [0.30420412, 0.11727661, 0.1488851],
        [0.30474395, 0.05999663, 0.30099538],
        [0.35592191, 0.11227639, 0.38412464],
        [0.32493874, 0.15305673, 0.25060999],
        [0.216913, 0.03978087, 0.01816397],
        [0.21815601, 0.04604394, 0.02531781],
        [0.18632958, 0.05129329, 0.03252319],
        [0.22314355, 0.07796154, 0.0618754],
        [0.55091001, 0.14385827, 0.01387215],
        [0.42493874, 0.12305673, 0.055799195],
        [0.55874165, 0.0518615, 0.0591001],
        [0.13039252, 0.18667714, 0.5539252],
        [0.10760831, 0.1567016, 0.60103],
        [0.15963731, 0.205724217, 0.733602],
        [0.4780358, 0.49429632, 0.69314718],
        [0.597837, 0.77652879, 1.23787436],
        [0.20760831, 0.21467016, 0.30103],
        [0.25963731, 0.33724217, 0.537602],
        # 新增5组核心参数（清水→高浊，覆盖真实水下主要场景）
        [0.12039252, 0.08667714, 0.0539252],  # 超清水（近水面，衰减极弱）
        [0.28039252, 0.16667714, 0.1239252],  # 轻度浊水（常见近岸/浅水区）
        [0.65091001, 0.24385827, 0.11387215],  # 中度浊水（水下1-3米，红通道明显衰减）
        [0.85091001, 0.34385827, 0.21387215],  # 高度浊水（水下3米以上，红通道严重衰减）
        [0.33039252, 0.28667714, 0.4539252],  # 高散射浊水（悬浮颗粒多，蓝绿色偏严重）
    ], dtype=torch.float32, device=device)  # 直接在目标GPU创建，无数据迁移
    # 随机索引+混合权重（全程GPU执行，无CPU→GPU传输）
    idx = torch.randint(0, len(params), (num_pic, 2), device=device)
    weights = torch.rand(num_pic, 1, device=device)
    # 混合计算（线性插值，生成多样化水质）
    mixed = weights * params[idx[:, 0]] + (1 - weights) * params[idx[:, 1]]
    mixed = mixed.view(num_pic, 3, 1, 1)  # 适配维度：(B,3,1,1)
    return mixed
# 注：深度补全基于OpenCV/numpy（CPU操作），分布式训练中每个进程独立处理，无需设备修改

class kitti_under(object):
    def __init__(self, calib_paths, image_2_paths, label_2_paths, velo_paths):
        self.image_2_paths = load_flist(image_2_paths)
        self.calib_paths = img2label(self.image_2_paths, calib_paths, suffix='.txt')
        self.label_2_paths = img2label(self.image_2_paths, label_2_paths, suffix='.txt')
        self.velo_paths = img2label(self.image_2_paths, velo_paths, suffix='.bin')
        self.Kitti_post_process = KITTI_post_process(self.calib_paths, self.image_2_paths, self.label_2_paths, self.velo_paths)
    def kitti_data(self, frame):
        # ================================prepare img, depth=====================================================
        if frame == None:
            frame = 0

        img_ori, lidar_depth2img, depth, _ = self.Kitti_post_process.velo2pixel_and_velo2depth(frame, clip_distance=2.0,
                                                                                               plt_img=False)
        #depth = sparse_depth_completion(img_ori,depth)
        completion = KITTIDepthCompleter(max_depth=70,sky_depth=50)
        depth = completion.nearest_neighbor_completion(depth)
        depth = completion.nearest_neighbor_completion(depth)
        depth = completion.morphological_completion(depth)
        depth = depth_completion(depth, plt_img=False)
        # ====================================================================

        return img_ori, depth

class KITTIDepthCompleter:
    """KITTI深度图智能补全器"""

    def __init__(self, max_depth=80.0, sky_depth=60.0):
        self.max_depth = max_depth
        self.sky_depth = sky_depth

    def load_depth_map(self, depth_path):
        """加载深度图"""
        depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            raise ValueError(f"无法加载深度图: {depth_path}")
        return depth_map.astype(np.float32)

    def load_rgb_image(self, rgb_path):
        """加载RGB图像"""
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            raise ValueError(f"无法加载RGB图像: {rgb_path}")
        return rgb_image

    def create_valid_mask(self, depth_map):
        """创建有效像素掩码"""
        return (depth_map > 0) & (depth_map < self.max_depth)

    def detect_sky_region(self, depth_map, rgb_image=None):
        """检测天空区域"""
        height, width = depth_map.shape

        # 方法1: 基于深度为零和顶部位置
        zero_mask = (depth_map == 0)
        top_region = np.zeros_like(depth_map, dtype=bool)
        top_region[:int(height * 0.3), :] = True
        basic_sky_mask = zero_mask & top_region

        # 方法2: 结合RGB颜色信息
        if rgb_image is not None:
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

            # 蓝色天空检测 (HSV空间)
            blue_sky_mask = ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 140) &
                             (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 150))

            # 明亮区域检测
            bright_mask = (cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) > 200)

            # 结合多种特征
            enhanced_sky_mask =( basic_sky_mask & (blue_sky_mask | bright_mask))

            return enhanced_sky_mask

        return basic_sky_mask

    def nearest_neighbor_completion(self, depth_map, max_distance=25):
        """最近邻补全 - 修复版本"""
        valid_mask = self.create_valid_mask(depth_map)
        completed = depth_map.copy()

        # 获取有效像素坐标
        valid_coords = np.column_stack(np.where(valid_mask))
        valid_values = depth_map[valid_mask]

        if len(valid_coords) == 0:
            return completed

        # 构建KD树
        tree = cKDTree(valid_coords)

        # 获取无效像素坐标
        invalid_coords = np.column_stack(np.where(~valid_mask))

        if len(invalid_coords) > 0:
            distances, indices = tree.query(invalid_coords, distance_upper_bound=max_distance)
            for i, (y, x) in enumerate(invalid_coords):
                if distances[i] < max_distance:
                    completed[y, x] = valid_values[indices[i]]

        return completed

    def morphological_completion(self, depth_map, kernel_size=5, iterations=3):
        """形态学补全"""
        valid_mask = self.create_valid_mask(depth_map)
        completed = depth_map.copy()

        # 转换为uint8用于形态学操作
        depth_uint8 = (depth_map / self.max_depth * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for _ in range(iterations):
            dilated = cv2.dilate(depth_uint8, kernel)
            completed_uint8 = depth_uint8.copy()
            completed_uint8[~valid_mask] = dilated[~valid_mask]
            valid_mask = self.create_valid_mask(completed)

        # 转换回原始范围
        return (completed_uint8 / 255.0 * self.max_depth).astype(np.float32)

    def bilateral_filter_completion(self, depth_map, d=7, sigma_color=80, sigma_space=80):
        """双边滤波补全"""
        valid_mask = self.create_valid_mask(depth_map)

        # 确保数据类型为float32
        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)

        # 归一化处理
        depth_normalized = depth_map / self.max_depth
        depth_normalized[~valid_mask] = 0

        # 应用双边滤波
        try:
            filtered = cv2.bilateralFilter(depth_normalized, d, sigma_color, sigma_space)
        except cv2.error:
            # 备选方案: 高斯滤波
            filtered = cv2.GaussianBlur(depth_normalized, (d, d), 0)

        return filtered * self.max_depth

    def guided_filter_completion(self, depth_map, rgb_guide, radius=15, eps=0.01):
        """引导滤波补全"""
        valid_mask = self.create_valid_mask(depth_map)

        # 转换为灰度图作为引导
        if len(rgb_guide.shape) == 3:
            guide_gray = cv2.cvtColor(rgb_guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 归一化深度图
        depth_normalized = depth_map / self.max_depth
        depth_normalized[~valid_mask] = 0

        # 应用引导滤波
        try:
            filtered = cv2.ximgproc.guidedFilter(
                guide_gray, depth_normalized, radius, eps
            )
        except:
            # 如果引导滤波不可用，使用双边滤波
            filtered = cv2.bilateralFilter(depth_normalized, d=5, sigmaColor=75, sigmaSpace=75)

        return filtered * self.max_depth

    def adaptive_sky_completion(self, depth_map, rgb_image=None):
        """自适应天空补全"""
        height, width = depth_map.shape

        # 检测天空区域
        sky_mask = self.detect_sky_region(depth_map, rgb_image)

        if np.sum(sky_mask) == 0:
            return depth_map

        completed = depth_map.copy()

        # 创建渐变深度图 - 从近到远
        y_coords = np.arange(height)
        gradient_values = np.linspace(self.sky_depth, self.max_depth, height)

        # 应用渐变天空深度
        for y in range(height):
            if np.any(sky_mask[y, :]):
                # 根据y坐标设置深度值
                depth_value = gradient_values[y]
                completed[y, sky_mask[y, :]] = depth_value

        return completed

    def hybrid_completion(self, depth_map, rgb_image=None):
        """
        混合补全方法 - 核心算法
        结合多种传统方法实现最优补全效果
        """
        print("步骤1: 最近邻补全...")
        step1 = self.nearest_neighbor_completion(depth_map)

        print("步骤2: 形态学补全...")
        step2 = self.morphological_completion(step1)

        print("步骤3: 天空区域补全...")
        step3 = self.adaptive_sky_completion(step2, rgb_image)

        if rgb_image is not None:
            print("步骤4: 引导滤波补全...")
            step4 = self.guided_filter_completion(step3, rgb_image)
        else:
            print("步骤4: 双边滤波补全...")
            step4 = self.bilateral_filter_completion(step3)

        print("步骤5: 最终优化...")

# ========== 提取公共工具函数（无self依赖） ==========
def repeat_tensor_by_idx(tensor: torch.Tensor, idx: torch.Tensor, repeat_times: int = 2) -> torch.Tensor:
    """
    按索引重复张量（替代多次torch.cat，GPU高效）
    Args:
        tensor: 输入张量 [BS, C, H, W]
        idx: 随机索引 [1,]
        repeat_times: 重复次数（默认2次）
    Returns:
        重复后的张量 [repeat_times, C, H, W]
    """
    repeat_idx = torch.cat([idx] * repeat_times, dim=0)
    return tensor[repeat_idx]

# ========== 修改process_tensor_pipeline函数，接收外部随机结果 ==========
def process_tensor_pipeline(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    depth_tensor: torch.Tensor = None,
    is_underwater: bool = False,
    target_ratio: float = 4/3,
    max_size: int = 580,
    device: torch.device = None,
    do_adjust_ratio: bool = None,  # 外部传入的是否调整比例
    selected_method: str = None    # 外部传入的选中方法
) -> torch.Tensor:
    """
    统一张量处理流水线：水下生成→裁剪→比例调整→缩放
    """
    # 1. 生成水下图像（仅当需要时执行）
    if not is_underwater and depth_tensor is not None:
        tensor = underwater_generate(tensor, depth_tensor, mask)
    # 2. 裁剪有效区域
    tensor = crop_valid_regions(tensor, mask)
    # 3. 用外部传入的随机结果调整宽高比（不再内部生成随机数）
    if do_adjust_ratio:
        tensor, _ = process_kitti_tensors(tensor, tensor, target_ratio=target_ratio, method=selected_method)
    # 4. 尺寸限制
    tensor = resize_if_needed([tensor], max_size=max_size)[0]
    return tensor


def build_nested_tensor(
        tensor: torch.Tensor,
        mask: torch.Tensor = None,
        device: torch.device = None,
        slice_first: bool = True
) -> misc.NestedTensor:
    """
    修复后：
    1. 动态构建tensor_list，兼容单batch（size=1）和多batch（size≥2）
    2. 移除强行构造长度2列表的冗余逻辑
    3. 保留mask的同步处理，确保NestedTensor完整性
    4. 兼容原slice_first逻辑，不破坏现有调用
    """
    # 核心修复：动态生成tensor_list，有多少样本就构建多长的列表
    # tensor.shape = [B, C, H, W]，B为batch_size（1或≥2）
    tensor_list = [tensor[i] for i in range(tensor.size(0))]

    # 构建NestedTensor（原生支持任意长度的tensor_list）
    nt = misc.nested_tensor_from_tensor_list(tensor_list)

    # 可选：仅保留第一个样本（兼容原slice_first逻辑）
    if slice_first:
        nt.tensors = nt.tensors[:1]  # 无论原batch多大，最终只保留第一个样本
        # 同步处理mask（如果存在）
        if mask is not None and hasattr(nt, 'mask'):
            # 先确保mask和tensor的batch_size一致
            if mask.size(0) != tensor.size(0):
                mask = mask[:tensor.size(0)]  # 截断mask到tensor的batch_size
            nt.mask = nt.mask[:1]

    # 设备迁移（仅当不匹配时）
    if device is not None and nt.tensors.device != device:
        nt = nt.to(device, non_blocking=True)

    return nt


def generate_train_data(
        clean,
        depth,
        under,
        underE,
        phase,
        device,  # 显式指定设备，默认GPU
        max_resize_size: int = 580,  # 可配置参数
        target_ratio: float = 4 / 3,  # 可配置参数
        # 新增：阶段数据配比超参（可外部调整）
        phase2_paired_ratio: float = 0.8,  # 2阶段配对数据占比70%
        phase3_sim_real_ratio: float = 0.5,  # 3阶段仿真:真实=4:6
        phase3_sim_paired_ratio: float = 0.4,  # 3阶段仿真数据中配对占比20%
        phase3_real_paired_ratio: float = 0.1,  # 3阶段真实数据中配对占比10%
        # Phase4新增配置：可外部指定用仿真水下/真实水下，默认用真实水下（更贴合预测场景）
        phase4_use_real_under: bool = True,
        # Phase4是否对仿真水下做预处理（与训练阶段保持一致，默认开启）
        phase4_sim_preprocess: bool = True
):
    """
    完善后策略（新增Phase4测试/预测阶段）：
    1. Phase1：纯仿真配对（基础能力）
    2. Phase2：纯仿真（配对70%+非配对30%）→ 过渡训练relight，避免过拟合
    3. Phase3：仿真(40%)+真实(60%) → 仿真（配对20%+非配对80%）、真实（配对10%+非配对90%）
    4. Phase4：纯测试/预测阶段 → train_data=水下图像（仿真/真实可选），high_quality=干净参考图像
    新增：为train_data添加is_real_data属性，标记是否为真实数据，适配各阶段场景
    """
    phase_use = phase
    # 获取仿真清洁样本总数（全局可用，Phase2核心使用）
    n_clean_samples = clean.tensors.size(0)
    # ========== Phase 1: 纯仿真配对数据（基础训练） ==========
    if phase == 1:
        # 1. 随机索引+张量重复
        idx = torch.randint(0, clean.tensors.size(0), (1,), device=device)
        clean_rep = repeat_tensor_by_idx(clean.tensors, idx)
        depth_rep = repeat_tensor_by_idx(depth.tensors, idx)
        mask_rep = repeat_tensor_by_idx(clean.mask, idx)

        # 2. 提前生成随机结果（确保两次process_tensor_pipeline用同一个）
        do_adjust_ratio = (torch.rand(1, device=device) < 0.5).item()
        selected_method = None
        if do_adjust_ratio:
            methods = ['crop', 'compress']
            method_idx = torch.randint(0, len(methods), (1,), device=device).item()
            selected_method = methods[method_idx]

        # 3. 统一处理流水线
        simulate_under = process_tensor_pipeline(
            clean_rep, mask_rep, depth_rep, is_underwater=False,
            target_ratio=target_ratio, max_size=max_resize_size, device=device,
            do_adjust_ratio=do_adjust_ratio, selected_method=selected_method
        )
        clean_simulate = process_tensor_pipeline(
            clean_rep, mask_rep, is_underwater=True,
            target_ratio=target_ratio, max_size=max_resize_size, device=device,
            do_adjust_ratio=do_adjust_ratio, selected_method=selected_method
        )

        # 构建NestedTensor
        train_data = build_nested_tensor(simulate_under, mask_rep, device, slice_first=False)
        refer = build_nested_tensor(clean_simulate, mask_rep, device)
        high_quality = build_nested_tensor(clean_simulate, mask_rep, device)
        train_data.has_gt = True
        train_data.is_paired = True
        # 新增：is_real_data 标记（Phase1纯仿真→False）
        train_data.is_real_data = False

    # ========== Phase 2: 纯仿真（配对+少量非配对）→ 过渡训练relight ==========
    elif phase == 2:
        # 边界校验：确保仿真清洁样本数≥1，提前暴露问题
        assert n_clean_samples >= 1, \
            f"Phase 2训练需要至少1个仿真清洁样本，当前仅{n_clean_samples}个"
        # 核心：70%配对（训练relight）+ 30%非配对（避免过拟合）
        is_paired = torch.rand(1, device=device) < phase2_paired_ratio

        # 【核心修改1】生成单样本索引，切片获取batch-size=1的基础张量（替代repeat_tensor_by_idx）
        idx1 = torch.randint(0, n_clean_samples, (1,), device=device).item()  # 主样本索引
        clean_simulate = clean.tensors[idx1:idx1 + 1, ...]  # [1, C, H, W] 固定batch=1
        depth_simulate = depth.tensors[idx1:idx1 + 1, ...]  # [1, C, H, W] 固定batch=1
        mask_rep = clean.mask[idx1:idx1 + 1, ...]  # [1, H, W]  固定batch=1（mask维度）

        # 2. 统一处理流水线（输入已为batch=1，输出保持batch=1）
        simulate_under = process_tensor_pipeline(
            clean_simulate, mask_rep, depth_simulate, is_underwater=False,
            target_ratio=target_ratio, max_size=max_resize_size, device=device
        )
        clean_simulate = process_tensor_pipeline(
            clean_simulate, mask_rep, is_underwater=True,
            target_ratio=target_ratio, max_size=max_resize_size, device=device
        )

        # 3. 构建训练数据（train_data固定为仿真水下图，batch=1）
        train_data = build_nested_tensor(simulate_under, mask_rep, device, slice_first=False)

        # 4. 配对/非配对分支（均保证batch=1）
        if is_paired:
            # 配对：参考+高质量都用当前样本（训练relight），batch=1
            refer = build_nested_tensor(clean_simulate, mask_rep, device)
            high_quality = build_nested_tensor(clean_simulate, mask_rep, device)
        else:
            # 非配对：参考用当前样本，高质量用随机样本（缓解过拟合），均为batch=1
            refer = build_nested_tensor(clean_simulate, mask_rep, device)
            # 【核心修改2】非配对样本也切片为batch=1，避免维度不一致
            idx2 = idx1
            while idx2 == idx1:  # 确保非配对时索引不同
                idx2 = torch.randint(0, n_clean_samples, (1,), device=device).item()
            unpaired_clean = clean.tensors[idx2:idx2 + 1, ...]  # [1, C, H, W] 固定batch=1
            unpaired_mask = clean.mask[idx2:idx2 + 1, ...]  # [1, H, W]  固定batch=1
            # 预处理保持原逻辑，输入为batch=1，输出也为batch=1
            unpaired_clean = crop_valid_regions(unpaired_clean, unpaired_mask)
            unpaired_clean = resize_if_needed([unpaired_clean], max_size=max_resize_size)[0]
            high_quality = build_nested_tensor(unpaired_clean, unpaired_mask, device)
            phase_use = 3  # 非配对时提前适配phase3逻辑

        train_data.has_gt = True
        train_data.is_paired = is_paired
        # 新增：is_real_data 标记（Phase2纯仿真→False，与is_paired解耦）
        train_data.is_real_data = False

    # ========== Phase 3: 仿真+真实混合数据（核心训练） ==========
    elif phase == 3:
        # 第一步：随机选择用仿真数据还是真实数据（4:6）
        use_sim_data = torch.rand(1, device=device) < phase3_sim_real_ratio  # 恢复随机逻辑
        # 统一初始化is_paired（仿真分支动态赋值，真实分支固定False）
        is_paired = False
        if use_sim_data:
            # 子分支1：仿真数据（20%配对+80%非配对）
            # 配对定义：train_data(仿真水下) + high_quality(同样本仿真清洁)
            is_paired = torch.rand(1, device=device) < phase3_sim_paired_ratio

            # 随机选1张仿真清洁图作为train_data的基础（单batch）
            idx_train = torch.randint(0, clean.tensors.size(0), (1,), device=device).item()
            # 非配对时随机选另一张，配对时复用当前张
            idx_high = idx_train if is_paired else torch.randint(0, clean.tensors.size(0), (1,), device=device).item()

            # 处理train_data（仿真水下图像，单batch）
            clean_train = clean.tensors[idx_train:idx_train + 1, ...]  # [1, C, H, W] 单batch
            depth_train = depth.tensors[idx_train:idx_train + 1, ...]
            mask_train = clean.mask[idx_train:idx_train + 1, ...]
            simulate_under = process_tensor_pipeline(
                clean_train, mask_train, depth_train, is_underwater=False,
                target_ratio=target_ratio, max_size=max_resize_size, device=device
            )
            train_data = build_nested_tensor(simulate_under, mask_train, device, slice_first=False)

            # 处理refer（仿真水下增强图，单batch，和train_data一一对应）
            clean_refer = process_tensor_pipeline(
                clean_train, mask_train, is_underwater=True,
                target_ratio=target_ratio, max_size=max_resize_size, device=device
            )
            refer = build_nested_tensor(clean_refer, mask_train, device)
            has_gt = True  # 仿真数据均有真实参考，标记为True
            # 处理high_quality（配对=同样本，非配对=随机样本，均为单batch）
            if is_paired:
                high_quality = refer  # 配对：复用refer
            else:
                clean_high = clean.tensors[idx_high:idx_high + 1, ...]  # 单batch
                mask_high = clean.mask[idx_high:idx_high + 1, ...]
                clean_high = crop_valid_regions(clean_high, mask_high)
                clean_high = resize_if_needed([clean_high], max_size=max_resize_size)[0]
                high_quality = build_nested_tensor(clean_high, mask_high, device)
            # 新增：is_real_data 标记（仿真数据→False）
            train_data.is_real_data = False

        else:
            # 子分支2：真实数据（10%配对+90%非配对）
            # 配对定义：train_data(真实水下) + high_quality(同样本真实增强图)
            # 随机选1张真实水下图像（单batch）
            idx_train = torch.randint(0, under.tensors.size(0), (1,), device=device).item()

            # 处理train_data（真实水下图像，单batch，无冗余重复）
            under_train = under.tensors[idx_train:idx_train + 1, ...]  # [1, C, H, W] 单batch
            mask_train = under.mask[idx_train:idx_train + 1, ...]
            train_data = build_nested_tensor(under_train, mask_train, device, slice_first=False)

            # 处理refer（真实水下增强图，单batch，和train_data一一对应）
            refer = train_data  # 保持数据结构与有参考场景一致，无需修改后续代码
            has_gt = False

            # 非配对：随机选1张仿真清洁图（单batch）
            idx_high = torch.randint(0, clean.tensors.size(0), (1,), device=device).item()
            clean_high = clean.tensors[idx_high:idx_high + 1, ...]
            mask_high = clean.mask[idx_high:idx_high + 1, ...]
            clean_high = crop_valid_regions(clean_high, mask_high)
            clean_high = resize_if_needed([clean_high], max_size=max_resize_size)[0]
            high_quality = build_nested_tensor(clean_high, mask_high, device)
            # 新增：is_real_data 标记（真实数据→True）
            train_data.is_real_data = True
        train_data.has_gt = has_gt
        train_data.is_paired = is_paired

    # ========== Phase 4: 纯测试/预测阶段（新增核心逻辑） ==========
    elif phase == 4:
        # 核心规则：train_data=水下图像（仿真/真实可选），high_quality=对应干净图像，无随机配对/非配对
        # phase_use保持4，标识测试/预测阶段
        phase_use = 4

        if phase4_use_real_under:
            # 分支1：使用真实水下图像（预测真实场景，更贴合实际需求）
            # 随机选1张真实水下图像（单batch，与训练阶段数据维度一致）
            idx = torch.randint(0, under.tensors.size(0), (1,), device=device).item()
            # train_data = 真实水下图像
            under_img = under.tensors[idx:idx + 1, ...]  # [1, C, H, W] 单batch
            #under_mask = under.mask[idx:idx + 1, ...]
            train_data = under #build_nested_tensor(under_img, under_mask, device, slice_first=False)

            # refer = 真实水下增强图（与train_data一一对应，保持格式统一）

            #underE_img = underE.tensors[idx:idx + 1, ...]
            #refer = build_nested_tensor(underE_img, under_mask, device)
            refer = under

            # high_quality = 随机选1张仿真干净图像（预测阶段的干净参考）
            idx_clean = torch.randint(0, clean.tensors.size(0), (1,), device=device).item()
            clean_img = clean.tensors[idx_clean:idx_clean + 1, ...]
            clean_mask = clean.mask[idx_clean:idx_clean + 1, ...]
            # 与训练阶段一致的预处理（裁剪有效区域+resize）
            clean_img = crop_valid_regions(clean_img, clean_mask)
            clean_img = resize_if_needed([clean_img], max_size=max_resize_size)[0]
            high_quality = build_nested_tensor(clean_img, clean_mask, device)
            # 新增：is_real_data 标记（使用真实水下→True）
            train_data.is_real_data = True

        else:
            # 分支2：使用仿真水下图像（测试模型对仿真数据的恢复能力）
            # 随机选1张仿真清洁图，生成仿真水下图像（与Phase1/2预处理逻辑一致）
            idx = torch.randint(0, clean.tensors.size(0), (1,), device=device).item()
            clean_img = clean.tensors[idx:idx + 1, ...]  # [1, C, H, W] 单batch
            depth_img = depth.tensors[idx:idx + 1, ...]
            clean_mask = clean.mask[idx:idx + 1, ...]

            # 可选：是否对仿真水下做与训练一致的预处理（默认开启）
            if phase4_sim_preprocess:
                simulate_under = process_tensor_pipeline(
                    clean_img, clean_mask, depth_img, is_underwater=False,
                    target_ratio=target_ratio, max_size=max_resize_size, device=device
                )
                # 干净图像同步预处理（保持尺寸/比例一致）
                clean_processed = process_tensor_pipeline(
                    clean_img, clean_mask, is_underwater=True,
                    target_ratio=target_ratio, max_size=max_resize_size, device=device
                )
            else:
                # 不预处理：直接使用原始仿真清洁图生成水下图像（极简模式）
                simulate_under = clean_img  # 可根据实际仿真逻辑替换
                clean_processed = clean_img

            # train_data = 仿真水下图像
            train_data = build_nested_tensor(simulate_under, clean_mask, device, slice_first=False)
            # refer = 预处理后的干净图像（保持格式统一）
            refer = build_nested_tensor(clean_processed, clean_mask, device)
            # high_quality = 预处理后的干净图像（预测阶段的干净参考）
            high_quality = refer
            # 新增：is_real_data 标记（使用仿真水下→False）
            train_data.is_real_data = False
        train_data.has_gt = False
        train_data.is_paired = False

    # 异常处理：未知阶段
    else:
        raise ValueError(f"未知训练阶段phase={phase}，仅支持1/2/3/4，其中4为测试/预测阶段")

    return train_data, refer, high_quality, phase_use