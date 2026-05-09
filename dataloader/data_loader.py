import math
import os
import random
import numpy as np
from typing import Optional, Tuple, List, Set, Union
from pathlib import Path

from scipy.io import loadmat
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# 自定义模块导入（按实际路径调整）
import dataloader.transforms as T
from utils import logger, misc
from utils.common_utils import (
    make_divisible, join_paths, validate_and_fix_paths,
    visualize_tensor, load_flist)

from dataloader.under_generate import kitti_under
# ======================== 全局常量 & 类型别名 ========================
# 提升代码可读性，减少魔法值
Default_crop_size = (256, 256)  # (h, w)
Default_scale_intervals = [40]
Default_scale_factor = 0.25
Default_max_scales = 10
Default_scale_div_factor = 64
Default_min_crop_size = (160, 160)  # (h, w)
Default_max_crop_size = (256, 256)  # (h, w)
Kitti_mix_ratio = 0.6  # KITTI/NYU 数据集混合比例
Nyu_crop_pixels = 10    # NYU 数据集裁剪像素数
# 类型别名
CropSize = Tuple[int, int]
BatchTuple = Tuple[int, int, int]  # (crop_h, crop_w, img_idx)
ImageBatchTuple = Tuple[int, int, int]  # (h, w, batch_size)

def make_transforms(training: bool, scales: CropSize) -> T.Compose:
    """
    生成数据增强变换（优化：参数校验 + 注释增强 + 标准化配置解耦）
    Args:
        training: 是否训练模式
        scales: 裁剪尺寸 (h, w)
    Returns:
        组合后的变换
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 保留你的 scales 变量，但适当调整范围

    # scales = [480, 512, 544, 576, 608]  # 比原版更大一些
    if training:
        # 训练阶段：正常随机缩放（支持放大/缩小）
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResize(scales, max_size=None),
            normalize,
        ])
    else:
        # 关键修改：启用 min_scale_only=True，目标最小边400，最大边1000
        # 效果：1. 原始图像最小边 <=400 → 保留原始尺寸（不放大）；2. 原始图像最小边 >400 → 等比例缩小到400，且最大边不超过1000
        return T.Compose([
            T.RandomResize([256], max_size=1000, min_scale_only=True), # 比训练时稍大，但不过大
            normalize,
        ])


# ------------------------------ 公共工具函数 ------------------------------
def _get_dataset_paths(args, dataset_type: str = "train") -> dict:
    """
    公共路径配置函数（复用路径逻辑，避免重复代码）
    Args:
        args: 命令行参数/配置对象
        dataset_type: 数据集类型 ("train"/"val"/"pred")
    Returns:
        paths: 对应数据集的路径字典
    """
    root_path = getattr(args, "dataset_root_path", "E:/Python-master/dataset")

    # 基础路径参数名映射
    path_param_map = {
        "clean": f"{dataset_type}_clean",
        "depth": f"{dataset_type}_depth",
        "image_2": f"{dataset_type}_path_image_2",
        "calib": f"{dataset_type}_path_calib",
        "label_2": f"{dataset_type}_path_label_2",
        "velo": f"{dataset_type}_path_velodyne",
        "under": f"{dataset_type}_under",
        "underE": f"{dataset_type}_underE"
    }

    paths = {}
    for key, param_suffix in path_param_map.items():
        # 拼接完整的参数名（比如 dataset_train_clean）
        current_param = f"dataset_{param_suffix}"
        # 回退的训练集参数名（比如 dataset_train_clean）
        fallback_param = f"dataset_{param_suffix.replace(dataset_type, 'train')}"
        # 核心修复：获取参数名对应的实际路径值，而不是用参数名本身
        if hasattr(args, current_param):
            # 取args中current_param对应的实际值（比如 /nyu/image）
            sub_path = getattr(args, current_param)
        else:
            # 回退到训练集参数的实际值
            sub_path = getattr(args, fallback_param)

        # 拼接根路径和子路径（得到正确路径：E:/Python-master/dataset/nyu/image）
        paths[key] = join_paths(root_path, sub_path)
    return paths


# ------------------------------ 训练集加载器 ------------------------------

def get_train_dataset(
        args,
        USE_FAKE_DDP: bool,
        rank: int = 0,
        world_size: int = 1
) -> Tuple[DataLoader, Sampler]:
    """
    仅构建训练集加载器（训练时专用）
    Args:
        args: 命令行参数/配置对象
        USE_FAKE_DDP: bool，本地伪DDP模式开关（True=本地，False=服务器分布式）
        rank: int，当前进程的全局rank（分布式用）
        world_size: int，总进程数（分布式用）
    Returns:
        train_loader: 训练数据加载器
        train_sampler: 训练采样器（含分布式逻辑）
    """
    # 1. 获取训练集路径
    train_paths = _get_dataset_paths(args, dataset_type="train")

    # 2. 创建训练集Dataset
    dataset_train = ImageDataset(
        clean_paths=train_paths["clean"][0],
        depth_paths=train_paths["depth"][0],
        image_2_paths=train_paths["image_2"][0],
        calib_paths=train_paths["calib"][0],
        label_2_paths=train_paths["label_2"][0],
        velo_paths=train_paths["velo"][0],
        under_paths=train_paths["under"][0],
        under_enhanced_paths=train_paths["underE"][0],
        is_training=True  # 训练模式：开启数据增强
    )
    n_train_samples = len(dataset_train)

    # 3. 训练集采样器（分布式/伪DDP分支）
    if USE_FAKE_DDP:
        train_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples=n_train_samples,
            batch_size_gpu0 = getattr(args,"runtime_batch_size",2),
            is_training=True
        )
    else:
        dist_sampler = DistributedSampler(
            dataset_train,
            num_replicas=world_size,  # 总进程数（GPU数）
            rank=rank,                # 当前进程rank
            shuffle=True,             # 保持原有shuffle逻辑
            drop_last=True
        )
        train_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples=n_train_samples,
            batch_size_gpu0=getattr(args, "runtime_batch_size", 2),
            is_training=True,
            sampler=dist_sampler  # 新增：传入分布式采样器
        )
    # 4. 构建DataLoader
    num_workers = getattr(args, "runtime_workers", 0)
    num_workers = num_workers if not USE_FAKE_DDP else 0  # 本地模式设为0避免冲突
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=1,  # 原有逻辑：批量由sampler处理
        batch_sampler=train_sampler,
        pin_memory=True,  # 建议开启，提升GPU加载速度
        num_workers=num_workers,  # 可通过args配置
        collate_fn=ImageDataset.collate_fn,
        # 分布式下关闭shuffle（由DistributedSampler控制）
        shuffle=False,  # 关键：分布式模式下必须设为False
        persistent_workers=True if num_workers > 0 else False,  # 优化：持久化worker，减少重复创建
    )

    # 5. 日志输出
    logger.info(
        f"[Rank {rank}] 训练集加载器初始化完成 | "
        f"样本数: {n_train_samples} | num_workers: {num_workers} | 迭代次数: {len(train_loader)}"
    )

    return train_loader, train_sampler


# ------------------------------ 验证集加载器 ------------------------------
def get_val_dataset(
        args,
        USE_FAKE_DDP: bool,
        rank: int = 0,
        world_size: int = 1,
        val_batch_size: int = 2
) -> Tuple[DataLoader, Sampler]:
    """
    仅构建验证集加载器（验证时专用）
    Args:
        args: 命令行参数/配置对象
        USE_FAKE_DDP: bool，本地伪DDP模式开关（True=本地，False=服务器分布式）
        rank: int，当前进程的全局rank（分布式用）
        world_size: int，总进程数（分布式用）
        val_batch_size: int，验证集批次大小（建议2/4，根据显存调整）
    Returns:
        val_loader: 验证数据加载器
    """
    # 1. 获取验证集路径
    val_paths = _get_dataset_paths(args, dataset_type="val")

    # 2. 创建验证集Dataset
    dataset_val = ImageDataset(
        clean_paths=val_paths["clean"][0],
        depth_paths=val_paths["depth"][0],
        image_2_paths=val_paths["image_2"][0],
        calib_paths=val_paths["calib"][0],
        label_2_paths=val_paths["label_2"][0],
        velo_paths=val_paths["velo"][0],
        under_paths=val_paths["under"][0],
        under_enhanced_paths=val_paths["underE"][0],
        is_training=False  # 验证模式：关闭数据增强
    )
    n_val_samples = len(dataset_val)

    # 3. 验证集采样器（顺序采样，保证可复现）
    if USE_FAKE_DDP:
        val_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples= n_val_samples,
            batch_size_gpu0=val_batch_size,
            is_training=False
        )
    else:
        dist_sampler = DistributedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # 验证集不打乱
            drop_last=False  # 不丢弃最后一个批次
        )
        val_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples= n_val_samples,
            batch_size_gpu0=val_batch_size,
            is_training=False,
            sampler=dist_sampler  # 新增：传入分布式采样器
        )

    # 4. 构建DataLoader
    num_workers = getattr(args, "runtime_workers", 0)
    num_workers = num_workers if not USE_FAKE_DDP else 0  # 本地模式设为0避免冲突
    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=1,  # 原有逻辑：批量由sampler处理
        batch_sampler=val_sampler,
        pin_memory=True,  # 建议开启，提升GPU加载速度
        num_workers=num_workers,  # 可通过args配置
        collate_fn=ImageDataset.collate_fn,
        # 分布式下关闭shuffle（由DistributedSampler控制）
        shuffle=False,  # 关键：分布式模式下必须设为False
        persistent_workers=True if num_workers > 0 else False,  # 优化：持久化worker，减少重复创建
    )
    # 6. 日志输出
    logger.info(
        f"[Rank {rank}] 验证集加载器初始化完成 | "
        f"样本数: {n_val_samples} | num_workers: {num_workers} | 批次大小: {val_batch_size} | 迭代次数: {len(val_loader)}"
    )

    return val_loader, val_sampler


# ------------------------------ 预测集加载器 ------------------------------
def get_pred_dataset(
        args,
        USE_FAKE_DDP: bool,
        rank: int = 0,
        world_size: int = 1,
        pred_batch_size: int = 1
) -> Tuple[DataLoader, Sampler]:
    """
    仅构建预测集加载器（推理时专用）
    Args:
        args: 命令行参数/配置对象
        USE_FAKE_DDP: bool，本地伪DDP模式开关（True=本地，False=服务器分布式）
        rank: int，当前进程的全局rank（分布式用）
        world_size: int，总进程数（分布式用）
        pred_batch_size: int，预测集批次大小（固定为1，便于单样本推理）
    Returns:
        pred_loader: 预测数据加载器
    """
    # 1. 获取预测集路径
    pred_paths = _get_dataset_paths(args, dataset_type="pred")

    # 2. 创建预测集Dataset
    dataset_pred = ImageDataset(
        clean_paths=pred_paths["clean"][0],
        depth_paths=pred_paths["depth"][0],
        image_2_paths=pred_paths["image_2"][0],
        calib_paths=pred_paths["calib"][0],
        label_2_paths=pred_paths["label_2"][0],
        velo_paths=pred_paths["velo"][0],
        under_paths=pred_paths["under"][0],
        under_enhanced_paths=pred_paths["underE"][0],
        is_training=False  # 预测模式：关闭数据增强
    )
    n_pred_samples = len(dataset_pred)

    # 3. 验证集采样器（顺序采样，保证可复现）
    if USE_FAKE_DDP:
        pred_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples= n_pred_samples,
            batch_size_gpu0=pred_batch_size,
            is_training=False
        )
    else:
        dist_sampler = DistributedSampler(
            dataset_pred,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # 验证集不打乱
            drop_last=False  # 不丢弃最后一个批次
        )
        pred_sampler = VariableBatchSamper(
            opts=args,
            n_data_samples= n_pred_samples,
            batch_size_gpu0=pred_batch_size,
            is_training=False,
            sampler=dist_sampler  # 新增：传入分布式采样器
        )

    # 4. 构建DataLoader
    num_workers = getattr(args, "runtime_workers", 0)
    num_workers = num_workers if not USE_FAKE_DDP else 0  # 本地模式设为0避免冲突
    pred_loader = DataLoader(
        dataset=dataset_pred,
        batch_size=1,  # 原有逻辑：批量由sampler处理
        batch_sampler=pred_sampler,
        pin_memory=True,  # 建议开启，提升GPU加载速度
        num_workers=num_workers,  # 可通过args配置
        collate_fn=ImageDataset.collate_fn,
        # 分布式下关闭shuffle（由DistributedSampler控制）
        shuffle=False,  # 关键：分布式模式下必须设为False
        persistent_workers=True if num_workers > 0 else False,  # 优化：持久化worker，减少重复创建
    )

    # 6. 日志输出
    logger.info(
        f"[Rank {rank}] 预测集加载器初始化完成 | "
        f"样本数: {n_pred_samples} | num_workers: {num_workers} | 批次大小: {pred_batch_size} | 迭代次数: {len(pred_loader)}"
    )

    return pred_loader, pred_sampler


class ImageDataset(Dataset):
    """
    混合数据集加载类（KITTI + NYU）（优化：可读性 + 错误处理 + 性能）
    """
    def __init__(
        self,
        clean_paths: str,
        depth_paths: str,
        image_2_paths: str,
        calib_paths: str,
        label_2_paths: str,
        velo_paths: str,
        under_paths: str,
        under_enhanced_paths: str,
        is_training: Optional[bool] = True
    ):
        self.is_training = is_training

        self.clean_samples = load_flist(clean_paths)
        self.depth_samples = load_flist(depth_paths)
        self.under_samples = load_flist(under_paths)
        self.under_enhanced_samples = load_flist(under_enhanced_paths)
        self.kitti_under = kitti_under(calib_paths,
                                       image_2_paths,
                                       label_2_paths,
                                       velo_paths)
    def __len__(self):
        """优化：逻辑更清晰 + 注释"""
        if self.is_training:
            # 训练模式：样本数 = 1.5 * clean样本数（混合KITTI+NYU）
            data_num = int(2.5 * len(self.clean_samples))
        else:
            # 验证模式：样本数 = under样本数
            data_num = len(self.under_samples)
        return data_num
    @staticmethod
    def collate_fn(batch):
        """
        自定义collate_fn（优化：类型注解 + 鲁棒性 + 注释）
        Args:
            batch: 批次数据，格式为 [(clean, depth, under, underE), ...]
        Returns:
            封装后的嵌套张量元组
        """
        # clean, depth, under, underE = zip(*batch)
        batch = list(zip(*batch))
        batch = [misc.nested_tensor_from_tensor_list(elem) for elem in batch]
        return tuple(batch)

    def __getitem__(self, batch_indexs_tup: Tuple):
        """
         获取单批次数据（优化：参数校验 + 逻辑解耦）
         Args:
             batch_indexs_tup: (crop_h, crop_w, img_index)
         Returns:
             clean, depth, under, underE: 处理后的张量
         """
        crop_size_h, crop_size_w, img_index = batch_indexs_tup
        self._transforms = make_transforms(self.is_training, (crop_size_h, crop_size_w))
        clean, depth, under, underE = self.load_item(img_index)
        return clean, depth, under, underE
    def load_item(self, img_index):
        """
        加载单样本数据（优化：逻辑解耦 + 错误处理 + 减少魔法值）
        Args:
            img_index: 样本索引
        Returns:
            clean, depth, under, underE: 处理后的张量
        """
        # 1. 加载under/underE数据（优化：索引生成更合理）
        if self.is_training:
            under_index = random.randint(0, len(self.under_samples) - 1)
            underE_index = under_index  # 保持under和underE索引一致
        else:
            #under_index = random.randint(0, len(self.under_samples) - 1)
            #underE_index = under_index  # 保持under和underE索引一致
            under_index = img_index
            underE_index = under_index  # 保持under和underE索引一致

        #np.random.random()
        if np.random.random() < Kitti_mix_ratio: #=0.6
            frame = random.randint(0, len(self.kitti_under.image_2_paths) - 1)
            clean, depth = self.kitti_under.kitti_data(frame)
            clean = Image.fromarray(clean)
            depth = Image.fromarray(depth)
        else:
            clean_index = random.randint(0, len(self.clean_samples) - 1)
            depth_index = clean_index
            # 路径加载 + 校验
            clean_path = self.clean_samples[clean_index]
            depth_path = self.depth_samples[depth_index]
            depth_path = validate_and_fix_paths(clean_path, depth_path)
            # -------
            clean = Image.open(clean_path).convert('RGB')
            depth = np.array(loadmat(depth_path)['dph']).astype(np.float32)
            depth = Image.fromarray(depth)

            # 获取原始尺寸
            clean_width, clean_height = clean.size
            depth_width, depth_height = depth.size

            # 裁剪（优化：使用常量 + 边界校验）
            clean = clean.crop((Nyu_crop_pixels, Nyu_crop_pixels, clean_width - Nyu_crop_pixels, clean_height - Nyu_crop_pixels))
            depth = depth.crop((Nyu_crop_pixels, Nyu_crop_pixels, clean_width - Nyu_crop_pixels, clean_height - Nyu_crop_pixels))
        # 3. 数据变换（优化：统一处理）
        (clean, depth), _ = self._transforms((clean,depth),[])
        # 4. 加载under/underE数据（优化：路径校验 + 变换）

        under_path = self.under_samples[under_index]
        under = Image.open(under_path).convert('RGB')

        underE_path = under_path  # 默认值：直接复用under路径，避免后续二次判断
        if hasattr(self, 'under_enhanced_samples') and self.under_enhanced_samples:
            underE_path = self.under_enhanced_samples[underE_index]
        underE_path = validate_and_fix_paths(under_path, underE_path)

        underE = under.copy()  # 先默认复用under，加载成功再覆盖
        if os.path.isfile(underE_path):
            underE = Image.open(underE_path).convert('RGB')

        (under, underE), _ = self._transforms((under, underE), [])
        return clean, depth, under, underE

class VariableBatchSamper(Sampler):
    def __init__(
        self,
        opts,
        n_data_samples: int,
        batch_size_gpu0: int,
        is_training: Optional[bool] = False,
        sampler: Optional[Sampler] = None,  # 新增：接收DistributedSampl
    ):
        # 1. 分布式相关初始化
        self.dist_sampler = sampler  # 分布式采样器（None=本地模式）
        # 分布式下从opts获取总GPU数，本地自动检测
        self.n_gpus: int = max(1, torch.cuda.device_count()) if self.dist_sampler is None else \
                           getattr(opts, 'dev_num_gpus', torch.cuda.device_count())  # 分布式下从opts取总GPU数
        self.batch_size_gpu0 = batch_size_gpu0
        # 2. 本地/分布式索引初始化
        if self.dist_sampler is not None:
            # 分布式模式：基于DistributedSampler的分片索引（不再手动扩展总样本数）
            self.n_samples = len(self.dist_sampler)  # 当前进程的样本数（分片后）
            self.img_indices = list(range(self.n_samples))  # 临时占位，__iter__时替换为真实分片索引
        else:
            # 本地：扩展样本数到GPU数的整数倍
            n_samples_per_gpu = math.ceil(n_data_samples / self.n_gpus)
            self.n_samples = n_samples_per_gpu * self.n_gpus
            self.img_indices = list(range(n_data_samples))
            # 补全样本（循环填充）
            self.img_indices += self.img_indices[:(self.n_samples - n_data_samples)]

        # 3. 原有动态尺度/批量相关参数（完全保留）
        self.shuffle = True if is_training else False
        self.epoch = 0
        self.opts = opts

        self.min_crop_size_w = getattr(opts, "sampler_vbs_min_crop_size_width", 160)
        self.max_crop_size_w = getattr(opts, "sampler_vbs_max_crop_size_width", 256)
        self.min_crop_size_h = getattr(opts, "sampler_vbs_min_crop_size_height", 160)
        self.max_crop_size_h =  getattr(opts, "sampler_vbs_max_crop_size_height", 256)

        self.crop_size_w = getattr(opts, "sampler_vbs_crop_size_width", 256)
        self.crop_size_h = getattr(opts, "sampler_vbs_crop_size_height", 256)

        scale_ep_intervals: list or int = getattr(opts, "sampler_vbs_ep_intervals", [40])
        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]
        self.scale_ep_intervals = scale_ep_intervals
        self.scale_inc_factor = getattr(opts, "sampler_vbs_scale_inc_factor", 0.25)
        self.scale_inc = getattr(opts, "sampler_vbs_scale_inc", False)

        self.max_img_scales = getattr(opts, "sampler_vbs_max_n_scales", 10)
        self.check_scale_div_factor = getattr(opts, "sampler_vbs_check_scale", 64)
        if is_training:

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h)
        else:

            crop_size_w: int = getattr(opts, "sampler_vbs_max_crop_size_width", 256)
            crop_size_h: int = getattr(opts, "sampler_vbs_max_crop_size_height", 256)
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

    def __len__(self):
        return self.n_samples
    def __iter__(self):
        """核心迭代逻辑：兼容本地/分布式索引"""
        # 1. 获取真实索引列表（分布式=分片索引，本地=原有扩展索引）
        if self.dist_sampler is not None:
            # 分布式模式：从DistributedSampler获取当前进程的分片索引
            indices = list(self.dist_sampler)
        else:
            # 本地模式：使用原有扩展索引
            indices = self.img_indices.copy()
        # 2. Shuffle（优化：分布式下seed结合epoch，保证一致性）

        if self.shuffle:
            common_seed = getattr(self.opts, "common_seed", 0)
            # 修复：获取真实的全局Rank（DDP场景）
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()  # 全局Rank（0/1/2/3）
            else:
                rank = self.dist_sampler.rank if self.dist_sampler else 0

            #rank = self.dist_sampler.rank if self.dist_sampler else 0
            base_seed = common_seed + rank

            # 给indices单独的种子（epoch偏移，保证每个epoch不同）
            rng_indices = random.Random(base_seed + self.epoch * 1000)
            rng_indices.shuffle(indices)

            # 给img_batch_tuples单独的种子（偏移base_seed，避免和indices的随机序列重复）
            rng_tuples = random.Random(base_seed + self.epoch * 1000 + 100)  # +100避免种子重叠
            rng_tuples.shuffle(self.img_batch_tuples)

        # 3. 动态批量生成（保留原有核心逻辑）
        start_index = 0
        while start_index < len(indices):
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)
            end_index = min(start_index + batch_size, len(indices))
            batch_ids = indices[start_index:end_index]
            # 补全批量（避免最后一批样本数不足）
            if len(batch_ids) != batch_size:
                # 循环填充（优化：只补全缺失的数量）
                need = batch_size - len(batch_ids)
                batch_ids += indices[:need]

            start_index += batch_size
            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def set_epoch(self, epoch: int) -> None:
        """新增：对齐DistributedSampler接口，保证多进程shuffle一致"""
        self.epoch = epoch
        # 分布式下同步更新内层sampler的epoch
        if self.dist_sampler is not None and hasattr(self.dist_sampler, 'set_epoch'):
            self.dist_sampler.set_epoch(epoch)
        # 原有尺度更新逻辑整合
        self.update_scales(epoch)

    def update_scales(self, epoch, *args, **kwargs):
        if epoch in self.scale_ep_intervals and self.scale_inc:
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h)
            logger.log('Scales updated in {}'.format(self.__class__.__name__))
            logger.log("New scales: {}".format(self.img_batch_tuples))

def _image_batch_pairs(crop_size_w: int,
                       crop_size_h: int,
                       batch_size_gpu0: int,
                       n_gpus: int,
                       max_scales: Optional[float] = 5,
                       check_scale_div_factor: Optional[int] = 64,
                       min_crop_size_w: Optional[int] = 160,
                       max_crop_size_w: Optional[int] = 320,
                       min_crop_size_h: Optional[int] = 160,
                       max_crop_size_h: Optional[int] = 320,
                       *args, **kwargs) -> list:
    """
    生成图像尺寸-批量大小映射（优化：可读性 + 性能 + 注释）
    Args:
        crop_size_w/h: 基准图像尺寸
        batch_size_gpu0: 基准批量大小
        n_gpus: GPU数
        max_scales: 生成的尺度数量
        check_scale_div_factor: 尺度整除因子
        min/max_crop_size_w/h: 裁剪尺寸范围
    Returns:
        排序后的 (h, w, batch_size) 列表
    """
    # 1. 生成宽度/高度维度（优化：去重 + 包含基准尺寸）
    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)
    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)
    # 2. 生成有效图像尺度（优化：去重 + 整除校验）
    image_scales = set()
    for h, w in zip(height_dims, width_dims):
        h = make_divisible(h, check_scale_div_factor)
        w = make_divisible(w, check_scale_div_factor)
        image_scales.add((h, w))
    image_scales = list(image_scales)

    # 3. 计算每个尺度对应的批量大小（优化：减少冗余计算）
    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in image_scales:
        # 按像素数调整批量大小（保证GPU内存利用率）
        _bsz = max(batch_size_gpu0, int(round(n_elements / (crop_h * crop_y), 2)))
        #_bsz = int(2)
        img_batch_tuples.add((crop_h, crop_y, _bsz))
    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)




