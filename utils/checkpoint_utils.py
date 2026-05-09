import copy
import os
import torch
from typing import Optional, Union
import math
import glob
from copy import deepcopy
import argparse
from utils import logger

CHECKPOINT_EXTN = "pt"
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
def get_weight_copy_strategy(source_shape: torch.Size, target_shape: torch.Size) -> str:
    """
    根据源形状和目标形状确定权重复制策略

    Args:
        source_shape: 源权重形状
        target_shape: 目标权重形状

    Returns:
        strategy: 复制策略名称
    """
    if len(source_shape) != len(target_shape):
        return "skip"

    # 卷积层权重 [out_channels, in_channels, kernel_h, kernel_w]
    if len(source_shape) == 4:
        # 输出通道数变化
        if (source_shape[1:] == target_shape[1:] and
                source_shape[0] != target_shape[0]):
            return "expand_output_channels"
        # 输入通道数变化
        elif (source_shape[0] == target_shape[0] and
              source_shape[2:] == target_shape[2:] and
              source_shape[1] != target_shape[1]):
            return "expand_input_channels"
        # 两个维度都变化
        elif (source_shape[2:] == target_shape[2:] and
              source_shape[0] != target_shape[0] and
              source_shape[1] != target_shape[1]):
            return "expand_both_channels"

    # 线性层权重 [out_features, in_features]
    elif len(source_shape) == 2:
        if source_shape[0] != target_shape[0] or source_shape[1] != target_shape[1]:
            return "expand_linear"

    return "exact_match"


def expand_output_channels(source_weight: torch.Tensor, target_weight: torch.Tensor,
                           source_bias: Optional[torch.Tensor] = None,
                           target_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    扩展输出通道数

    Args:
        source_weight: 源权重 [old_out, in, h, w]
        target_weight: 目标权重 [new_out, in, h, w]

    Returns:
        updated_weight: 更新后的权重
        updated_bias: 更新后的偏置
    """
    old_out_channels = source_weight.shape[0]
    new_out_channels = target_weight.shape[0]

    # 复制前old_out_channels个通道
    target_weight.data[:old_out_channels] = source_weight.data

    # 剩余通道使用Kaiming初始化
    if new_out_channels > old_out_channels:
        nn.init.kaiming_uniform_(target_weight.data[old_out_channels:], nonlinearity='relu')

    # 处理偏置
    if source_bias is not None and target_bias is not None:
        target_bias.data[:old_out_channels] = source_bias.data
        if new_out_channels > old_out_channels:
            nn.init.constant_(target_bias.data[old_out_channels:], 0.0)

    return target_weight, target_bias


def expand_input_channels(source_weight: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    """
    扩展输入通道数

    Args:
        source_weight: 源权重 [out, old_in, h, w]
        target_weight: 目标权重 [out, new_in, h, w]

    Returns:
        updated_weight: 更新后的权重
    """
    old_in_channels = source_weight.shape[1]
    new_in_channels = target_weight.shape[1]

    # 复制前old_in_channels个输入通道
    target_weight.data[:, :old_in_channels] = source_weight.data

    # 剩余输入通道初始化为0
    if new_in_channels > old_in_channels:
        target_weight.data[:, old_in_channels:].zero_()

    return target_weight


def expand_both_channels(source_weight: torch.Tensor, target_weight: torch.Tensor,
                         source_bias: Optional[torch.Tensor] = None,
                         target_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    同时扩展输入和输出通道数

    Args:
        source_weight: 源权重 [old_out, old_in, h, w]
        target_weight: 目标权重 [new_out, new_in, h, w]

    Returns:
        updated_weight: 更新后的权重
        updated_bias: 更新后的偏置
    """
    old_out, old_in = source_weight.shape[:2]
    new_out, new_in = target_weight.shape[:2]

    min_out = min(old_out, new_out)
    min_in = min(old_in, new_in)

    # 复制重叠区域的权重
    target_weight.data[:min_out, :min_in] = source_weight.data[:min_out, :min_in]

    # 初始化剩余区域
    if new_out > old_out:
        nn.init.kaiming_uniform_(target_weight.data[old_out:], nonlinearity='relu')

    # 处理偏置
    if source_bias is not None and target_bias is not None:
        min_bias = min(old_out, new_out)
        target_bias.data[:min_bias] = source_bias.data[:min_bias]
        if new_out > old_out:
            nn.init.constant_(target_bias.data[old_out:], 0.0)

    return target_weight, target_bias


def expand_linear_weights(source_weight: torch.Tensor, target_weight: torch.Tensor,
                          source_bias: Optional[torch.Tensor] = None,
                          target_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    扩展线性层权重

    Args:
        source_weight: 源权重 [old_out, old_in]
        target_weight: 目标权重 [new_out, new_in]

    Returns:
        updated_weight: 更新后的权重
        updated_bias: 更新后的偏置
    """
    old_out, old_in = source_weight.shape
    new_out, new_in = target_weight.shape

    min_out = min(old_out, new_out)
    min_in = min(old_in, new_in)

    # 复制重叠区域的权重
    target_weight.data[:min_out, :min_in] = source_weight.data[:min_out, :min_in]

    # 初始化剩余区域
    if new_out > old_out or new_in > old_in:
        nn.init.kaiming_uniform_(target_weight.data, nonlinearity='relu')
        # 重新复制重叠区域（因为初始化会覆盖）
        target_weight.data[:min_out, :min_in] = source_weight.data[:min_out, :min_in]

    # 处理偏置
    if source_bias is not None and target_bias is not None:
        min_bias = min(old_out, new_out)
        target_bias.data[:min_bias] = source_bias.data[:min_bias]
        if new_out > old_out:
            nn.init.constant_(target_bias.data[old_out:], 0.0)

    return target_weight, target_bias



class EMA(object):
    '''
        Exponential moving average of model weights
    '''

    def __init__(self, model, ema_momentum: float = 0.1, device: str = ''):
        # make a deep copy of the model for accumulating moving average of parameters
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.momentum = ema_momentum
        self.device = device
        if device:
            self.ema_model.to(device=device)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update_parameters(self, model):
        # correct a mismatch in state dict keys
        has_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                if has_module:
                    # .module is added if we use DistributedDataParallel or DataParallel wrappers around model
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * model_v))
        """
        has_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                if has_module:
                    # .module is added if we use DistributedDataParallel or DataParallel wrappers around model
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * model_v))
        """


class BaseOptim(object):
    def __init__(self, opts) -> None:
        self.eps = 1e-8
        self.lr = getattr(opts, "scheduler.lr", 0.1)
        self.weight_decay = getattr(opts, "optim.weight_decay", 4e-5)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser


def get_model_state_dict(model):
    if isinstance(model, EMA):
        return get_model_state_dict(model.ema_model)
    else:
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()











# 新增：剥离DDP的module.前缀（核心适配DDP）
def strip_ddp_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    """
    剥离DDP保存的state_dict中"module."前缀，适配原始模型加载
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[key] = v
    return new_state_dict

# 新增：分布式同步工具（确保所有进程加载完成后再继续）
def dist_barrier_if_available(USE_FAKE_DDP: bool):
    """仅在真实DDP模式下执行进程同步"""
    if not USE_FAKE_DDP and dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_state_dict(model, state_dict):
    """
    适配DDP模型加载：自动处理module.前缀 + 兼容原始模型/DDP模型
    """
    # 第一步：剥离state_dict中的module.前缀（如果有）
    state_dict = strip_ddp_module_prefix(state_dict)

    # 第二步：判断模型是否被DDP包装（兼容FakeDDP/真实DDP）
    if hasattr(model, 'module'):
        # 情况1：模型已被DDP/FakeDDP包装，加载到module
        model.module.load_state_dict(state_dict, strict=False)
    else:
        # 情况2：原始模型，直接加载
        model.load_state_dict(state_dict, strict=False)
    return model


# 补充：兼容式加载（原代码中的load_state_dict_compatible依赖）
def load_state_dict_compatible(model, checkpoint_path: str, verbose=True):
    """兼容加载：先加载checkpoint，再剥离前缀"""
    # 加载checkpoint（仅读model_state_dict）
    checkpoint = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 剥离DDP前缀 + 加载
    model_state_dict = strip_ddp_module_prefix(model_state_dict)
    # 过滤可加载的参数
    filtered_state_dict = OrderedDict()
    loaded_count = 0
    skipped_count = 0

    for key, value in model_state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
                loaded_count += 1
            else:
                if verbose:
                    print(f"跳过形状不匹配的参数: {key}")
                    print(f"  期望形状: {model_state_dict[key].shape}, 实际形状: {value.shape}")
                skipped_count += 1
        else:
            if verbose:
                print(f"跳过不存在的参数: {key}")
            skipped_count += 1

    # 加载兼容的参数（strict=False 忽略剩余不匹配）
    if hasattr(model, 'module'):
        model.module.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(filtered_state_dict, strict=False)

    if verbose:
        print(f"参数加载完成: 成功加载 {loaded_count} 个, 跳过 {skipped_count} 个")
    return model


def save_checkpoint(
        iterations: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Union[BaseOptim, torch.optim.Optimizer],
        gradient_scalar: torch.cuda.amp.GradScaler,
        best_metric: float,
        is_best: bool,
        save_dir: str,
        is_ema_best: Optional[bool] = False,
        ema_best_metric: Optional[float] = None,
        max_ckpt_metric: Optional[bool] = False,
        k_best_checkpoints: Optional[int] = -1,
        rank: int = 0,  # 新增：当前进程rank
        USE_FAKE_DDP: bool = False,  # 新增：是否伪DDP
        *args, **kwargs
) -> None:
    """
    分布式适配：仅主进程（rank=0）保存checkpoint，避免多进程写文件冲突
    """
    # 仅主进程执行保存操作
    if rank != 0:
        return

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 获取模型state_dict（自动处理DDP/原始模型）
    def get_model_state_dict(model: nn.Module) -> OrderedDict:
        # 真实DDP模型：解包module；伪DDP/单机：直接返回
        if hasattr(model, 'module') and not USE_FAKE_DDP:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        # 清理state_dict中的冗余前缀（可选，避免加载报错）
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module.' in k else k
            new_state_dict[name] = v
        return new_state_dict

    model_state = get_model_state_dict(model)

    checkpoint = {
        'iterations': iterations,
        'epoch': epoch,
        'model_state_dict': model_state,
        'optim_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'gradient_scalar_state_dict': gradient_scalar.state_dict()
    }

    # 保存完整checkpoint（含优化器/缩放器）
    ckpt_str = os.path.join(save_dir, f"checkpoint{epoch}")  # 跨平台路径拼接
    ckpt_fname = '{}.{}'.format(ckpt_str, CHECKPOINT_EXTN)
    torch.save(checkpoint, ckpt_fname)

    # 仅保存模型权重
    checkpoint_save = {'model_state_dict': model_state}
    ckpt_str_model = '{}/checkpoint_model{}'.format(save_dir, epoch)
    ckpt_fname_model = '{}.{}'.format(ckpt_str_model, CHECKPOINT_EXTN)
    torch.save(checkpoint_save, ckpt_fname_model)

    logger.log(f"✅ Checkpoint saved to {ckpt_fname} (epoch: {epoch})")


def load_checkpoint(
        opts,
        model: torch.nn.Module,
        optimizer: Union[BaseOptim, torch.optim.Optimizer],
        gradient_scalar: torch.cuda.amp.GradScaler,
        USE_FAKE_DDP: bool = False,  # 新增：是否伪DDP
        rank: int = 0,  # 新增：当前进程rank
        local_gpu_id: int = 0  # 新增：本地GPU ID
):
    """
    分布式适配：
    1. 所有进程加载相同checkpoint，但仅主进程打印日志
    2. 自动处理DDP模型的module.前缀
    3. 设备映射到本地GPU，避免跨卡加载
    """
    # 优先使用local_gpu_id（分布式环境），兼容原有dev_id
    dev_id = local_gpu_id if (not USE_FAKE_DDP and dist.is_initialized()) else getattr(opts, "dev_device_id", None)
    device = torch.device(f'cuda:{dev_id}') if dev_id is not None else getattr(opts, "dev_device", torch.device('cpu'))

    start_epoch = start_iteration = 0
    best_metric = 0.0 if getattr(opts, "stats_checkpoint_metric_max", False) else math.inf
    resume_loc = getattr(opts, "model_pretrained", None)

    if resume_loc is not None and os.path.isfile(resume_loc):
        # 1. 加载checkpoint（映射到本地GPU/CPU）
        if dev_id is None:
            checkpoint = torch.load(resume_loc, map_location=device)
        else:
            checkpoint = torch.load(resume_loc, map_location=f'cuda:{dev_id}')

        # 2. 提取基础信息
        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iterations'] + 1
        best_metric = checkpoint['best_metric']

        # 3. 兼容式加载模型权重（自动剥离DDP前缀）
        model = load_state_dict_compatible(model, resume_loc, verbose=(rank == 0))
        # 备选：直接加载（如果load_state_dict_compatible已处理前缀，可注释上一行，打开下一行）
        # model = load_state_dict(model, checkpoint['model_state_dict'])

        # 4. 加载优化器（过滤不匹配的参数，保留当前param_groups）
        pretrained_optim_dict = checkpoint['optim_state_dict']
        current_optim_dict = optimizer.state_dict()

        # 筛选：仅保留当前优化器中存在的state
        filtered_state_dict = {
            k: v for k, v in pretrained_optim_dict['state'].items()
            if k in current_optim_dict['state']
        }

        # 构建新的优化器状态（用当前param_groups，避免学习率/权重衰减不匹配）
        new_state_dict = {
            'state': filtered_state_dict,
            'param_groups': current_optim_dict['param_groups']
        }
        optimizer.load_state_dict(new_state_dict)

        # 5. 加载梯度缩放器
        gradient_scalar.load_state_dict(checkpoint['gradient_scalar_state_dict'])

        # 6. 仅主进程打印日志
        if rank == 0:
            logger.log(f'Loaded checkpoint from {resume_loc}')
            logger.log(f'Resuming training for epoch {start_epoch}')
    else:
        # 仅主进程打印缺失日志
        if rank == 0:
            logger.log(f"No checkpoint found at '{resume_loc}'")

    # 7. 分布式同步：确保所有进程加载完成后再继续
    dist_barrier_if_available(USE_FAKE_DDP)

    return model, optimizer, gradient_scalar, start_epoch, start_iteration, best_metric