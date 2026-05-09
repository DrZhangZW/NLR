import sys, time
import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist
from utils import logger
from typing import Optional, Dict, Union, Any, Tuple


class Statistics:
    def __init__(self, metric_names: Optional[list] = ['loss']) -> None:
        if len(metric_names) == 0:
            logger.error('Metric names list cannot be empty')
            raise ValueError("指标名称列表不能为空")
        # 初始化指标字典和计数器（仅处理数值型指标，如loss）
        self.metric_dict: Dict[str, Optional[float]] = {m: None for m in metric_names}
        self.metric_counters: Dict[str, int] = {m: 0 for m in metric_names}
        self.supported_metrics = metric_names
        self.round_places = 4  # 保留4位小数

        # 批次时间统计
        self.batch_time = 0.0
        self.batch_counter = 0

    def update(self, metric_vals: dict, batch_time: float, n: Optional[int] = 1) -> None:
        """
        更新指标统计（兼容DDP，累计全局聚合后的指标值）
        Args:
            metric_vals: 聚合后的指标字典（已通过tensor_to_python_float处理）
            batch_time: 批次加载时间
            n: 批次样本数（当前GPU处理的样本数）
        """
        for k, v in metric_vals.items():
            if k not in self.supported_metrics:
                continue

            # 累计指标值（已聚合所有GPU的数据）
            if self.metric_dict[k] is None:
                self.metric_dict[k] = v * n
            else:
                self.metric_dict[k] += v * n

            # 累计样本数
            self.metric_counters[k] += n

        # 累计批次时间
        self.batch_time += batch_time
        self.batch_counter += 1

    def avg_statistics_all(self, sep=": ") -> list:
        """计算所有指标的全局平均值（兼容DDP）"""
        metric_stats = []
        for k, v in self.metric_dict.items():
            if v is None or self.metric_counters[k] == 0:
                continue

            counter = self.metric_counters[k]
            v_avg = (v * 1.0) / counter

            # 四舍五入并格式化
            v_avg = round(v_avg, self.round_places)
            metric_stats.append("{:<}{}{:.4f}".format(k, sep, v_avg))

        return metric_stats

    def avg_statistics(self, metric_name: str) -> Optional[float]:
        """计算单个指标的全局平均值"""
        if metric_name not in self.supported_metrics:
            logger.warning(f"不支持的指标名称：{metric_name}")
            return None

        counter = self.metric_counters[metric_name]
        if counter == 0:
            return None

        v = self.metric_dict[metric_name]
        if v is None:
            return None

        avg_val = (v * 1.0) / counter

        return round(avg_val, self.round_places)

    def iter_summary(self,
                     epoch: int,
                     n_processed_samples: int,
                     total_samples: int,
                     elapsed_time: float,
                     learning_rate: Union[float, list]):
        """生成迭代过程的指标摘要"""
        metric_stats = self.avg_statistics_all()
        el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)

        # 格式化学习率
        if isinstance(learning_rate, float):
            lr_str = "LR: {:1.7f}".format(learning_rate)
        else:
            learning_rate = [round(lr, 7) for lr in learning_rate]
            lr_str = "LR: {}".format(learning_rate)

        # 格式化epoch和样本进度
        epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(epoch, n_processed_samples, total_samples)
        # 格式化平均批次加载时间
        batch_str = "Avg. batch load time: {:1.3f}".format(
            self.batch_time / self.batch_counter if self.batch_counter > 0 else 0.0
        )

        # 拼接摘要
        stats_summary = [epoch_str, lr_str]
        stats_summary.extend(metric_stats)
        stats_summary.append(batch_str)
        stats_summary.append(el_time_str)

        summary_str = ", ".join(stats_summary)
        return summary_str

    def epoch_summary(self, epoch: int, stage: Optional[str] = "Training"):
        metric_stats = self.avg_statistics_all(sep="=")
        metric_stats_str = " || ".join(metric_stats)
        logger.log('*** {} summary for epoch {}'.format(stage.title(), epoch))
        print("\t {}".format(metric_stats_str))
        sys.stdout.flush()
        return metric_stats_str
    def reset(self):
        """重置所有统计指标（epoch结束后调用）"""
        for m_name in self.supported_metrics:
            self.metric_dict[m_name] = None
            self.metric_counters[m_name] = 0
        self.batch_time = 0.0
        self.batch_counter = 0


def metric_monitor(
    loss: Union[Tensor, float, Dict[str, Union[Tensor, float]]],
    metric_names: list = ['loss'],
) -> Dict[str, Union[int, float]]:
    metric_vals = dict()

    if isinstance(loss, Dict):
        # 处理多Loss字典（如{'loss1': 0.1, 'loss2': 0.2}）
        for k, v in loss.items():
            if k in metric_names:
                metric_vals[k] = tensor_to_python_float(v)
    else:
        # 处理单个Loss（默认key为'loss'）
        if 'loss' in metric_names:
            metric_vals['loss'] = tensor_to_python_float(loss)


    return metric_vals
    # -------------------------- 彻底移除dist，仅保留张量转数值 --------------------------


def tensor_to_python_float(
        inp_tensor: Union[int, float, torch.Tensor],
) -> Union[int, float]:
    """
    极简版：仅做张量→Python数值转换，无任何分布式逻辑
    分布式聚合由外部处理，函数内部只负责基础类型转换
    """
    if isinstance(inp_tensor, torch.Tensor):
        # 仅处理单元素张量（Loss场景），无硬编码设备
        if inp_tensor.numel() != 1:
            raise ValueError(f"Loss张量必须是单元素标量，当前元素数：{inp_tensor.numel()}")
        return inp_tensor.item()  # 自动适配张量所在设备（CPU/GPU），无需手动转
    elif isinstance(inp_tensor, (int, float)):
        return float(inp_tensor)
    else:
        raise NotImplementedError(
            f"仅支持int/float/torch.Tensor类型，当前类型：{type(inp_tensor)}"
        )
