# ===================== 统一导入（删除重复 + 规范顺序） =====================
import os
import re
import random
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any
import glob
import torch
import torch.distributed as dist
import yaml
from torch import nn
from typing import List, Union,Dict,Optional, Tuple
import matplotlib.pyplot as plt
import cv2
# 项目路径定义
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# 格式定义
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'mat']  # 支持的图片后缀
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # 支持的视频后缀

# 导入自定义logger（utils.logger，无需替换）
from utils import logger


def read_simple_config(opts: argparse.Namespace) -> argparse.Namespace:
    """
    核心功能：读取YAML配置文件，解析条件表达式/自动转类型，合并到argparse的opts对象中
    输入：argparse解析后的opts对象（需包含common_config_file字段，即配置文件路径）
    输出：合并了YAML配置的最终opts对象（可直接用opt.xxx访问所有参数）
    优化点：全程无递归，避免递归深度不足问题；增加鲁棒性检查和详细注释
    """
    # ===================== 第一步：读取YAML配置文件（鲁棒性增强） =====================
    # 1.1 提取配置文件路径并转为Path对象（兼容Windows/Linux路径）
    yaml_path = Path(opts.common_config_file)

    # 1.2 读取YAML文件（用safe_load避免安全风险）
    with open(yaml_path, "r", encoding="utf-8") as f:
        # cfg是嵌套字典，结构和YAML完全一致（比如cfg["runtime"]["debug_mode"]）
        cfg: Dict[str, Any] = yaml.safe_load(f)
    # ===================== 第二步：解析核心参数（debug_mode） =====================
    # 2.1 提取debug_mode（所有条件表达式的核心依赖），默认设为False（防止键缺失）
    debug_mode: bool = cfg.get("runtime", {}).get("debug_mode", False)
    # 2.2 自动转换debug_mode类型（防止YAML里写的是字符串"true"）
    if isinstance(debug_mode, str):
        debug_mode = debug_mode.lower() == "true"

    # ===================== 第三步：工具函数（无递归） =====================
    def auto_convert(val: Any) -> Any:
        """
        非递归！自动转换值的类型：字符串→布尔/数值/列表，其他类型直接返回
        处理场景：
        - "true"/"false" → True/False
        - "123"/"1.23" → int/float
        - "[4,5]" → [4,5]
        - "null" → None
        """
        # 空值直接返回None
        if val is None:
            return None

        # 仅处理字符串类型，其他类型（int/float/list）直接返回
        if isinstance(val, str):
            val_strip = val.strip()  # 去除首尾空格（比如" [4,5] "→"[4,5]"）

            # 场景1：布尔值转换
            if val_strip.lower() == "true":
                return True
            elif val_strip.lower() == "false":
                return False

            # 场景2：null字符串转换
            elif val_strip.lower() == "null":
                return None

            # 场景3：数值转换（整数/浮点数）
            try:
                return int(val_strip) if "." not in val_strip else float(val_strip)
            except ValueError:
                pass  # 不是数值，继续处理其他场景

            # 场景4：列表字符串转换（非递归！仅拆分一层）
            if val_strip.startswith("[") and val_strip.endswith("]"):
                # 拆分列表元素（比如"4,5"→["4","5"]）
                elements = [e.strip() for e in val_strip[1:-1].split(",") if e.strip()]
                # 逐个转换元素类型（非递归，仅调用auto_convert处理基础类型）
                converted_elements = []
                for elem in elements:
                    converted_elements.append(auto_convert(elem))
                return converted_elements

        # 非字符串类型（如int/float/list/dict），直接返回
        return val

    def parse_if_expression(val: Any) -> Any:
        """
        非递归！解析YAML中的${if:${debug_mode},A,B}条件表达式
        逻辑：如果debug_mode为True，返回A；否则返回B（自动转换类型）
        """
        # 仅处理字符串类型的条件表达式，其他类型直接返回
        if not isinstance(val, str) or "${if:" not in val:
            return auto_convert(val)

        # 匹配debug_mode相关的条件表达式（比如${if:${debug_mode},0,8}）
        pattern = r"\$\{if:\$\{debug_mode\},([^,]+),([^}]+)\}"
        match = re.match(pattern, val.strip())
        if match:
            # 提取条件结果（A和B）
            val_debug, val_server = match.groups()
            # 根据debug_mode选择值，并转换类型
            return auto_convert(val_debug) if debug_mode else auto_convert(val_server)

        # 非debug_mode的条件表达式（暂不处理），直接转换类型返回
        return auto_convert(val)

    # ===================== 第四步：迭代式解析所有配置（无递归！） =====================
    # 4.1 用栈（stack）实现迭代式遍历，替代递归（彻底避免递归深度问题）
    # 栈元素格式：(当前字典, 父级键前缀)，初始栈包含根字典
    stack = [(cfg, "")]

    while stack:
        current_dict, parent_prefix = stack.pop()  # 弹出栈顶元素（后进先出）

        # 遍历当前字典的所有键值对
        for key, value in list(current_dict.items()):  # 用list避免遍历中修改字典报错
            # 子场景1：值是字典 → 压入栈，后续处理（替代递归）
            if isinstance(value, dict):
                # 拼接父级前缀（比如"runtime" + "vbs" → "runtime_vbs"）
                new_prefix = f"{parent_prefix}_{key}" if parent_prefix else key
                stack.append((value, new_prefix))  # 压入栈，后续迭代处理

            # 子场景2：值不是字典 → 解析条件表达式+转换类型，写回原字典
            else:
                current_dict[key] = parse_if_expression(value)

    # ===================== 第五步：伪DDP参数自动修正（避免train()阻塞） =====================
    # 5.1 提取runtime配置（防止键缺失）
    runtime_cfg = cfg.get("runtime", {})

    # 5.2 伪DDP模式强制修正参数（核心：避免多GPU/多进程导致的阻塞）
    if runtime_cfg.get("use_fake_ddp", False):
        runtime_cfg["use_ddp"] = False  # 伪DDP强制关闭真DDP
        # 伪DDP强制单GPU（用physical_gpu_id）
        # runtime_cfg["target_gpus"] = [runtime_cfg.get("physical_gpu_id", 0)]
        runtime_cfg["workers"] = 0  # 伪DDP关闭多进程加载（避免阻塞）
        runtime_cfg["persistent_workers"] = False  # 关闭常驻进程
        runtime_cfg["pin_memory"] = False  # 本地调试关闭页锁定内存

    # 把修正后的runtime写回原配置
    cfg["runtime"] = runtime_cfg

    # ===================== 第六步：迭代式扁平化配置并合并到opts（无递归！） =====================
    # 6.1 把argparse的opts转为字典（方便合并）
    opts_dict = vars(opts)

    # 6.2 再次用栈实现迭代式扁平化，把嵌套配置转为单层键（比如runtime_debug_mode）
    stack = [(cfg, "")]
    while stack:
        current_dict, parent_prefix = stack.pop()

        for key, value in current_dict.items():
            # 拼接当前键名（比如"runtime" + "debug_mode" → "runtime_debug_mode"）
            current_key = f"{parent_prefix}_{key}" if parent_prefix else key

            if isinstance(value, dict):
                # 子字典压入栈，后续处理
                stack.append((value, current_key))
            else:
                # 非字典值直接添加到opts_dict（合并配置）
                opts_dict[current_key] = value

    # ===================== 第七步：返回最终opts对象 =====================
    # 把合并后的字典转回argparse.Namespace对象（保持和原有代码兼容）
    final_opts = argparse.Namespace(**opts_dict)

    return final_opts


# ===================== 多卡[4,5]场景测试说明 =====================
"""
服务器多卡[4,5]运行命令示例：
torchrun --nproc_per_node=2 --master_port=29500 your_script.py

环境变量自动设置（torchrun）：
- 进程0：RANK=0, WORLD_SIZE=2, LOCAL_RANK=0, CUDA_VISIBLE_DEVICES=4,5
- 进程1：RANK=1, WORLD_SIZE=2, LOCAL_RANK=1, CUDA_VISIBLE_DEVICES=4,5

预期输出（进程0）：
================================================================================
🚀 服务器DDP（多卡[4,5]）初始化完成
全局Rank: 0 (预期：0) (环境变量RANK=0)
进程总数: 2 (预期：2) (环境变量WORLD_SIZE=2)
本地GPU逻辑ID: 0 (预期：0) (环境变量LOCAL_RANK=0)
物理GPU ID: 4 (预期：4) (环境变量CUDA_VISIBLE_DEVICES=4,5)
GPU名称: NVIDIA A100-SXM4-80GB (预期：GPU4名称)
是否主进程: True (预期：True) | 进程PID: 12345
系统检测GPU数: 8 (预期：≥2)
================================================================================
✅ 单节点多卡模式 | 使用GPU：物理卡4 (预期：4)（目标GPU：[4,5] (预期：[4,5])）
✅ CUDNN enabled (多卡确定性模式)
"""


# ======================== DDP初始化（纯初始化逻辑，无验证调用，避免闭环） ========================
def init_ddp(opts, sim_rank: int = None):
    """
    整合版DDP初始化（支持伪DDP模拟多rank + 真实DDP）
    核心优化：
    1. 新增sim_rank参数，统一伪DDP/真实DDP的初始化入口
    2. 复用原有映射逻辑，消除device_setup中的冗余代码
    3. 严格区分伪DDP/真实DDP的变量计算逻辑

    :param opts: 配置对象
    :param sim_rank: 伪DDP模拟的rank（None=真实DDP，整数=伪DDP）
    :return: rank, world_size, local_gpu_id（逻辑ID）, physical_gpu_id（映射的物理ID）
    """
    # ========== 1. 提取所有公共逻辑（伪DDP/真实DDP共用） ==========
    is_fake_ddp = getattr(opts, "runtime_use_fake_ddp", True)
    target_gpus = getattr(opts, "runtime_target_gpus", [4, 5])
    master_addr = getattr(opts, "runtime_master_addr", "127.0.0.1")
    master_port = getattr(opts, "runtime_master_port", 29501)
    runtime_nproc = getattr(opts, "runtime_nproc", None)

    # 处理CUDA_VISIBLE_DEVICES：优先级 环境变量 > 配置的物理卡
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, target_gpus)))
    physical_gpu_ids = list(map(int, cuda_visible.split(",")))

    # ========== 2. 本地单卡适配（伪DDP专用） ==========
    if torch.cuda.is_available() and torch.cuda.device_count() == 1 and is_fake_ddp:
        target_gpus = physical_gpu_ids   # 对齐可见GPU，避免world_size不匹配
        logger.info(f"🔧 本地单卡伪DDP模式，自动对齐target_gpus={target_gpus}")

    # ========== 3. 初始化核心变量 ==========
    rank, world_size, local_gpu_id, physical_gpu_id = 0, 0, 0, -1

    # ========== 4. 伪DDP逻辑（sim_rank≠None） ==========
    if sim_rank is not None:
        # 伪DDP：模拟分布式参数 + 本地GPU设置
        rank = sim_rank
        # 核心：world_size = 模拟的总卡数（4），而非物理GPU数（1）
        world_size = runtime_nproc if runtime_nproc is not None else len(target_gpus)
        local_gpu_id = sim_rank  # 伪DDP下local_gpu_id = 模拟rank

        # 映射模拟rank到目标物理GPU ID（边界防护）
        physical_gpu_id = target_gpus[sim_rank] if (0 <= sim_rank < len(target_gpus)) else target_gpus[0]

        # 伪DDP：绑定本地物理GPU 0（单卡模拟）
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logger.info(f"🎭 伪DDP模式 | 模拟Rank={rank} | 逻辑ID={local_gpu_id} | 映射物理ID={physical_gpu_id}")
    else:
        # 真实DDP：仅读取环境变量，不调用验证函数
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_gpu_id = int(os.environ["LOCAL_RANK"])

        # 映射逻辑ID到物理ID（LOCAL_RANK=0→物理4，LOCAL_RANK=1→物理5）
        physical_gpu_id = physical_gpu_ids[local_gpu_id] if (0 <= local_gpu_id < len(physical_gpu_ids)) else -1

        torch.cuda.set_device(local_gpu_id)  # 绑定逻辑卡→映射到4/5物理卡
        device = torch.device(f"cuda:{local_gpu_id}")

        assert torch.cuda.current_device() == local_gpu_id, \
            f"GPU绑定失败！当前设备={torch.cuda.current_device()}，期望={local_gpu_id}"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
            #device_id= device   # 关键新增：指定当前进程绑定的逻辑GPU ID
        )
        # 6. 同步所有进程（显式指定device_ids，兜底防护）
        dist.barrier(device_ids=[local_gpu_id])
        if rank==0:
            logger.info(f"📌 真实DDP模式 - Rank={rank} 绑定GPU逻辑ID={local_gpu_id}（物理ID={physical_gpu_id}）")
    # ========== 3. 公共日志输出（仅主进程/伪DDP输出，避免多进程刷屏） ==========

    if rank == 0:
        mode = "本地伪DDP（单卡模拟多卡）" if is_fake_ddp else "服务器真实DDP（多卡分布式）"
        gpu_name = torch.cuda.get_device_name(local_gpu_id) if (
                    torch.cuda.is_available() and local_gpu_id >= 0) else "CUDA不可用"

        logger.info("=" * 80)
        logger.info(f"🚀 {mode} 初始化完成")
        logger.info(f"Rank: {rank} | 总进程数: {world_size}")
        logger.info(f"逻辑GPU ID: {local_gpu_id} | 映射物理GPU ID: {physical_gpu_id}")
        logger.info(f"可见物理GPU列表: {physical_gpu_ids} | DDP通信端口: {master_port}")
        logger.info(f"当前GPU名称: {gpu_name}")
        logger.info("=" * 80)
    return rank, world_size, local_gpu_id, physical_gpu_id


# ======================== 设备初始化入口（严格控制调用链路） ========================
def device_setup(opts,sim_rank: int = None):
    """
    设备初始化总入口（支持伪DDP模拟多卡）
    【递归防护】：
    1. 先完成所有初始化，再调用验证
    2. 仅主进程执行验证，且仅执行一次
    3. 验证函数不反向调用本函数

    :param opts: 配置对象
    :param sim_rank: 伪DDP模拟的rank（None=真实DDP，整数=伪DDP）
    :return: 初始化后的opts
    """
    # ========== 第一步：调用init_ddp完成核心ID初始化（统一入口） ==========
    if torch.cuda.is_available():
        # 核心优化：所有ID计算都交给init_ddp，device_setup仅做后续处理
        rank, world_size, local_gpu_id, physical_gpu_id = init_ddp(opts, sim_rank=sim_rank)
    else:
        logger.warning("⚠️ CUDA不可用，使用CPU")
    # ========== 第二步：用真实的local_gpu_id初始化种子（核心修复） ==========
    random_seed = getattr(opts, "common_seed", 0)
    # 每个进程专属种子 = 基础种子 + local_gpu_id（伪DDP用sim_rank，真实DDP用真实local_rank）
    final_seed = random_seed + local_gpu_id
    # 初始化随机种子（分布式下每个进程种子不同）
    torch.manual_seed(final_seed)  # 当前进程CPU专属种子
    random.seed(final_seed)  # Python内置random模块
    np.random.seed(final_seed)  # NumPy模块



    # ========== 仅补充这1行！和epoch更新逻辑对齐 ==========
    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)  # 新增：给当前GPU设种子（和epoch更新逻辑一致）
        torch.cuda.manual_seed_all(random_seed)  # 保留原有：给所有GPU设基础种子（兜底）
        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(f'📌 随机种子已设置: 基础={random_seed} | 进程专属={final_seed} (多卡差异化)')

    # ========== 第三步：后续CUDA配置 + 变量保存 ==========
    device = torch.device("cpu")
    num_valid_gpus = 0
    is_simulate_multi_gpu = False
    valid_gpus = []
    logical_gpu_id_final = -1
    if torch.cuda.is_available():
        target_gpus = getattr(opts, "runtime_target_gpus")
        # 多卡模拟判断：伪DDP强制开启 | 真实场景单卡多目标
        is_simulate_multi_gpu = (sim_rank is not None)
        # 过滤有效GPU
        if is_simulate_multi_gpu:
            valid_gpus = [0] * len(target_gpus)
            num_valid_gpus = len(target_gpus)
            logical_gpu_id_final = 0  # 强制绑定本地GPU 0
            if rank == 0:
                logger.warning(f'=== ⚠️ 单卡模拟多卡[{target_gpus}]模式 === 映射到GPU {logical_gpu_id_final}')
        else:
            # 真实DDP：物理卡→逻辑卡映射（核心修改）
            def physical2logical(pid):
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") # visible = ['3','4','5','6']
                return visible.index(str(pid)) if str(pid) in visible else pid # visible.index('3') → 0
            valid_gpus = [physical2logical(g) for g in target_gpus] # [3,4,5,6]-->[0,1,2,3]
            num_valid_gpus = len(valid_gpus)
            #logical_gpu_id_final = local_gpu_id  # 使用真实逻辑ID
            logical_gpu_id_final = valid_gpus[local_gpu_id % num_valid_gpus] if num_valid_gpus > 0 else local_gpu_id

        # 最终设备绑定
        device = torch.device(f'cuda:{logical_gpu_id_final}')
        # CUDNN配置（确定性模式）
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = not getattr(opts, "runtime_debug_mode", True)
            torch.backends.cudnn.deterministic = True
            if rank == 0:
                logger.info('✅ CUDNN enabled (多卡确定性模式)')

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ========== 第四步：保存核心变量到opts（统一命名） ==========
    core_vars = {
        "dev_device": device,                    # 最终绑定的设备
        "dev_rank": rank,                        # 进程rank
        "dev_world_size": world_size,            # 总进程数
        "dev_local_gpu_id": local_gpu_id,        # 本地逻辑GPU ID
        "dev_physical_gpu_id": physical_gpu_id,  # 映射的物理GPU ID
        "dev_logical_gpu_id_final": logical_gpu_id_final,  # 最终绑定的逻辑ID
        "dev_num_gpus": num_valid_gpus,          # 有效GPU数量
        "dev_target_gpus": valid_gpus,           # 有效物理GPU列表
        "dev_simulate_multi_gpu": is_simulate_multi_gpu,  # 是否单卡模拟多卡
        "dev_is_master_node": (rank == 0),       # 是否主进程
        "dev_is_fake_ddp": (sim_rank is not None or getattr(opts, "runtime_use_fake_ddp", True))  # 是否伪DDP
    }
    for k, v in core_vars.items():
        setattr(opts, k, v)

    if rank == 0 and torch.cuda.is_available():
        SEP = "=" * 60
        logger.info(f"\n{SEP}")
        logger.info("📌 CUDA 环境配置验证（仅主进程输出）")
        logger.info(SEP)

        # 1. 核心环境变量检查（保留关键项）
        logger.info("\n【1】环境变量 & 基础ID")
        logger.info(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        logger.info(f"   LOCAL_RANK: {os.environ.get('LOCAL_RANK', '未设置')} | local_gpu_id: {local_gpu_id}")
        logger.info(f"   逻辑GPU ID: {logical_gpu_id_final} | 物理GPU ID: {physical_gpu_id}")

        # 2. CUDA基础信息（精简冗余项）
        logger.info("\n【2】CUDA 基础信息")
        logger.info(f"   CUDA可用: {torch.cuda.is_available()} | 可见GPU数: {torch.cuda.device_count()}")
        logger.info(
            f"   当前设备: {device} | 是否为CUDA: {device.type == 'cuda'} | 设备索引: {device.index if device.type == 'cuda' else 'N/A'}")

        # 3. GPU设备映射验证（精简循环输出，仅保留核心信息）
        logger.info("\n【3】GPU 设备映射")
        bound_id = torch.cuda.current_device()
        bound_name = torch.cuda.get_device_name(bound_id)
        logger.info(f"   全局绑定GPU: 逻辑ID {bound_id} ({bound_name})")
        # 仅遍历可见GPU，输出逻辑ID→物理ID映射（适配任意CUDA_VISIBLE_DEVICES）
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        for logical_id in range(torch.cuda.device_count()):
            phys_id = visible_devices[logical_id] if logical_id < len(visible_devices) else "未知"
            gpu_name = torch.cuda.get_device_name(logical_id)
            logger.info(f"   逻辑ID {logical_id} → 物理ID {phys_id}: {gpu_name}")

        # 4. 张量分配测试（增加异常捕获，避免崩溃）
        logger.info("\n【4】张量分配测试")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        logger.info(f"   ✅ 成功在 {device} 创建张量 | 张量设备: {x.device}")

        # 5. 最终设备校验（精简输出）
        logger.info("\n【5】最终设备校验")
        logger.info(f"   当前CUDA设备ID: {bound_id} | 名称: {bound_name}")
        if device.type == 'cuda':
            logger.info(f"   设备匹配度: {'✅ 匹配' if device.index == bound_id else '❌ 不匹配'}")

        logger.info(f"{SEP}\n")

    return opts

def log_gpu_memory(rank, gpu_id):
    """输出GPU显存使用（替换原print_gpu_memory，用logger）"""
    if not torch.cuda.is_available():
        logger.info(f"[Rank {rank}] GPU不可用，跳过显存查询")
        return

    # 多进程下：主进程打印，或当前进程打印自己的显存（避免重复）
    if rank == 0:
        mem_alloc_peek = torch.cuda.max_memory_allocated(gpu_id) / 1024 / 1024
        mem_reserved_peek = torch.cuda.max_memory_reserved(gpu_id) / 1024 / 1024
        logger.info(f"[Rank 0] GPU显存峰值 - 已分配: {mem_alloc_peek:.2f}MB | 已预留: {mem_reserved_peek:.2f}MB")
        # 重置峰值统计（可选，用于按epoch监控）


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

import datetime

def print_log(msg, rank=None, is_main_only=False):
    """极致极简版（仅保留时间戳+Rank+消息）"""
    # 自动获取rank
    if rank is None:
        rank = torch.distributed.get_rank() if (
                    torch.distributed.is_available() and torch.distributed.is_initialized()) else 0

    # 主进程控制
    if is_main_only and rank != 0:
        return

    # 核心输出
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [Rank {rank}] {msg}", flush=True)

def crop_valid_regions(tensor, mask):
    """
        根据mask裁剪张量的有效区域，返回统一尺寸的批处理张量（全GPU执行）

        Args:
            tensor: 输入张量 [bs, c, H, W]（GPU张量）
            mask: 掩码张量 [bs, H, W]（GPU张量），True表示填充区域
        Returns:
            裁剪后的张量 [bs, c, h, w]（GPU张量），其中h和w是批处理中最大的有效区域尺寸
        """
    bs, c, H, W = tensor.shape
    device = tensor.device  # 统一获取GPU设备，避免硬编码

    # 存储每个样本的有效区域边界
    valid_regions = []

    # 计算每个样本的有效区域边界
    for i in range(bs):
        # 反转mask：True表示填充区域，False表示有效区域
        valid_mask = ~mask[i]  # 有效区域为True

        # 检查是否有有效区域
        if not valid_mask.any():
            valid_regions.append(None)  # 表示无有效区域
            continue

        # 获取有效区域边界
        rows = torch.any(valid_mask, dim=1)  # 每行是否有有效像素
        cols = torch.any(valid_mask, dim=0)  # 每列是否有有效像素

        # 找到有效行和列的索引
        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]

        # 检查索引是否为空
        if row_indices.numel() == 0 or col_indices.numel() == 0:
            valid_regions.append(None)
            continue

        # 🔥 关键优化：避免.item()传输到CPU，直接在GPU上计算边界（仅最后用item()取标量）
        y_min = row_indices[0]
        y_max = row_indices[-1]
        x_min = col_indices[0]
        x_max = col_indices[-1]

        # 存储GPU张量形式的边界（后续计算max_h/max_w时仍在GPU）
        valid_regions.append((y_min, y_max, x_min, x_max))

    # 🔥 关键优化：全程在GPU上计算最大尺寸（避免CPU/GPU来回传）
    max_h = torch.tensor(0, device=device)
    max_w = torch.tensor(0, device=device)

    for region in valid_regions:
        if region is not None:
            y_min, y_max, x_min, x_max = region
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            max_h = torch.max(max_h, h)  # GPU上取最大值
            max_w = torch.max(max_w, w)

    max_h_scalar = max_h.item()
    max_w_scalar = max_w.item()
    # 创建结果张量（GPU张量）
    result = torch.zeros(bs, c, max_h_scalar, max_w_scalar, device=device)
    # 填充每个样本的有效区域
    for i in range(bs):
        if valid_regions[i] is not None:
            y_min, y_max, x_min, x_max = valid_regions[i]
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            result[i, :, :h, :w] = tensor[i, :, y_min:y_max + 1, x_min:x_max + 1]

    return result


def separate_parameters2(
        model: nn.Module,
        keywords: Union[str, List[Union[str, List[str]]]]  # 支持字符串或列表（含嵌套列表）
) -> (Dict[str, List[torch.Tensor]], Dict[str, List[str]]):
    """
    从PyTorch模型中分离参数，按关键词分类（列表关键词需同时满足所有条件）

    Args:
        model: PyTorch模型
        keywords: 关键词或关键词列表，支持两种格式：
            - 字符串：单个关键词匹配（参数名包含该关键词即可）
            - 列表：多个关键词必须同时满足（参数名需包含所有关键词）

    Returns:
        Tuple[Dict, Dict]:
            第一个字典：按关键词分类的参数（key为关键词/组合关键词，value为参数列表）
            第二个字典：按关键词分类的参数名（key为关键词/组合关键词，value为参数名列表）
            均包含"base"类别（未匹配到任何关键词的参数）
    """
    matched_params = {}
    matched_names = {}

    # 标准化关键词：确保输入是列表格式
    if not isinstance(keywords, list):
        keywords = [keywords]

    # 处理关键词：为嵌套列表生成唯一key，初始化字典
    processed_keywords = []
    for kw in keywords:
        if isinstance(kw, list):
            # 列表关键词：用"_AND_"连接生成key，体现同时满足逻辑
            kw_key = "_AND_".join(kw)
            processed_keywords.append((kw, kw_key))
        else:
            # 单个关键词：直接用自身作为key
            processed_keywords.append((kw, kw))

    # 初始化参数和参数名字典（包含所有关键词和base）
    for _, kw_key in processed_keywords:
        matched_params[kw_key] = []
        matched_names[kw_key] = []
    matched_params["base"] = []
    matched_names["base"] = []

    # 计算需要更新的参数总数（可训练参数）
    num_update = sum(1 for p in model.parameters() if p.requires_grad)
    matched_count = 0

    # 分离参数核心逻辑
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数

        is_matched = False
        for kw, kw_key in processed_keywords:
            if isinstance(kw, list):
                # 列表关键词：必须同时包含所有关键词才匹配
                if all(sub_kw in name for sub_kw in kw):
                    matched_params[kw_key].append(param)
                    matched_names[kw_key].append(name)
                    is_matched = True
                    break
            else:
                # 单个关键词：包含关键词即匹配
                if kw in name:
                    matched_params[kw_key].append(param)
                    matched_names[kw_key].append(name)
                    is_matched = True
                    break

        # 未匹配到任何关键词，放入base类别
        if not is_matched:
            matched_params["base"].append(param)
            matched_names["base"].append(name)
        matched_count += 1

    # 校验：匹配的参数数量必须等于可训练参数总数
    assert matched_count == num_update, \
        f"Parameter count mismatch: matched {matched_count}, expected {num_update}"

    return matched_params, matched_names

def separate_parameters(
        model: nn.Module,
        keywords: Union[str, List[str]]
) -> Dict[str, List[torch.Tensor]]:
    """
    从PyTorch模型中分离参数，按关键词分类

    Args:
        model: PyTorch模型
        keywords: 关键词或关键词列表

    Returns:
        Dict: 按关键词分类的参数字典，包含base类别
    """
    matched_params = {}
    matched_names = {}
    # 标准化关键词并初始化字典
    if isinstance(keywords, str):
        keywords = [keywords]

    # 为每个关键词初始化空列表，并添加base类别
    for keyword in keywords:

        matched_params[keyword] = []
        matched_names[keyword] = []
    matched_params["base"] = []
    matched_names["base"] = []
    # 计算需要更新的参数总数
    num_update = sum(1 for p in model.parameters() if p.requires_grad)

    # 分离参数
    matched_count = 0
    for name, param in model.named_parameters():
        #print(name)
        if not param.requires_grad:
            continue

        is_matched = False
        for keyword in keywords:
            if keyword in name:
                matched_params[keyword].append(param)
                matched_names[keyword].append(name)
                is_matched = True
                matched_count += 1
                break  # 匹配到一个关键词后就跳出，避免重复匹配

        # 如果没有匹配到任何关键词，放入base类别
        if not is_matched:
            matched_params["base"].append(param)
            matched_names["base"].append(name)
            matched_count += 1

    # 正确的断言检查：匹配的参数数量应该等于需要更新的参数数量
    assert matched_count == num_update, f"Parameter count mismatch: matched {matched_count}, expected {num_update}"

    return matched_params,matched_names


def create_multi_lr_optimizer(param_dicts, learning_rates, beta1=0.9, beta2=0.98, weight_decay=0.001):
    """
    为不同参数组设置不同学习率

    Args:
        param_dicts: 参数字典
        learning_rates: 学习率字典
        beta1: AdamW参数
        beta2: AdamW参数
        weight_decay: 权重衰减
    """
    # 构建参数组列表
    param_groups = []

    for key, params in param_dicts.items():
        if key in learning_rates and len(params) > 0:
            param_groups.append({
                'params': params,
                'lr': learning_rates[key],
                'weight_decay': weight_decay,
                'name': key
            })

    optimizer = torch.optim.AdamW(param_groups, betas=(beta1, beta2),weight_decay=weight_decay)
    return optimizer


def load_flist(flist):
    f = []
    path = flist
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        else:
            raise Exception(f'{p} does not exist')

    f2 = []
    f2 += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    return f2
def img2label(list, path, suffix):
    f = []
    f += [path + os.sep + str(Path(x).stem) + suffix for x in list]
    #f += ['{}/{}'.format(path, str(Path(x).stem) + suffix) for x in list]
    return f
import torch.nn.functional as F



def adjust_tensor_ratio(tensor, target_ratio=4/3, method='crop'):
    """
    调整张量宽高比（纯PyTorch GPU实现，批量化处理）

    Args:
        tensor: 输入张量 (BS, C, H, W) → GPU张量
        target_ratio: 目标宽高比 (width/height)
        method: 调整方法 'crop'=中心裁剪, 'resize'=缩放填充, 'compress'=压缩长轴

    Returns:
        调整后的张量 (BS, C, new_H, new_W) → GPU张量
    """
    bs, c, h, w = tensor.shape
    current_ratio = w / h

    # 如果当前比例接近目标比例，直接返回（避免无意义计算）
    if abs(current_ratio - target_ratio) < 0.1:
        return tensor
    if method == 'crop':
        # 中心裁剪（纯GPU张量切片，无CPU操作）
        if current_ratio > target_ratio:
            # 图像太宽，裁剪宽度
            new_width = int(h * target_ratio)
            start_x = (w - new_width) // 2
            # 批量化切片：对所有样本统一裁剪
            result = tensor[:, :, :, start_x:start_x + new_width]
        else:
            # 图像太高，裁剪高度
            new_height = int(w / target_ratio)
            start_y = (h - new_height) // 2
            result = tensor[:, :, start_y:start_y + new_height, :]

    elif method == 'resize':
        # 缩放填充（等价原cv2.resize + copyMakeBorder，纯GPU实现）
        if current_ratio > target_ratio:
            # 先缩放宽度到目标比例，再填充高度
            new_width = int(h * target_ratio)
            scale_factor = new_width / w
            temp_height = int(h * scale_factor)
            temp_width = new_width

            # Step1: 等比例缩放（GPU版cv2.resize）
            resized = F.interpolate(
                tensor,
                size=(temp_height, temp_width),
                mode='bilinear',  # 等价cv2.resize的双线性插值
                align_corners=False
            )

            # Step2: 上下填充黑色边框（GPU版cv2.copyMakeBorder）
            pad_top = (h - temp_height) // 2
            pad_bottom = h - temp_height - pad_top
            # PyTorch的pad格式：(左, 右, 上, 下)
            result = F.pad(
                resized,
                pad=(0, 0, pad_top, pad_bottom),  # 仅上下填充
                mode='constant',
                value=0.0  # 黑色填充，等价[0,0,0]
            )

    elif method == 'compress':
        # 压缩长轴（保持短轴不变，压缩长轴，纯GPU实现）
        if current_ratio > target_ratio:
            # 图像太宽：保持高度不变，压缩宽度到目标比例
            new_height = h
            new_width = int(h * target_ratio)
        else:
            # 图像太高：保持宽度不变，压缩高度到目标比例
            new_width = w
            new_height = int(w / target_ratio)

        # GPU版cv2.resize（批量化缩放）
        result = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )

    return result


def process_kitti_tensors(clean_tensor, depth_tensor, target_ratio=4/3, method='crop'):
    """
    处理KITTI数据集张量（纯GPU执行，无CPU/GPU互传）

    Args:
        clean_tensor: 清洁图像张量 (BS, C, H, W) → GPU张量
        depth_tensor: 深度图像张量 (BS, C, H, W) → GPU张量
        target_ratio: 目标宽高比 (width/height)
        method: 调整方法 'crop'/'resize'/'compress'

    Returns:
        处理后的清洁图像和深度图像张量 → 均为GPU张量
    """
    clean_adjusted = adjust_tensor_ratio(clean_tensor, target_ratio, method)
    depth_adjusted = adjust_tensor_ratio(depth_tensor, target_ratio, method)
    return clean_adjusted, depth_adjusted


def resize_if_needed(tensor_list, max_size=480):
    """
    缩放张量到指定最大尺寸（优化版：适配批量/多设备/性能）

    Args:
        tensor_list: 张量列表，支持 [C, H, W] 或 [BS, C, H, W] 格式（GPU张量）
        max_size: 最大尺寸阈值

    Returns:
        resized_tensors: 缩放后的张量列表（与输入同设备、同格式）
    """
    resized_tensors = []
    for tensor in tensor_list:
        # 1. 兼容批量张量 [BS, C, H, W] 和单张张量 [C, H, W]
        if len(tensor.shape) == 3:  # [C, H, W]
            h, w = tensor.shape[-2:]
            is_batch = False
        elif len(tensor.shape) == 4:  # [BS, C, H, W]
            h, w = tensor.shape[-2:]
            is_batch = True
        # 2. 无需缩放则直接返回（保留原设备）
        if h <= max_size and w <= max_size:
            resized_tensors.append(tensor)
            continue

        # 3. 计算缩放比例（纯GPU/CPU标量计算，无开销）
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 4. 优化插值逻辑：避免不必要的unsqueeze/squeeze（批量张量直接处理）
        if is_batch:
            resized_tensor = F.interpolate(
                tensor,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
        else:
            resized_tensor = F.interpolate(
                tensor.unsqueeze(0),  # 加batch维度
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # 删batch维度

        resized_tensors.append(resized_tensor)

    return resized_tensors



def make_divisible(v: Union[float, int], divisor: Optional[int] = 8,
                   min_value: Optional[Union[float, int]] = None, ) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def show_tensor(tensor, title="Tensor", max_channels=9, cmap=None):
    """
    可视化NxCxHxW格式的张量
    参数:
        tensor: 输入张量(支持CPU/GPU)
        title: 图像标题
        max_channels: 最大显示通道数
        cmap: 单通道使用的颜色映射 'viridis'
    """
    # 确保张量在CPU上并转为float类型
    tensor = tensor.detach().cpu().float()

    # 处理不同维度输入
    if tensor.dim() == 2:  # HxW
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    elif tensor.dim() == 3:  # CxHxW
        tensor = tensor.unsqueeze(0)  # 1xCxHxW

    n, c, h, w = tensor.shape

    # 归一化处理
    #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    # 创建子图网格
    cols = min(c, max_channels)
    rows = n
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # 处理单样本情况
    if rows == 1:
        axs = axs[np.newaxis, :] if cols > 1 else [axs]

    for i in range(rows):
        for j in range(cols):
            # 获取当前子图轴
            ax = axs[i, j] if cols > 1 else axs[i]

            # 显示图像
            img = tensor[i, j].numpy()
            if c >= 3 and j < 3:  # 前3通道作为RGB
                ax.imshow(img, cmap=None)
            else:  # 其他通道使用指定cmap
                ax.imshow(img, cmap=cmap)

            ax.set_title(f'B{i}C{j}', fontsize=8)
            ax.axis('off')

    plt.suptitle(f"{title} (Batch:{n}, Channels:{min(c, max_channels)}/{c})")
    plt.tight_layout()
    plt.show(block=True)
def visualize_tensor(tensor: torch.Tensor, title: str = "Tensor Visualization"):
    """
    超高分辨率显示 4D tensor (NxCxHxW)
    完全不模糊，保持图像原始清晰度
    """
    assert tensor.dim() == 4, "Input must be 4D tensor (NxCxHxW)"
    n, c, h, w = tensor.shape
    assert c in [1, 3], "Channel must be 1 (grayscale) or 3 (RGB)"

    tensor = tensor.detach().cpu().float()

    # 计算行列
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # ======================
    # 🔥 核心：DPI 必须在这里设置！
    # ======================


    # ======================
    # 🔥 每个子图给足够大的空间
    # ======================
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if n == 1:
        axes = np.array([axes])

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            img = tensor[idx].permute(1, 2, 0).numpy()
            if c == 1:
                img = img.squeeze(-1)

            # ======================
            # 🔥 禁止插值模糊！
            # ======================
            ax.imshow(img, cmap='gray' if c == 1 else None, interpolation='none')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=10)
    plt.tight_layout(pad=0.5)
    plt.show(block=True)



def batch_psnr(pred, target, data_range=1.0):
    """
    计算批次图像的PSNR（
支持2x3xhxw输入）

    参数:
        pred (Tensor): 预测图像组 [2,3,h,w]
        target (Tensor): 目标图像组 [2,3,h,w]
        data_range (float): 像素值范围(默认1.0)
    """
    mse = torch.mean((pred - target) ** 2, dim=[2, 3])  # 计算每幅图的MSE
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return torch.mean(psnr)  # 返回批次平均PSNR

def validate_and_fix_paths(source_path, target_path):
    """
    验证并自动修正图像和深度图路径名称一致性

    参数:
        image_path: 图像文件路径
        depth_path: 深度文件路径

    返回:
        tuple: (修正后的深度图路径, 是否进行了修正)
    """
    image_name = os.path.splitext(os.path.basename(source_path))[0]
    depth_name = os.path.splitext(os.path.basename(target_path))[0]
    if image_name == depth_name:
        return target_path
    else:
        depth_dir = os.path.dirname(target_path)
        depth_ext = os.path.splitext(target_path)[1]
        fixed_path = os.path.join(depth_dir, f"{image_name}{depth_ext}")
        #shutil.move(target_path, fixed_path)
        return fixed_path


from typing import List
def join_paths(root: str, subs: Union[str, List[str], Tuple[str]]) -> List[str]:
    """支持字符串/列表/元组的路径拼接工具"""
    if isinstance(subs, str):
        return [os.path.normpath(os.path.join(root, subs.lstrip('/')))]
    return [os.path.normpath(os.path.join(root, p.lstrip('/'))) for p in subs]

# ======================== 核心：伪DDP包装器（本地调试用） ========================

class FakeDDP:
    """模拟DDP接口的伪包装器，完全脱离分布式依赖"""

    def __init__(self, model, device_ids=None, output_device=None, find_unused_parameters=False):
        self.module = model  # 保留DDP的.module属性
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters
    def __call__(self, *args, **kwargs):
        """支持model(inputs)调用方式（核心修复not callable问题）"""
        return self.module(*args, **kwargs)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        """模拟参数迭代"""
        return self.module.parameters()

    def named_parameters(self):
        """模拟命名参数迭代"""
        return self.module.named_parameters()

    def zero_grad(self, set_to_none=False):
        """模拟梯度清零"""
        self.module.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """模拟状态字典"""
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        """模拟加载状态字典"""
        self.module.load_state_dict(state_dict)


# 先添加这2行，消除tkagg警告（放在文件开头，import区域）
import time
from collections import defaultdict
from utils import misc
# 保留修正后的FLOPsCounter类（兼容池化层kernel_size整数/元组）
class FLOPsCounter:
    """自定义FLOPs统计类（PyTorch原生，无第三方依赖）- 兼容池化层kernel_size整数/元组"""
    def __init__(self):
        self.flops = defaultdict(int)
        self.handles = []

    def _conv_flops_hook(self, module, input, output):
        batch_size = input[0].size(0)
        in_c = module.in_channels
        out_c = module.out_channels
        k_h, k_w = module.kernel_size
        out_h, out_w = output.size()[2:]
        groups = module.groups if hasattr(module, 'groups') else 1
        flops = batch_size * out_c * out_h * out_w * (k_h * k_w * in_c // groups)
        if module.bias is not None:
            flops += batch_size * out_c * out_h * out_w
        self.flops[module.__class__.__name__] += flops

    def _linear_flops_hook(self, module, input, output):
        batch_size = input[0].size(0)
        in_feat = module.in_features
        out_feat = module.out_features
        flops = batch_size * in_feat * out_feat
        if module.bias is not None:
            flops += batch_size * out_feat
        self.flops[module.__class__.__name__] += flops

    def _bn_flops_hook(self, module, input, output):
        flops = input[0].numel() * 4
        self.flops[module.__class__.__name__] += flops

    def _pool_flops_hook(self, module, input, output):
        # 兼容整数/元组kernel_size
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        k_h, k_w = kernel_size
        flops = output.numel() * k_h * k_w
        self.flops[module.__class__.__name__] += flops

    def _activate_flops_hook(self, module, input, output):
        flops = input[0].numel()
        self.flops[module.__class__.__name__] += flops

    def register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv_flops_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_flops_hook))
            elif isinstance(module, nn.BatchNorm2d):
                self.handles.append(module.register_forward_hook(self._bn_flops_hook))
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                self.handles.append(module.register_forward_hook(self._pool_flops_hook))
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh)):
                self.handles.append(module.register_forward_hook(self._activate_flops_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def get_total_flops(self):
        return sum(self.flops.values())

# ======================== 最终修正版：compute_model_metrics_native 函数 ========================
def compute_model_metrics_native(model, device, img_h=320, img_w=320, batch_size=1, test_times=100):
    """
    纯PyTorch原生API计算4个核心指标（无第三方依赖）
    适配NLRNET_Plus.forward(x0, y0, phase) 3参数+NestedTensor输入
    已精简：无冗余forward，只跑必要次数
    """
    # 1. 构造输入（只构造一次）
    dummy_x0 = [torch.randn(3, img_h, img_w, dtype=torch.float32).to(device, non_blocking=True) for _ in range(batch_size)]
    dummy_y0 = [torch.randn(3, img_h, img_w, dtype=torch.float32).to(device, non_blocking=True) for _ in range(batch_size)]
    dummy_x0 = misc.nested_tensor_from_tensor_list(dummy_x0)
    dummy_y0 = misc.nested_tensor_from_tensor_list(dummy_y0)
    dummy_x0 = dummy_x0.to(device, non_blocking=True)
    dummy_y0 = dummy_y0.to(device, non_blocking=True)
    dummy_phase = 2

    # 2. 参数量（不用跑模型）
    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 10**6, 1)

    # ---------------------- 关键优化 ----------------------
    # 只跑一次前向，同时完成：FLOPs + 显存 + 预热
    # ------------------------------------------------------
    flops_counter = FLOPsCounter()
    flops_counter.register_hooks(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device)

    with torch.no_grad():
        model_output = model(dummy_x0, dummy_y0, dummy_phase)  # 这里只跑一次！

    # 取 FLOPs
    total_flops = flops_counter.get_total_flops()
    flops_g = round(total_flops / 10**9, 1)
    flops_counter.remove_hooks()

    # 取显存
    mem_peak = torch.cuda.max_memory_allocated(device)
    mem_usage_gb = round(mem_peak / (1024**3), 1)

    # 3. 计算 FPS（单独测速度）
    with torch.no_grad():
        start_time = time.perf_counter()
        for _ in range(test_times):
            model(dummy_x0, dummy_y0, dummy_phase)
        total_time = time.perf_counter() - start_time

    fps = round((test_times * batch_size) / total_time, 1)

    return params_m, flops_g, fps, mem_usage_gb