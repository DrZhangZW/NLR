"""
支持服务器多卡DDP和本地单卡伪DDP模拟的训练框架
核心特性：
1. 仅需执行 python your_script.py 即可自动启动4卡DDP训练
2. 自动检测是否为主进程，避免重复启动
3. 完整的分布式训练流程和日志监控
4. 单卡模拟多rank调试模式（新增）
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ======================== 全局默认配置（无需命令行输入） ========================
Sever_Config = {
    "runtime_use_fake_ddp": True, # False,  # 真实DDP=False,伪DDP=True
    "gpu_ids": "1,2,4,6",  # 服务器可用GPU列表
    "dev_world_size": 1,  # 默认4卡训练 = len(gpu_ids)
    "master_port": 29501,  # 默认主端口
    "master_addr": "127.0.0.1",
    "omp_num_threads": 4,  # 多人共用服务器推荐4，独占可选8
    "mkl_num_threads": 4,  # 同步设置MKL线程数，提升CPU效率
}


# ======================== 工具函数：分布式启动增强 ========================
def print_launch_log(msg: str, level: str = "INFO"):
    """分布式启动专用日志（仅主进程打印）"""
    # ========== 修复核心：正确判定主进程 ==========
    # 场景1：非分布式（无LOCAL_RANK/RANK）→ 直接打印
    # 场景2：分布式 → 仅 RANK=0（全局主进程）打印
    is_main_process = False
    if "RANK" in os.environ:
        is_main_process = (int(os.environ["RANK"]) == 0)
    elif "LOCAL_RANK" not in os.environ:
        is_main_process = True

    if is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")


# ======================== 判断是否为torchrun子进程 ========================
def is_ddp_launch():
    """校验torchrun自动注入的环境变量"""
    return all([
        "RANK" in os.environ,
        "WORLD_SIZE" in os.environ,
        "LOCAL_RANK" in os.environ,
        "MASTER_ADDR" in os.environ,
        "MASTER_PORT" in os.environ
    ])


# ======================== 新增：GPU ID映射工具（伪DDP核心） ========================
def get_simulated_gpu_id(rank: int):
    """
    伪DDP模式下，将rank映射到服务器的GPU ID（模拟多GPU编号）
    例如：rank 0 → 3, rank 1 →4, rank2→5, rank3→6（对应gpu_ids="3,4,5,6"）
    """
    gpu_list = [int(g) for g in Sever_Config["gpu_ids"].split(",")]
    # 超出长度时默认用第一个GPU
    return gpu_list[rank] if rank < len(gpu_list) else gpu_list[0]


# ======================== 第一步：自动启动torchrun（核心修改） ========================
def launch_torchrun():
    """
    智能启动torch.distributed.run多进程
    - 本地伪DDP模式自动跳过
    - 分布式模式自动启动多进程
    - 兼容参数透传和异常处理
    - 强化进程判断，避免子进程重复执行
    """
    # 强化跳过条件：子进程存在RANK/LOCAL_RANK任一环境变量即跳过（覆盖所有PyTorch版本）
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        return
    # 跳过条件2：启用本地伪DDP模式
    if Sever_Config["runtime_use_fake_ddp"]:
        print_launch_log("ℹ️  启用本地伪DDP模式，跳过分布式启动", "INFO")
        return
    # 1. 设置CPU线程数（从配置读取）
    os.environ["OMP_NUM_THREADS"] = str(Sever_Config["omp_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(Sever_Config["mkl_num_threads"])
    os.environ["OMP_DYNAMIC"] = "FALSE"
    # 2. 设置GPU可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = Sever_Config["gpu_ids"]
    # 关键：开启弹性训练traceback溯源，定位子进程具体报错
    os.environ["TORCHELASTIC_ERROR_FILE"] = "/tmp/torch_elastic_error.log"
    # 3. NCCL通信优化配置（已修复弃用警告）
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["TORCH_NCCL_TIMEOUT"] = "3600000"
    # 4. 构建启动命令
    cmd = [
              sys.executable,
              "-m", "torch.distributed.run",
              "--nproc_per_node", str(Sever_Config["dev_world_size"]),
              "--nnodes", "1",
              "--node_rank", "0",
              "--master_port", str(Sever_Config["master_port"]),
              "--master_addr", Sever_Config["master_addr"],
              sys.argv[0]
          ] + sys.argv[1:]
    # 5. 打印启动信息
    print_launch_log("=" * 80, "INFO")
    print_launch_log("🚀 启动分布式训练", "INFO")
    print_launch_log(f"📋 配置：GPU={Sever_Config['gpu_ids']} | 进程数={Sever_Config['dev_world_size']}", "INFO")
    print_launch_log(f"💻 命令：{' '.join(cmd)}", "INFO")
    print_launch_log("=" * 80, "INFO")
    # 6. 启动子进程（保留check=True，确保主进程感知子进程失败）
    subprocess.run(cmd, check=True)
    sys.exit(0)  # 主进程退出，避免重复执行
# 自动启动torchrun（仅在主进程执行）
launch_torchrun()
# ======================== 必须在torchrun启动后导入torch ========================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import copy
from utils.common_utils import print_log, device_setup, read_simple_config, increment_path, log_gpu_memory, FakeDDP, \
    separate_parameters2, create_multi_lr_optimizer,compute_model_metrics_native
from utils import logger
from utils.checkpoint_utils import load_checkpoint
from dataloader.data_loader import get_train_dataset, get_val_dataset, get_pred_dataset
from model.NLR_model_Plus import NLRNET_Plus
from model.cosine_warmup import CosineScheduler
from train_code import Trainer


# ======================== 主训练函数 ========================
def main(opts):
    """
    主训练流程
    :param opts: 配置参数对象
    """
    USE_FAKE_DDP = opts.runtime_use_fake_ddp
    rank = opts.dev_rank
    world_size = opts.dev_world_size
    local_gpu_id = opts.dev_local_gpu_id
    physical_gpu_ids = opts.dev_physical_gpu_id
    bs = opts.runtime_batch_size
    is_main_process = (rank == 0)  # 修复：定义主进程标识
    # ========== 核心：伪DDP下模拟GPU ID映射 ==========
    if USE_FAKE_DDP:
        # 模拟：rank映射到服务器GPU ID（如rank0→3，rank1→4）
        local_gpu_id = 0  # 本地物理GPU只有1张（固定为0）
        physical_gpu_id = get_simulated_gpu_id(rank)  # 日志层面模拟服务器GPU ID
        print_log(f"🎭 伪DDP模拟 | Rank={rank} | 本地物理GPU={local_gpu_id} | 模拟服务器GPU={physical_gpu_id}", rank)
    else:
        # 真实DDP：使用服务器实际GPU ID
        physical_gpu_id = physical_gpu_ids  # 日志层面模拟服务器GPU ID
        print_log(
            f"🚀 真实DDP运行 | Rank={rank}/{world_size} | 本地逻辑GPU={local_gpu_id} | 服务器物理GPU={physical_gpu_id}",
            rank
        )
    # ======================== 数据加载器（核心修改：调用自定义数据集函数） ========================
    train_loader, train_sampler = get_train_dataset(args=opts,  # 配置参数对象（原opts，函数内用getattr兼容）
                                                    USE_FAKE_DDP=USE_FAKE_DDP,  # 本地/服务器模式开关
                                                    rank=rank,  # 当前进程rank
                                                    world_size=world_size  # 总进程数（GPU数）
                                                    )

    val_loader, val_sampler = get_val_dataset(args=opts,  # 配置参数对象（原opts，函数内用getattr兼容）
                                              USE_FAKE_DDP=USE_FAKE_DDP,  # 本地/服务器模式开关
                                              rank=rank,  # 当前进程rank
                                              world_size=world_size,  # 总进程数（GPU数）
                                              val_batch_size=1,#getattr(opts, "dataset_val_batch_size", 2),
                                              )
    pred_loader, pred_sampler = get_pred_dataset(args=opts,  # 配置参数对象（原opts，函数内用getattr兼容）
                                                 USE_FAKE_DDP=USE_FAKE_DDP,  # 本地/服务器模式开关
                                                 rank=rank,  # 当前进程rank
                                                 world_size=world_size,  # 总进程数（GPU数）
                                                 pred_batch_size=getattr(opts, "dataset_pred_batch_size", 2),
                                                 )
    # ======================== 模型初始化 ========================
    # 确保模型在正确的GPU上
    model = NLRNET_Plus().to(local_gpu_id)
    if USE_FAKE_DDP:
        # 本地伪DDP：使用FakeDDP包装（模拟DDP接口）
        ddp_model = FakeDDP(model, device_ids=[local_gpu_id], output_device=local_gpu_id, find_unused_parameters=False)
    else:
        # 服务器模式：真实DDP
        ddp_model = DDP(
            model,
            device_ids=[local_gpu_id],  # 当前进程绑定的逻辑GPU ID（0-3）
            output_device=local_gpu_id,  # 可选：与device_ids[0]一致，高版本可省略
            find_unused_parameters=True,  # 核心：允许未使用参数，解决梯度归约报错
            gradient_as_bucket_view=True,  # 可选：减少显存占用（高版本推荐）
            static_graph=False  # 可选：动态图场景（默认False，无需修改）
        )
    print_log(
        f"🧠 模型信息 - 参数量: {sum(p.numel() for p in model.parameters()):,} | DDP初始化成功: {True}",
        rank=rank
    )
    # ====================== 3. 分布式适配的参数分组与优化器构建 ======================
    # 关键：参数分组必须基于原始model（而非ddp_model），因为DDP的parameters()是包装后的，分组会失效
    param_dicts, param_dicts_names = separate_parameters2(
        model,  # 注意：这里用原始model，而非ddp_model
        keywords=['CTDM', 'Proj_J', 'relight']
    )
    learning_rates = {'base': 1e-4,
                      'CTDM': 1e-4,
                      'Proj_J': 1e-4,
                      'relight': 1e-4
                      }
    # 构建多学习率优化器（核心：优化器参数仍指向原始model，但DDP会自动同步梯度）
    optimizer = create_multi_lr_optimizer(
        param_dicts,
        learning_rates,
        beta1=getattr(opts, "optim_adamw_beta1", 0.9),
        beta2=getattr(opts, "optim_adamw_beta2", 0.98),
        weight_decay=4e-4,  # getattr(opts, "optim_weight_decay", 4e-5)
    )
    # ======================== 混合精度训练（分布式兼容） ========================
    gradient_scaler = GradScaler(
        enabled=True,
        init_scale=2. ** 8,  # 从2^7→2^8，小幅提升初始缩放，适配aux权重提升后的梯度
        growth_factor=1.1,  # 1.2→1.1，更保守的增长，避免梯度爆炸
        backoff_factor=0.5,
        growth_interval=250  # 200→250，延长增长检查间隔，减少缩放波动
    )
    # ======================== 余弦学习率调度器（分布式适配） ========================
    # 调度器参数统一配置（所有rank保持一致）
    max_iterations = getattr(opts, "scheduler_max_iterations", len(train_loader))
    warmup_iterations = getattr(opts, "scheduler_warmup_iterations",
                                int(0.2 * max_iterations))  # 默认预热20%
    lr_multipliers = {
        'base': 1.0,
        'CTDM': 1.0,
        'Proj_J': 1.0,
        'relight': 1.0
    }
    lr_scheduler = CosineScheduler(
        warmup_iterations=0,
        max_iterations=40000,
        min_lr=5e-6,  # 从8e-7→1e-6，提升后期最小LR，避免CTDM/Proj-J优化停滞
        max_lr=4e-5,  # 从4.8e-5→5.5e-5，提升峰值LR，给CTDM更多优化余量
        warmup_init_lr=1e-6,
        period_epochs=1200,
        lr_multipliers=lr_multipliers,
        is_iter_based=False
    )
    # ======================== 原有调用逻辑修改 ========================
    start_iteration, best_metric = 0, 2
    start_epoch = 0
    pretrained = getattr(opts, "model_pretrained", None)
    if pretrained != "None":
        # 关键修改：传入分布式相关参数
        model, optimizer, gradient_scaler, start_epoch, start_iteration, best_metric = load_checkpoint(
            opts=opts,
            model=model,  # 原始model（非ddp_model）
            optimizer=optimizer,
            gradient_scalar=gradient_scaler,
            USE_FAKE_DDP=USE_FAKE_DDP,  # 新增：传入是否伪DDP
            rank=rank,  # 新增：当前进程rank
            local_gpu_id=local_gpu_id  # 新增：本地GPU ID
        )
    # ========== 初始化 Trainer（新增分布式参数） ==========

    training_engine = Trainer(
        opts=opts,
        model=ddp_model,  # 传入DDP/FakeDDP包装后的模型（关键！）
        validation_loader=val_loader,
        training_loader=train_loader,
        optimizer=optimizer,
        gradient_scaler=gradient_scaler,
        scheduler=lr_scheduler,
        start_epoch=start_epoch,
        start_iteration=start_iteration,
        best_metric=best_metric,
        # ========== 新增分布式参数 ==========
        rank=rank,
        local_gpu_id=local_gpu_id,
        USE_FAKE_DDP=USE_FAKE_DDP
    )
    #training_engine.run(train_sampler, phase=2)
    #training_engine.val(val_loader, phase=4)
    training_engine.predict(pred_loader, phase=4)
    t=1



if __name__ == '__main__':

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    parser = argparse.ArgumentParser(description="Training Arguments", add_help=True)
    parser.add_argument('--common_config_file', type=str,
                        default=ROOT / "mobilevitv3_small_multiserver.yaml",
                        help="Configuration file")
    opts = parser.parse_args()
    # 服务器上每个进程都会执行这一步，本地伪DDP只需执行1次（配置内容一致）
    opts = read_simple_config(opts)
    # ========== 伪DDP：模拟多rank循环执行（核心调试） ==========
    if Sever_Config["runtime_use_fake_ddp"]:
        print_launch_log(
            f"🔧 启动伪DDP模拟 | 模拟GPU ID列表={Sever_Config['gpu_ids']} | 进程数={Sever_Config['dev_world_size']}",
            "DEBUG")
        # 循环模拟每个rank（0~3），映射到服务器GPU ID（3~6）
        for sim_rank in range(Sever_Config["dev_world_size"]):
            # 1. 复制opts（避免不同rank的变量互相污染）
            sim_opts = copy.deepcopy(opts)
            # 2. 模拟服务器每个进程的device_setup：根据rank初始化GPU变量,执行device_setup（传入模拟rank，触发伪DDP逻辑）
            sim_opts = device_setup(sim_opts, sim_rank=sim_rank)
            # 提取核心变量（严格匹配core_vars命名）
            rank = sim_opts.dev_rank
            world_size = sim_opts.dev_world_size
            local_gpu_id = sim_opts.dev_local_gpu_id
            physical_gpu_id = sim_opts.dev_physical_gpu_id
            logical_gpu_id_final = sim_opts.dev_logical_gpu_id_final  # 最终绑定的逻辑ID（替代原actual_gpu_id）
            is_master = sim_opts.dev_is_master_node

            is_main_process = (rank == 0)  # 统一使用core_vars中的变量名
            gpu_name = torch.cuda.get_device_name(logical_gpu_id_final) if torch.cuda.is_available() else "CUDA不可用"
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

            logger.info("=" * 80)
            logger.info(f"📌 伪DDP/单卡模式初始化完成 [Rank {rank}/{world_size}]*******")
            logger.info("┌────────────────────────────────────────────────────────────────┐")
            logger.info(f"│ 全局Rank        : {rank:>2} (主进程: {'✅' if is_main_process else '❌'})       │")
            logger.info(f"│ 总进程数(world_size): {world_size:>2} (伪DDP/单卡)                  │")
            logger.info(f"│ 本地Rank(LOCAL_RANK): {local_gpu_id:>2}                           │")
            logger.info(f"│ 映射物理GPU ID  : {physical_gpu_id:>2} (目标卡)                  │")
            logger.info(f"│ 实际绑定GPU ID  : {logical_gpu_id_final:>2} ({gpu_name})          │")
            logger.info("├────────────────────────────────────────────────────────────────┤")
            logger.info(f"│ 进程PID         : {os.getpid():>6}                                │")
            logger.info(f"│ 可见GPU列表     : {cuda_visible:>10}                              │")
            logger.info(f"│ 系统检测GPU数   : {n_gpus:>6}                                    │")
            logger.info("└────────────────────────────────────────────────────────────────┘")
            logger.info("=" * 80)
            log_gpu_memory(rank, logical_gpu_id_final)
            # 4. 执行主训练流程（当前模拟rank）
            save_dir = str(increment_path(Path(sim_opts.common_project) / sim_opts.common_name))
            setattr(sim_opts, "common_save_dir", save_dir)
            main(sim_opts)  # 注意：传入sim_opts而非原始opts，保证变量隔离
    else:
        # 每个进程独立执行device_setup（读取系统环境变量）
        opts = device_setup(opts)
        # 提取核心变量（严格匹配core_vars命名）
        rank = opts.dev_rank
        world_size = opts.dev_world_size
        local_gpu_id = opts.dev_local_gpu_id
        physical_gpu_id = opts.dev_physical_gpu_id
        logical_gpu_id_final = opts.dev_logical_gpu_id_final
        is_main_process = (rank == 0)  # 统一使用core_vars中的变量名

        if torch.distributed.is_initialized():
            # ========== 关键优化2：按rank顺序打印，彻底避免竞争 ==========
            for wait_rank in range(world_size):
                if rank == wait_rank:
                    # 补充GPU信息
                    gpu_name = torch.cuda.get_device_name(
                        logical_gpu_id_final) if torch.cuda.is_available() else "CUDA不可用"
                    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
                    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

                    # ========== 关键优化3：每行添加[Rank X]前缀，归属清晰 ==========
                    logger.info(f"[Rank {rank}] " + "=" * 80)
                    logger.info(f"[Rank {rank}] 📌 真实DDP进程初始化完成 [Rank {rank}/{world_size}]*******")
                    logger.info(f"[Rank {rank}] ┌────────────────────────────────────────────────────────────────┐")
                    logger.info(
                        f"[Rank {rank}] │ 全局Rank        : {rank:>2} (主进程: {'✅' if is_main_process else '❌'})       │")
                    logger.info(f"[Rank {rank}] │ 总进程数(world_size): {world_size:>2} (真实四卡)                  │")
                    logger.info(f"[Rank {rank}] │ 本地Rank(LOCAL_RANK): {local_gpu_id:>2}                           │")
                    logger.info(f"[Rank {rank}] │ 映射物理GPU ID  : {physical_gpu_id:>2} (目标卡)                  │")
                    logger.info(f"[Rank {rank}] │ 实际绑定GPU ID  : {logical_gpu_id_final:>2} ({gpu_name})          │")
                    logger.info(f"[Rank {rank}] ├────────────────────────────────────────────────────────────────┤")
                    logger.info(f"[Rank {rank}] │ 进程PID         : {os.getpid():>6}                                │")
                    logger.info(f"[Rank {rank}] │ 可见GPU列表     : {cuda_visible:>10}                              │")
                    logger.info(f"[Rank {rank}] │ 系统检测GPU数   : {n_gpus:>6}                                    │")
                    logger.info(f"[Rank {rank}] └────────────────────────────────────────────────────────────────┘")
                    logger.info(f"[Rank {rank}] " + "=" * 80)

                    # 显存日志也加rank标识
                    log_gpu_memory(rank, logical_gpu_id_final)

                # 等待当前rank打印完成，再执行下一个rank（核心：避免同时输出）
                torch.distributed.barrier()
        # ========== 外层简化版：主进程生成目录 + 屏障 + 统一前缀 ==========
        import tempfile

        if torch.distributed.is_initialized():
            base_dir = Path(opts.common_project) / opts.common_name
            # 定义所有进程可访问的临时文件路径（系统临时目录，所有进程可读可写）
            temp_file = os.path.join(tempfile.gettempdir(), "ddp_save_dir.txt")

            if is_main_process:
                # 主进程生成目录 + 创建目录 + 写入临时文件
                save_dir = str(increment_path(base_dir))
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                with open(temp_file, "w") as f:
                    f.write(save_dir)  # 将目录路径写入文件
                logger.info(f"[Rank {rank}] ✅ 主进程生成目录：{save_dir}（已写入临时文件）")

            # 所有进程等待主进程完成目录生成+文件写入
            torch.distributed.barrier()

            # 所有进程（包括主进程）读取临时文件中的统一目录
            with open(temp_file, "r") as f:
                save_dir = f.read().strip()  # 读取主进程写入的路径

            # ======================== 核心修复：新增同步屏障 ========================
            torch.distributed.barrier()  # 等待所有进程完成文件读取，再执行后续删除操作
            # ========================================================================

            # 主进程清理临时文件（避免残留，此时所有进程已读完，无冲突）
            if is_main_process:
                os.remove(temp_file)
                logger.info(f"[Rank {rank}] ✅ 已删除临时文件：{temp_file}")
        else:
            # 单卡模式保持原有逻辑
            save_dir = str(increment_path(Path(opts.common_project) / opts.common_name))

        # 写入opts，确保所有Rank的common_save_dir完全一致
        setattr(opts, "common_save_dir", save_dir)
        logger.info(f"[Rank {rank}] 最终统一目录: {save_dir}")
        main(opts)

        if dist.is_initialized():
            dist.destroy_process_group()
            print_log(f"✅ Rank {rank} 分布式进程组已销毁", rank)