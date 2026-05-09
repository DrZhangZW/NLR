import torchvision.utils as vutils
import os
import pathlib
import torch.distributed as dist
import time
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from PIL import Image
from typing import List, Dict, Optional
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils import logger
from utils.metrics_utils import Statistics, metric_monitor
from utils.checkpoint_utils import save_checkpoint
from utils.common_utils import visualize_tensor, batch_psnr, crop_valid_regions,process_kitti_tensors
from dataloader.under_generate import generate_train_data
from model.loss_functions import total_color_loss,physical_loss,gradient_loss,grad_denoise_loss
class Trainer:
    def __init__(
            self,
            opts,
            model,
            validation_loader,
            training_loader,
            optimizer,
            gradient_scaler,  # 修复：拼写错误 scalar → scaler
            scheduler,
            start_epoch,
            start_iteration,
            best_metric,
            # ========== 新增分布式参数 ==========
            rank: int = 0,  # 当前进程rank（主进程为0）
            local_gpu_id: int = 0,  # 本地GPU ID
            USE_FAKE_DDP: bool = False,  # 是否使用伪DDP
            world_size: int = 1  # 补充：总进程数，分布式核心参数
    ):
        self.opts = opts
        self.model = model
        self.gradient_scaler = gradient_scaler  # 同步修复参数名

        self.training_loader = training_loader
        self.validation_loader = validation_loader  # 修复：原代码少空格，格式不规范
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.train_iterations = start_iteration
        self.max_iterations = 10000
        self.best_metric = best_metric
        self.device = getattr(opts, "dev_device")
        # ========== 保存分布式变量 ==========
        self.rank = rank
        self.local_gpu_id = local_gpu_id
        self.USE_FAKE_DDP = USE_FAKE_DDP
        self.world_size = world_size  # 补充：总进程数

        self.lr_history = []  # 存储各epoch的LR列表
        self.curve_names = []  # 存储参数组的曲线名称（仅初始化一次）]
        self.loss_history = {'train': [], 'val': []}
        # 仅主进程打印初始化信息（增强日志）
        if self.rank == 0:
            logger.log(
                f"📋 Trainer初始化完成 | rank: {self.rank}/{self.world_size} "
                f"| GPU: {self.local_gpu_id} | 伪DDP模式: {self.USE_FAKE_DDP} "
                f"| 设备: {self.device}"
            )
        # ====================saved folder===================================
        # 训练配置
        self._setup_training_config()
        # 日志和保存配置
        self._setup_logging_and_saving()
    def run(self, train_sampler, phase=1):
        """运行训练过程（适配分布式/伪DDP，含真实DDP进程同步）

        Args:
            train_sampler: 训练数据采样器（DistributedSampler）
            phase: 训练阶段 (1, 2, 3)
        """
        self.max_epochs = getattr(self.opts, "scheduler_max_epochs", 500)

        # 1. 仅主进程打印训练启动日志（避免多Rank刷屏）
        if self.rank == 0:
            logger.info(
                f"📌 Starting training phase {phase} | Rank={self.rank}/{self.world_size} | Epochs={self.max_epochs}")

        for epoch in range(self.start_epoch, self.max_epochs):
            self._update_seed_per_epoch(epoch)

            # 更新数据采样器（如果提供）
            if train_sampler is not None:
                # 直接调用set_epoch：你的采样器已在set_epoch中整合update_scales + 分布式sampler同步
                train_sampler.set_epoch(epoch)  # 保证多Rank数据分片对齐

            # 3. 所有Rank执行训练（核心流程不区分主从）
            train_metrics = self.train_epoch(epoch, phase)

            # 4. 真实DDP：所有Rank完成当前epoch训练后，再执行主进程操作（进程屏障）
            if not self.USE_FAKE_DDP and dist.is_available() and dist.is_initialized():
                dist.barrier()  # 阻塞直到所有Rank到达此位置
            # 5. 仅主进程保存检查点+验证（避免多Rank重复保存/验证）
            if self.rank == 0:
                self._save_checkpoint(epoch, train_metrics)
                # 真实DDP：主进程完成保存/验证后，释放其他Rank（可选，视需求添加）
            if not self.USE_FAKE_DDP and dist.is_available() and dist.is_initialized():
                dist.barrier()
            # 6. 伪DDP模式下：手动同步epoch状态（模拟真实DDP的barrier，便于调试）
            if self.USE_FAKE_DDP:
                logger.info(f"🎭 伪DDP模拟 | Rank={self.rank} | Epoch={epoch} 训练完成，等待同步")


    def train_epoch(self,epoch,phase=1):
        self.model.train()
        epoch_start_time = time.time()
        batch_load_start = time.time()
        # 配置梯度累积
        accum_after_epoch = getattr(self.opts, "common_accum_after_epoch", 20)
        accum_freq = 2 if (epoch > accum_after_epoch) or (epoch > self.max_epochs - 10) else 1

        # 初始化统计和进度条
        # progress_bar = tqdm(self.training_loader, desc=f"Epoch {epoch} - Training Phase {phase}", leave=True)
        self.train_stats.reset()

        # 更新学习率
        self.optimizer = self.update_optimizer_lr(self.optimizer, epoch, self.train_iterations, phase=phase)
        self.optimizer.zero_grad()
        # 记录学习率
        lrs_and_names = self.record_learning_rates(self.optimizer, epoch)
        # ------------------------------------
        img_num = 0
        loss_dict = {}
        if epoch >620:
            if dist.is_available() and dist.is_initialized():
                _ = dist.destroy_process_group()  # 用下划线吞返回值/异常
            sys.exit()

        for batch_id, batch in enumerate(self.training_loader):

            batch_load_toc = time.time() - batch_load_start
            # 准备数据
            (clean,depth,under,underE) = batch
            clean = clean.to(self.device, non_blocking=True)
            depth = depth.to(self.device, non_blocking=True)
            under = under.to(self.device, non_blocking=True)
            underE = underE.to(self.device, non_blocking=True)

            batch_size = clean.tensors.shape[0]
            img_num = img_num + batch_size
            # 优化1：数据增强禁用梯度计算
            with torch.no_grad():
                train_datas, refer_datas, high_quality, phase_use = generate_train_data(
                    clean, depth, under, underE, phase, device=self.device,
                    max_resize_size=480, target_ratio=4 / 3
                )
            del clean, depth, under, underE  # 及时删除原始数据
            torch.cuda.synchronize()
            #visualize_tensor(train_datas.tensors)
            #visualize_tensor(refer_datas.tensors)
            #visualize_tensor(high_quality.tensors)
            # --------------------------------update parameters-------------------------
            use_amp=True
            # 1. 启用AMP自动精度转换（包裹前向传播）
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 模型前向传播
                pred_img, loss_aux, cmd_loss = self.model(train_datas, high_quality, phase_use)
                # 冗余损失置0（可删除，保留仅为兼容原有代码结构）
                # 2. 计算基础损失（保留计算，分场景取舍，兼容原有代码）
                l1_loss = nn.functional.l1_loss(pred_img, refer_datas.tensors)
                mse_loss = nn.functional.mse_loss(pred_img, refer_datas.tensors)  # 新增：对齐PSNR的MSE损失
                phy_loss = physical_loss(pred_img, train_datas)  # 原有：物理伪监督损失
                grad_loss = gradient_loss(pred_img, refer_datas.tensors)  # 新增：高频梯度损失
                color_loss = total_color_loss(pred_img, refer_datas.tensors)  # 你的颜色损失（仅仿真用）

                # 核心调整：488epoch直接进入“冲刺阶段”，最大化aux-J权重，CTDM（cmd_loss）同步加权
                if epoch <= 240:
                    aux_weight = 0.1
                    cmd_weight = 0.1
                elif epoch <= 280:
                    aux_weight = 0.1
                    cmd_weight = 0.1
                elif epoch <= 350:
                    aux_weight = 0.5
                    cmd_weight = 0.5
                elif epoch <= 450:
                    aux_weight = 1.2
                    cmd_weight = 0.8
                else:
                    # 450+epoch（含488）：冲刺收敛，aux权重拉满，cmd权重同步提升
                    aux_weight = 2  # 从1.0→1.8，强制优化Proj-J的loss_aux
                    cmd_weight = 1.5  # 从0.9→1.2，加快CTDM的cmd_loss收敛
                # 4. 核心判断：是否为真实数据（is_paired=False即为真实数据，Phase3已强制设置）

                is_real_data = train_datas.is_real_data
                # 总损失：L1权重小幅降，给aux/cmd更多占比
                # 5. 分场景构建总损失（核心！仿真双损失，真实仅物理+aux/cmd）
                # 4. 600轮后核心：差异化损失配置（仿真强监督冲PSNR，真实纯伪监督避偏移）
                if not is_real_data:
                    # 场景1：仿真数据 - 加入L1+MSE+color_loss+梯度Loss，强监督冲击高PSNR
                    loss_phase1 = (
                                          3 * l1_loss +  # L1：基础像素拟合，保证鲁棒性
                                          5 * mse_loss +  # MSE：对齐PSNR指标，直接提升数值
                                          0.01 * color_loss +  # 颜色损失：保证色彩一致性（仿真数据无域偏移，可放心用）
                                          1.2 * grad_loss +  # 梯度损失：压榨高频细节，提升边缘PSNR（低权重不主导）
                                          1 * phy_loss +  # 物理损失：辅助贴合水下物理规律
                                          aux_weight * loss_aux +  # Proj-J：满权重，光照/结构优化
                                          cmd_weight * cmd_loss  # CTDM：满权重，特征匹配优化
                                  ) / accum_freq  # 保留梯度累积，适配大批次训练
                else:
                    # 场景2：真实数据 - 完全移除color_loss/L1/MSE/梯度Loss，仅物理伪监督+降权aux/cmd
                    # 核心原因：真实数据color_loss偏高（11+），强行拟合会导致向仿真分布偏移，破坏真实感
                    t=1
                    """
                    grad_denoise = grad_denoise_loss(pred_img)
                    loss_phase1 = (
                                          1.0 * phy_loss +  # 核心：物理伪监督（无偏移，保证增强合理性）
                                          0.8 * aux_weight * loss_aux +  # Proj-J：降权，稳定训练
                                          0.8 * cmd_weight * cmd_loss +  # CTDM：降权，保留结构优化
                                          0.15 * grad_denoise  # 辅助：梯度降噪（极低权重，仅抑制噪声，不干扰主监督）
                                  ) / accum_freq  # 梯度累积保留，适配大批次
                    """
            # 3. 反向传播（AMP模式下缩放Loss，同时适配梯度累积）
            if use_amp:
                # ✅ 正确：scaler直接scale归一化后的loss，不额外除accum_freq
                scaled_loss = self.gradient_scaler.scale(loss_phase1)
                scaled_loss.backward()
            else:
                loss_phase1.backward()

            # 梯度累积更新
            if (batch_id + 1) % accum_freq == 0:
                if use_amp:
                    # ✅ 正确：unscale恢复梯度原值（grad/accum_freq）
                    self.gradient_scaler.unscale_(self.optimizer)
                    # ✅ 正确：max_norm不除以accum_freq，限制单次梯度上限
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=0.8,  # 无需 / accum_freq
                        norm_type=2
                    )
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # ================================moniter================================
            if not is_real_data or train_datas.has_gt:
                psnr = batch_psnr(refer_datas.tensors, pred_img)
            else:
                psnr = 0.0  # 无参考真实数据，PSNR置0

            loss_dict = {'total_loss': loss_phase1,'loss_color':color_loss ,'loss_aux_J': loss_aux,'loss_cmd':cmd_loss, 'psnr': psnr}
            # 更新统计
            metrics_obj = metric_monitor(loss=loss_dict,
                                         metric_names=self.metric_names)
            # 生成迭代摘要
            self.train_stats.update(metric_vals=metrics_obj, batch_time=batch_load_toc, n=batch_size)
            summary_str_obj = self.train_stats.iter_summary(epoch=epoch,
                                                            n_processed_samples=self.train_iterations,
                                                            total_samples=self.max_iterations,
                                                            learning_rate=lrs_and_names['learning_rates'][-1],
                                                            elapsed_time=epoch_start_time)
            # 4. 仅主进程打印日志+写入文件（避免多Rank刷屏/重复写入）
            if self.rank == 0:
                # 提取shape（CPU字符串），避免GPU张量参与字符串拼接
                tensor_shape = str(train_datas.tensors.shape)
                logger.log(f'{tensor_shape}: {summary_str_obj}')
                with open(self.loss_iter, 'a') as f:
                    f.write(f'\n{tensor_shape}: {summary_str_obj}')

            #processBar.set_description_str(f'{clean.tensors.shape}-{summary_str_obj}')

            #if self.rank == 0 and (self.train_iterations % 500 == 0):
            if  (batch_id==0) or (self.train_iterations % 500 == 0):
                #['Input', 'Prediction', 'Ground Truth']
                plot_img = [train_datas.tensors, pred_img, refer_datas.tensors,high_quality.tensors]
                visualize_tensors(tensor_list=plot_img, save_dir=self.train_result,
                                  epoch=epoch,
                                  iteration=self.train_iterations, titles = 'train',rank=self.rank )
            # 优化5：强制释放无用张量+清空缓存
            del pred_img, train_datas, refer_datas, high_quality, loss_phase1, l1_loss, color_loss, loss_aux, cmd_loss
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.train_iterations = self.train_iterations + 1
            img_num = img_num + batch_size
        # ======================epoch总结（仅主进程执行）=============================
        if self.rank == 0:
            metric_stats_avg_epoch = self.train_stats.epoch_summary(epoch=epoch, stage="training")
            print(f'Epoch: {epoch} Total image: {img_num}: {metric_stats_avg_epoch} mean psnr; mean total loss')
            with open(self.loss_epoch, 'a', encoding='utf-8') as f:
                f.write(f'\nEpoch: {epoch} Total image: {img_num}: {metric_stats_avg_epoch} mean psnr; mean total loss')

        return loss_dict


    def val(self,dataloader,phase=1):
        """
        phase=1: simulate image, phase=2: real underwater images
        """
        self.model.eval()

        accum_freq = 2
        # 初始化统计信息
        self.val_stats.reset()
        val_start_time = time.time()
        val_metrics = {}
        img_num = 0
        # 仅主进程打印验证启动日志
        if self.rank == 0:
            logger.info(f"📊 Starting validation phase {phase} | Rank={self.rank}/{self.world_size}")

        # 禁用梯度计算（验证阶段核心优化）
        with torch.no_grad():
            for batch_id, batch in enumerate(dataloader):
                batch_load_start = time.time()
                # 1. 数据加载与设备迁移（和训练保持一致）
                (clean, depth, under, underE) = batch
                clean = clean.to(self.device, non_blocking=True)
                depth = depth.to(self.device, non_blocking=True)
                under = under.to(self.device, non_blocking=True)
                underE = underE.to(self.device, non_blocking=True)

                # 2. 数据预处理（和训练保持一致）
                train_datas, refer_datas, high_quality, phase_use = generate_train_data(
                    clean, depth, under, underE, phase, device=self.device,
                    max_resize_size=480, target_ratio=4 / 3
                )
                batch_size = train_datas.tensors.shape[0]
                img_num += batch_size
                # 3. 模型前向推理（启用AMP加速）
                use_amp = True
                # 1. 启用AMP自动精度转换（包裹前向传播）
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_img, loss_aux, cmd_loss = self.model(train_datas, high_quality, phase_use)

                    # 4. 计算验证指标
                    l1_loss = 5*nn.functional.l1_loss(pred_img, refer_datas.tensors)/accum_freq
                    color_loss = total_color_loss(pred_img, refer_datas.tensors)
                    psnr = batch_psnr(refer_datas.tensors, pred_img)
                # 5. 记录批次指标
                batch_load_toc = time.time() - batch_load_start
                loss_dict = {
                    'total_loss': l1_loss,
                    'loss_color': color_loss,
                    'loss_aux_J': loss_aux,
                    'loss_cmd': cmd_loss,
                    'psnr': psnr
                }
                # 6. 更新验证统计
                metrics_obj = metric_monitor(loss=loss_dict, metric_names=self.metric_names)
                self.val_stats.update(metric_vals=metrics_obj, batch_time=batch_load_toc, n=batch_size)
                # 7. 仅主进程打印批次日志
                if self.rank == 0:
                    summary_str_obj = self.val_stats.iter_summary(
                        epoch=-1,  # 验证无epoch概念，用-1标识
                        n_processed_samples=batch_id,
                        total_samples=len(dataloader),
                        learning_rate=0.0,  # 验证无学习率
                        elapsed_time=val_start_time
                    )
                    tensor_shape = str(train_datas.tensors.shape)
                    logger.log(f'[VAL] {tensor_shape}: {summary_str_obj}')
                # 8. 定期保存验证可视化结果（仅主进程）
                if self.rank == 0: #and (batch_id % 100 == 0):
                    #plot_img = [train_datas.tensors, pred_img, refer_datas.tensors, high_quality.tensors]
                    plot_img = [pred_img]
                    visualize_tensors(
                        tensor_list=plot_img,
                        save_dir=os.path.join(self.train_result, f'phase_{phase}'),
                        epoch=-1,
                        iteration=batch_id,
                        titles='val',
                        rank=self.rank
                    )

                # 9. 内存优化：及时清理无用张量
                del pred_img, train_datas, refer_datas, high_quality, l1_loss, color_loss, loss_aux, cmd_loss
                del clean, depth, under, underE
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        # 10. 分布式同步：等待所有Rank完成验证
        if not self.USE_FAKE_DDP and dist.is_available() and dist.is_initialized():
            dist.barrier()

        # 11. 仅主进程计算并保存验证集整体指标
        if self.rank == 0:
            # 计算epoch级别的平均指标
            metric_stats_avg = self.val_stats.epoch_summary(epoch=-1, stage="validation")
            val_metrics = metric_stats_avg

            # 打印并保存验证总结
            print(f'[VAL SUMMARY] Phase {phase} | Total images: {img_num} | {metric_stats_avg}')
            with open(self.loss_val, 'a', encoding='utf-8') as f:
                f.write(
                    f'\n[VAL] Phase {phase} | Epoch: {getattr(self, "current_epoch", -1)} | Total images: {img_num}: {metric_stats_avg}')

            # 保存验证指标到文件
            val_time = time.time() - val_start_time
            logger.info(
                f"✅ Validation phase {phase} completed | Time: {val_time:.2f}s | Avg PSNR: {val_metrics.get('psnr', 0):.4f}")

        # 切换回训练模式
        self.model.train()

        return val_metrics

    def predict(self, dataloader, phase=1, save_results=True):
        """
        模型预测/推理过程（适配分布式/伪DDP）
        Args:
            dataloader: 预测数据加载器
            phase: 预测阶段 (1, 2, 3)
            save_results: 是否保存预测结果
        Returns:
            predict_results: 预测结果列表（包含输入、预测、GT等信息）
        """
        # 切换模型到评估模式
        #self.model.eval()
        #self.model.train()

        predict_results = []
        predict_start_time = time.time()
        img_num = 0

        # 仅主进程打印预测启动日志
        if self.rank == 0:
            logger.info(
                f"🚀 Starting prediction phase {phase} | Rank={self.rank}/{self.world_size} | Save results: {save_results}")
            # 创建预测结果保存目录
            self.predict_save_dir = os.path.join(self.train_result, f'predict_phase_{phase}')
            os.makedirs(self.predict_save_dir, exist_ok=True)

        # 禁用梯度计算
        with torch.no_grad():
            for batch_id, batch in enumerate(dataloader):
                # 1. 数据加载与设备迁移
                # 注意：预测阶段batch可能包含文件名等额外信息
                if len(batch) == 4:
                    clean, depth, under, underE = batch
                    file_names = [f"batch_{batch_id}_img_{i}" for i in range(clean.tensors.shape[0])]
                else:
                    clean, depth, under, underE, file_names = batch

                clean = clean.to(self.device, non_blocking=True)
                depth = depth.to(self.device, non_blocking=True)
                under = under.to(self.device, non_blocking=True)
                underE = underE.to(self.device, non_blocking=True)

                batch_size = clean.tensors.shape[0]
                img_num += batch_size

                # 2. 数据预处理（和训练/验证保持一致）
                train_datas, refer_datas, high_quality, phase_use = generate_train_data(
                    clean, depth, under, underE, phase, device=self.device,
                    max_resize_size=512, target_ratio=4 / 3
                )
                use_amp = True
                # 3. 模型前向推理（启用AMP）
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_img, _, _ = self.model(train_datas, high_quality, phase_use)
                mask_pad = train_datas.mask

                pred_img_cropped = crop_valid_regions(pred_img, mask_pad)
                under_cropped = crop_valid_regions(train_datas.tensors, mask_pad)

                def tensor_to_numpy(img_t):
                    img = img_t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    return img

                img_in = tensor_to_numpy(under_cropped)
                img_out = tensor_to_numpy(pred_img_cropped)

                # --------------------------
                # 梯度阈值拉高！更干净！
                # --------------------------
                def get_edge(img):
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    edge = cv2.Canny(gray, 70, 180)  # 👈 高阈值！干净强边缘
                    return edge

                def get_density(edge):
                    return np.mean(edge > 0)

                edge_in = get_edge(img_in)
                edge_out = get_edge(img_out)

                dens_in = get_density(edge_in)
                dens_out = get_density(edge_out)

                print(f"Batch {batch_id:03d} | INPUT: {dens_in:.4f} | OURS: {dens_out:.4f}")

                # ============================
                # 👇 强制保存 380x380
                # ============================
                save_dir = "edge_vis_380"
                os.makedirs(save_dir, exist_ok=True)

                size = (380, 380)

                Image.fromarray(img_in).resize(size, Image.LANCZOS).save(f"{save_dir}/{batch_id:03d}_input.png")
                Image.fromarray(img_out).resize(size, Image.LANCZOS).save(f"{save_dir}/{batch_id:03d}_enhanced.png")
                Image.fromarray(edge_in).resize(size, Image.LANCZOS).save(f"{save_dir}/{batch_id:03d}_edge_input.png")
                Image.fromarray(edge_out).resize(size, Image.LANCZOS).save(
                    f"{save_dir}/{batch_id:03d}_edge_enhanced.png")

                # 3. 后续直接使用 cropped 的图即可
                pred_img = pred_img_cropped
                under_ori_crop = under_cropped
                # 4. 保存预测结果（仅主进程）
                if save_results and self.rank == 0:
                    # 遍历批次中的每张图片
                    for idx in range(batch_size):
                        # 构建保存路径
                        img_name = file_names[idx] if isinstance(file_names, list) else f"batch_{batch_id}_img_{idx}"
                        save_path = os.path.join(self.predict_save_dir, img_name)

                        # 保存预测结果
                        plot_img = [
                            under_ori_crop[idx:idx + 1],#train_datas.tensors[idx:idx + 1],
                            #pred_img[idx:idx + 1]
                            #refer_datas.tensors[idx:idx + 1]
                        ]
                        visualize_tensors(
                            tensor_list=plot_img,
                            save_dir=self.predict_save_dir,
                            epoch=-1,
                            iteration=f"{batch_id}_{idx}",
                            titles='ori_img',
                            rank=self.rank
                        )
                        # 保存预测结果
                        plot_img = [
                            #under_ori_crop[idx:idx + 1],#train_datas.tensors[idx:idx + 1],
                            pred_img[idx:idx + 1]
                            #refer_datas.tensors[idx:idx + 1]
                        ]
                        visualize_tensors(
                            tensor_list=plot_img,
                            save_dir=self.predict_save_dir,
                            epoch=-1,
                            iteration=f"{batch_id}_{idx}",
                            titles='predict',
                            rank=self.rank
                        )

                # 5. 记录预测结果
                batch_result = {
                    'batch_id': batch_id,
                    'file_names': file_names,
                    'pred_img': pred_img.cpu().numpy(),
                    'gt_img': refer_datas.tensors.cpu().numpy(),
                    'input_img': train_datas.tensors.cpu().numpy(),
                    #'psnr': batch_psnr(refer_datas.tensors, pred_img).cpu().numpy()
                }
                predict_results.append(batch_result)

                # 6. 内存优化
                del pred_img, train_datas, refer_datas, high_quality
                del clean, depth, under, underE
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # 7. 打印进度（仅主进程）
                if self.rank == 0 and (batch_id % 50 == 0):
                    logger.info(f"[PREDICT] Progress: {batch_id}/{len(dataloader)} | Processed: {img_num} images")

        # 8. 分布式同步
        if not self.USE_FAKE_DDP and dist.is_available() and dist.is_initialized():
            dist.barrier()

        # 9. 仅主进程输出预测总结
        if self.rank == 0:
            predict_time = time.time() - predict_start_time
            avg_time_per_img = predict_time / img_num if img_num > 0 else 0
            logger.info(
                f"🎯 Prediction phase {phase} completed | "
                f"Total images: {img_num} | "
                f"Total time: {predict_time:.2f}s | "
                f"Avg time per img: {avg_time_per_img:.4f}s"
            )

            # 保存预测结果统计
            with open(os.path.join(self.predict_save_dir, 'predict_summary.txt'), 'w') as f:
                f.write(f"Prediction Phase: {phase}\n")
                f.write(f"Total images: {img_num}\n")
                f.write(f"Total time: {predict_time:.2f}s\n")
                f.write(f"Average time per image: {avg_time_per_img:.4f}s\n")

        # 切换回训练模式
        self.model.train()

        return predict_results

    # 2. 学习率更新函数优化（聚焦Proj-J/CTDM）
    def update_optimizer_lr(self, optimizer, epoch, iterations, phase):
        # 动态获取CTDM、Relight乘数（解耦，各自适配）
        ctdm_multi = get_ctdm_multiplier(epoch)
        relight_multi = get_relight_multiplier(epoch)  # 新增：获取Relight动态乘数

        if phase == 1:
            # [base, CTDM, Proj-J, relight] → Relight替换为动态乘数
            lr_multipliers = [1.0, ctdm_multi, 1.0, relight_multi]
        elif phase == 2:
            lr_multipliers = [1.0, 1.0, 1.0, relight_multi]  # 同上
        else:
            lr_multipliers = [1.0, 1.0, 1.0, relight_multi]  # 所有阶段均用动态乘数

        lr = self.scheduler.get_lr(epoch=epoch, curr_iter=iterations)
        # Proj-J LR：epoch506续训适配（506≤550高LR冲刺，550后降LR稳定）
        if epoch <= 550:
            proj_j_lr = 6e-5
        else:
            proj_j_lr = 4e-5

        learning_rates = {
            'base': lr * lr_multipliers[0],
            'CTDM': lr * lr_multipliers[1],
            'Proj_J': proj_j_lr,
            'relight': lr * lr_multipliers[3]  # 自动适配动态乘数，无需手动修改
        }
        # 原有LR更新逻辑（完全不变）
        matched_names = []
        for param_group in optimizer.param_groups:
            if 'name' in param_group and param_group['name'] in learning_rates:
                param_group['lr'] = learning_rates[param_group['name']]
                matched_names.append(param_group['name'])
        if self.rank == 0:
            unmatched = [k for k in learning_rates if k not in [pg['name'] for pg in optimizer.param_groups]]
            if unmatched:
                print(f"[LR Update Warning] unmatched：{unmatched}")
        return optimizer

    def record_learning_rates(self, optimizer, epoch: int) -> Dict[str, List[float]]:
        lr = []
        curve_counter = 1
        current_curve_name = [] # getattr(self.opts, 'scheduler_lr_name', None)
        # 1. 遍历参数组收集LR（所有Rank执行）
        for param_group in optimizer.param_groups:
            # 确保LR是浮点数（防止张量类型）
            current_lr = param_group['lr']
            lr.append(float(current_lr) if isinstance(current_lr, (int, float, torch.Tensor)) else 0.0)

            # 收集当前参数组的曲线名称
            if 'name' in param_group and param_group['name']:
                current_curve_name.append(param_group['name'])
            else:
                current_curve_name.append(f"curve{curve_counter}")
                curve_counter += 1
        # 2. 初始化曲线名称（仅第一次执行时保存，保证全程名称一致）
        if not self.curve_names and current_curve_name:
            self.curve_names = current_curve_name
        # 3. 追加LR历史，并限制最大长度（避免内存溢出）
        self.lr_history.append(lr)
        max_history_len = 500  # 可根据需求调整
        if len(self.lr_history) > max_history_len:
            self.lr_history = self.lr_history[-max_history_len:]  # 保留最近500轮

        # 3. 仅主进程执行LR可视化（避免多Rank重复绘图/覆盖）
        if self.rank == 0 and epoch > 50:
            self.plot_lr(self.lr_history, self.curve_names, epoch)  # 传入epoch和历史曲线名
            #logger.info(f"📌 LR曲线绘制完成 | Epoch={epoch} | 曲线数={len(curve_name)}")

        return {'learning_rates': lr, 'curve_names': current_curve_name}

    def plot_lr(self, lr_list: List[List[float]], curve_name: List[str] = None, epoch: int = None):
        """
        绘制LR曲线（纯CPU执行，适配DDP）
        :param lr_list: 嵌套列表，如[[1e-4,1e-5], [1e-4,1e-5]]（必须是CPU数值）
        :param curve_name: 曲线名称列表
        :return:
        """

        # 2. 强制转到CPU并转为普通数值（防止传入GPU张量/张量对象）
        def to_cpu_num(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().item()  # 张量转CPU标量
            return float(x) if isinstance(x, (int, float)) else 0.0  # 异常值兜底为0

        # 处理所有数值，确保格式正确
        lr_list = [[to_cpu_num(num) for num in sub_list] for sub_list in lr_list]
        # 转置：从[epoch数, 参数组数] → [参数组数, epoch数]
        lr_list = list(map(list, zip(*lr_list)))

        # 2. 绘图（纯CPU逻辑）
        x_list = list(range(len(lr_list[0])))
        fig, ax = plt.subplots(figsize=(8, 5))  # 指定画布大小，避免压缩
        for i, y_values in enumerate(lr_list):
            label = curve_name[i] if curve_name and i < len(curve_name) else f'Curve {i + 1}'
            ax.plot(x_list, y_values, label=label, linewidth=1.5)  # 增加线宽更清晰

        # 优化可视化细节
        ax.set_title('Learning Rate Schedule', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Learning Rate', fontsize=10)
        ax.set_yscale('log')  # LR通常用对数坐标，更易看变化
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # 3. 保存+关闭画布（释放CPU内存）
        save_path = os.path.join(self.train_result, 'lr_list.png')
        # 确保保存目录存在（避免FileNotFoundError）
        os.makedirs(self.train_result, exist_ok=True)
        plt.savefig(
            save_path,
            bbox_inches='tight',
            dpi=150,  # 提高分辨率，图片更清晰
            pad_inches=0.1
        )
        plt.close(fig)  # 显式关闭画布，避免内存泄漏
        print(f"LR曲线已保存至: {save_path}")
    def _setup_training_config(self):
        """设置训练配置参数"""

        self.metric_names = getattr(self.opts, "stats_name", ['loss', 'psnr', 'ssim'])
        # 训练监控
        self.train_stats = Statistics(metric_names=self.metric_names)
        self.val_stats = Statistics(metric_names=self.metric_names)

    def _setup_logging_and_saving(self):
        """设置日志记录和模型保存路径"""
        common_save_dir = getattr(self.opts, "common_save_dir", "./results")
        self.train_result = '{}/{}'.format(common_save_dir, "weights")

        self.loss_iter = '{}/{}'.format(common_save_dir, "loss_iter.txt")
        self.loss_epoch = '{}/{}'.format(common_save_dir, "loss_epoch.txt")

        self.loss_val = '{}/{}'.format(common_save_dir, "loss_val.txt")
        self.loss_predict = '{}/{}'.format(common_save_dir, "loss_predict.txt")


        # ========== DDP同步：主进程创建目录 + 所有进程等待 ==========
        if dist.is_available() and dist.is_initialized():
            # 核心优化：所有进程先等主进程完成全局目录创建（和外层逻辑对齐）
            dist.barrier()
            # 兜底校验：无论rank，只要目录不存在就创建（仅1行，极简容错）
            pathlib.Path(self.train_result).mkdir(parents=True, exist_ok=True)

            # 仅主进程初始化日志文件（保留原有逻辑，避免重复写入）
            if self.rank == 0:
                with open(self.loss_epoch, 'a') as f:
                    f.write('\nModel config\n')
                    for k, v in vars(self.opts).items():
                        f.write(f'\n{k}={v}')
                print(f"[Rank {self.rank}] ✅ 目录校验完成 + 日志初始化: {self.train_result}")
        else:
            # 单卡模式：保持原有逻辑（无冗余）
            pathlib.Path(self.train_result).mkdir(parents=True, exist_ok=True)
            with open(self.loss_epoch, 'a') as f:
                f.write('\nModel config\n')
                for k, v in vars(self.opts).items():
                    f.write(f'\n{k}={v}')

    def _save_checkpoint(self, epoch: int, train_metrics: Dict):
        """保存检查点（适配分布式/伪DDP）

        Args:
            epoch: 当前epoch
            train_metrics: 训练指标（含GPU张量）
        """
        # 1. 仅主进程执行保存逻辑（核心！避免多Rank重复保存）
        if self.rank != 0:
            return

        # 2. 读取配置参数
        min_checkpoint_metric = getattr(self.opts, "stats_checkpoint_metric_min", True)
        checkpoint_frequency = getattr(self.opts, "checkpoint_frequency", 10)

        # 4. GPU张量安全转数值（detach→cpu→item）
        tmp_loss = round(train_metrics['total_loss'].detach().cpu().item(), 4)

        # 5. 判断是否为最优模型
        is_best = tmp_loss <= self.best_metric
        """
        if is_best:
            self.best_metric = tmp_loss
            logger.info(f"📌 New best metric (Epoch {epoch}): {self.best_metric:.6f}")
        """
        # 6. DDP模型解包（核心！适配多卡训练）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # 7. 保存检查点
        save_checkpoint(
            iterations=self.train_iterations,
            epoch=epoch,
            model=model_to_save,  # 传入解包后的模型
            optimizer=self.optimizer,
            gradient_scalar=self.gradient_scaler,
            best_metric=self.best_metric,
            is_best=is_best,
            save_dir=self.train_result,
            max_ckpt_metric=min_checkpoint_metric,
            rank=self.rank,  # 传递当前进程rank
            USE_FAKE_DDP=self.USE_FAKE_DDP  # 传递伪DDP标识

        )

        logger.info(f"✅ Checkpoint saved at epoch {epoch} | Save dir: {self.train_result}")

    def _update_seed_per_epoch(self, epoch):
        """每个epoch更新一次随机种子（极简版）"""
        # 核心：种子 = 基础种子 + rank + epoch（保证每个epoch、每个进程种子都不同）
        base_seed = getattr(self.opts, "common_seed", 0)
        new_seed = base_seed + self.rank + epoch  # 仅加了epoch
        # 重新设置种子（覆盖初始值，保证每个epoch随机性）
        random.seed(new_seed)
        torch.manual_seed(new_seed)
        np.random.seed(new_seed)
        # 可选：GPU种子（如果用CUDA）
        if torch.cuda.is_available():
            torch.cuda.manual_seed(new_seed)  # 给当前进程的GPU设种子
            torch.cuda.manual_seed_all(new_seed)  # 给所有GPU设种子（兜底）

# ========== Relight专属LR乘数（复用线性插值逻辑，单独超参数，epoch506续训适配） ==========
def get_relight_multiplier(epoch):
    start_epoch = 500    # 乘数动态调整起始epoch
    boost_epoch = 510    # 乘数拉升结束、衰减起始epoch
    end_epoch = 600      # 乘数衰减结束epoch
    pre_500_multi = 6.0  # epoch<500：固定乘数，慢更适配
    boost_start = 6.0    # 500epoch乘数起始值
    boost_end = 10.0     # 510epoch乘数峰值（大LR快速收敛新权重）
    end_multi = 3.0      # 600epoch乘数最小值（后期慢更防过拟合）

    if epoch < start_epoch:
        # 500前：固定低乘数，Relight随基础网络慢更
        return pre_500_multi
    elif start_epoch <= epoch <= boost_epoch:
        # 500-510epoch：线性拉升（6.0→10.0），快速提升LR，适配新权重收敛
        progress = (epoch - start_epoch) / (boost_epoch - start_epoch)
        multiplier = boost_start + (boost_end - boost_start) * progress
        return round(multiplier, 1)  # 保留1位小数，避免乘数震荡
    elif boost_epoch < epoch <= end_epoch:
        # 510-600epoch：线性衰减（10.0→3.0），逐步降LR，后期稳更
        progress = (epoch - boost_epoch) / (end_epoch - boost_epoch)
        multiplier = boost_end - (boost_end - end_multi) * progress
        return round(multiplier, 1)
    else:
        # 600后：固定最小乘数，收尾阶段稳定更新
        return end_multi

# ========== 核心：动态计算CTDM乘数（500-600epoch线性衰减） ==========
def get_ctdm_multiplier(epoch):
    start_epoch = 500
    end_epoch = 600
    start_multiplier = 72
    end_multiplier = 10
    # 新增：500-510epoch先小幅提升，再衰减（避免骤升）
    if epoch < 500:
        return 1.8
    elif 500 <= epoch <= 510:
        # 500-510epoch：从30→72，逐步拉高
        progress = (epoch - 500) / 10
        multiplier = 30 + (72 - 30) * progress
        return round(multiplier, 1)
    elif 510 < epoch <= 600:
        # 510-600epoch：从72→10，线性衰减
        progress = (epoch - 510) / 90
        multiplier = 72 - (72 - 10) * progress
        return round(multiplier, 1)
    else:
        return 10

