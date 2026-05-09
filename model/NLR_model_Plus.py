from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import to_ntuple, trunc_normal_, DropPath, to_2tuple, LayerNorm
import torch.nn.functional as F
from model.NLR_model_Plus_Module import feature_pyramid, feature_pyramid2, Multi_Scaler_Solver, upsampling, \
    downsampling

from model.rank_GPU import tensor_rank1_ddp
from utils.common_utils import visualize_tensor, show_tensor
from model.loss_functions import loss_aux_J
from torch.utils.checkpoint import checkpoint


class NLRNET_Plus(nn.Module):
    def __init__(self,
                 in_nc=3,
                 iters=(1, 1, 1, 1, 1),
                 depths=(1, 2, 2, 2, 2),
                 embed_dims=(16, 32, 64, 128, 192),
                 num_heads=(1, 2, 4, 8, 8),
                 sr_ratios=(8, 4, 2, 2, 1),
                 mlp_ratios=(1.5, 1.5, 2, 1.5, 1.5),
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):  # 改用原生LayerNorm，减少显存波动
        """
        pvt-v2-base
        :param depths: 元组类型，定义4个阶段的Transformer块堆叠深度
        :param embed_dims: 元组类型，定义下采样上采样阶段的特征维度变化
        :param num_heads: 元组类型，定义下采样上采样阶段的注意力头数配置
        :param sr_ratios: 元组类型，定义下采样上采样阶段的空间下采样率
        :param mlp_ratios: 元组类型，定义下采样上采样阶段的MLP扩展比例
        :param qkv_bias: 布尔值，控制是否在QKV计算中使用偏置
        """
        super(NLRNET_Plus, self).__init__()

        self.encoder = NLRNET_Plus_encoder(in_nc=in_nc, iters=iters, depths=depths, embed_dims=embed_dims,
                                           num_heads=num_heads, sr_ratios=sr_ratios, mlp_ratios=mlp_ratios,
                                           qkv_bias=qkv_bias, proj_drop_rate=proj_drop_rate,
                                           attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=drop_path_rate,
                                           norm_layer=norm_layer)
        self.decoder = NLRNET_Plus_decoder(in_nc=in_nc, iters=iters, depths=depths, embed_dims=embed_dims,
                                           num_heads=num_heads, sr_ratios=sr_ratios, mlp_ratios=mlp_ratios,
                                           qkv_bias=qkv_bias, proj_drop_rate=proj_drop_rate,
                                           attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=drop_path_rate,
                                           norm_layer=norm_layer)

    def forward(self, x0, y0,phase):
        # 编码器前向传播 - 保持计算图连接
        multi_t_b_J_result = self.encoder(x0, y0,phase)
        # 解码器前向传播
        pred_result, loss_aux, loss_cmd = self.decoder(multi_t_b_J_result,phase)
        return pred_result, loss_aux, loss_cmd

class NLRNET_Plus_encoder(nn.Module):
    def __init__(self,
                 in_nc=3,
                 iters=(2, 2, 2, 2, 2),  # 补充第5个iter（对应body）
                 depths=(2, 2, 2, 2, 2),
                 embed_dims=(32, 64, 128, 256, 384),  # 补充第5个维度（对应body）
                 num_heads=(1, 2, 4, 8, 16),
                 sr_ratios=(8, 4, 2, 1, 1),
                 mlp_ratios=(2., 3., 2., 2., 2.),
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super(NLRNET_Plus_encoder, self).__init__()
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)
        assert len(embed_dims) == num_stages, f"embed_dims length {len(embed_dims)} != num_stages {num_stages}"

        # 基础配置
        self.aux_train = True
        self.num_scales = num_stages  # 5个尺度（0-4）
        self.embed_dims = embed_dims

        # 金字塔特征提取
        self.pyramid2 = feature_pyramid2(embed_dims)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 各尺度网络层
        self.J_head = nn.Conv2d(6, embed_dims[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.m_down1 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 sr_ratio=sr_ratios[0], transformer=False, CTDM_use=False, depths=depths[0], light=True)
             for _ in range(iters[0])])
        self.downsample2 = downsampling(embed_dims[0], embed_dims[1], pooling=False)
        self.m_down2 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 sr_ratio=sr_ratios[1], transformer=False, CTDM_use=False, depths=depths[1], light=True)
             for _ in range(iters[1])])
        self.downsample3 = downsampling(embed_dims[1], embed_dims[2], pooling=False)
        self.m_down3 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 sr_ratio=sr_ratios[2], transformer=True, CTDM_use=False, depths=depths[2], light=True)
             for _ in range(iters[2])])
        self.downsample4 = downsampling(embed_dims[2], embed_dims[3], pooling=False)
        self.m_down4 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 sr_ratio=sr_ratios[3], transformer=True, CTDM_use=False, depths=depths[3], light=False)
             for _ in range(iters[3])])
        self.downsample5 = downsampling(embed_dims[3], embed_dims[4], pooling=False)
        self.m_body = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[4], num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 sr_ratio=sr_ratios[4], transformer=True, CTDM_use=False, depths=depths[4], light=False)
             for _ in range(iters[4])])

        # 初始化训练相关变量（避免未定义）
        self.aux_J = []
        self.total_loss = torch.tensor(0.0)

    def _initialize_variables(self, b, c, h1, w1, device, x):
        """Initialize variables for subproblems（优化：减少无用张量+设备对齐）"""
        with torch.no_grad():
            # 直接在目标设备创建，匹配输入dtype，避免类型转换
            t_down_1 = torch.zeros((b, 3, h1, w1), device=device, dtype=x.dtype)
            b_down_1 = torch.zeros((b, 6, h1, w1), device=device, dtype=x.dtype)

        J_down_1 = self.J_head(x)
        del x  # 释放已用完的输入张量
        return (t_down_1, b_down_1, J_down_1)

    def forward(self, x0, y0,stage):
        """
        :param x0.tensors x0.mask, low quality image
        :param y0.tensors,y0.mask, high quality image
        :return:
        """
        mask = x0.mask
        y0 = self._adjust_y0_size(x0, y0)
        mask_y =y0.mask
        with torch.no_grad():
            # 初始化金字塔
            J_init,t_hat_init, A_init = self._initialize_transmission_and_airlight(x0)
            # 提取高质量图像特征
            pyramid_data = self._extract_pyramid_features(x0.tensors, A_init.tensors, t_hat_init.tensors)
            high_features = self._get_pyramid_features(y0.tensors, self.pyramid2)
        # 初始化变量: t,b,J
        b = mask.shape[0]
        _, c, h1, w1 = high_features[0].shape
        device = high_features[0].device
        t_b_J_var = self._initialize_variables(
            b, c, h1, w1, device, torch.concat((x0.tensors, J_init.tensors), dim=1)
        )
        #t_b_J_var = self._initialize_variables(b, c, h1, w1, device, x0.tensors)
        outs = {}
        outs["-1"] = t_b_J_var
        outs["-2"] = J_init.tensors
        #t_b_J_var = self._initialize_variables(b, c, h1, w1, device,  x0.tensors)
        if self.aux_train:
            self.aux_J = []
            self.total_loss = torch.tensor(0.0, device=device)
        # 整合多尺度特征
        multi_scale_feats = self._integrate_high_features(pyramid_data, high_features)

        del A_init, t_hat_init, high_features, pyramid_data, x0, y0, J_init

        # 第一尺度处理
        t_b_J_var = self._process_single_scale(
            self.m_down1, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale=0)
        outs["0"] = t_b_J_var

        # 第二尺度处理
        t_b_J_var = self._down_process_sigle_scale(
            self.m_down2, self.downsample2, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale=1)
        outs["1"] = t_b_J_var

        # 第三尺度处理
        t_b_J_var = self._down_process_sigle_scale(
            self.m_down3, self.downsample3, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale=2)
        outs["2"] = t_b_J_var

        # 第四尺度处理
        t_b_J_var = self._down_process_sigle_scale(
            self.m_down4, self.downsample4, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale=3)
        outs["3"] = t_b_J_var

        # 最底层处理
        t_b_J_var = self._down_process_sigle_scale(
            self.m_body, self.downsample5, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale=4)
        outs["4"] = t_b_J_var
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return {
            't_b_J_vars': outs,
            'multi_scale_feats': multi_scale_feats,
            'total_loss': self.total_loss,
            'aux_J': self.aux_J,
            'x0_mask': mask,
            'y0_mask': mask_y,
        }

    def _down_process_sigle_scale(self, blocks, downsample, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale):
        t_b_J_var = list(t_b_J_var)
        # 下采样到第二尺度 - 使用现有的下采样层
        with torch.no_grad():
            t_b_J_var[0] = self.downsample(t_b_J_var[0])
            t_b_J_var[1] = self.downsample(t_b_J_var[1])
        t_b_J_var[2] = downsample(t_b_J_var[2])
        t_b_J_var = tuple(t_b_J_var)
        outs = self._process_single_scale(blocks, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale)
        return outs

    def _process_single_scale(self, blocks, multi_scale_feats, t_b_J_var, mask, mask_y, stage, scale):
        """单尺度处理"""
        t_var, b_var, J_var = t_b_J_var
        for i, block in enumerate(blocks):
            # x, A_feat, t_hat_feat, high_feat, t_prev, b_prev, J_prev, mask
            outputs = block(multi_scale_feats[scale]['I_feat'],
                            multi_scale_feats[scale]['A_feat'],
                            multi_scale_feats[scale]['t_hat_feat'],
                            multi_scale_feats[scale]['high_feat'],
                            t_var, b_var, J_var, mask, mask_y, stage)
            t_var_new, b_var_new, J_var_new, tmp_J, loss_block = outputs
            t_var, b_var, J_var = t_var_new, b_var_new, J_var_new
            del t_var_new, b_var_new, J_var_new  # 释放中间张量
        # 定期清理GPU缓存
        if scale % 2 == 0 and t_var.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if self.aux_train:
            self.aux_J.append(tmp_J)
            self.total_loss = self.total_loss + loss_block

        return t_var, b_var, J_var

    def _adjust_y0_size(self, x0, y0) -> torch.Tensor:
        """调整y0尺寸以匹配x0（全程no_grad，减少梯度图）"""
        with torch.no_grad():  # 尺寸调整无梯度，核心优化
            h, w = x0.tensors.size()[-2:]
            y0_h, y0_w = y0.tensors.shape[-2:]

            if (y0_h, y0_w) != (h, w):
                return self.scale_y0(y0, h, w)
        return y0

    def scale_y0(self, y0, h, w):
        """缩放y0（修正device参数错误，保留其他显存优化）"""
        # 1. 提取输入张量的设备和类型（用于后续校验，不传入interpolate）
        input_tensor = y0.tensors
        device = input_tensor.device
        dtype = input_tensor.dtype

        # 2. 插值tensors（一步到位，核心优化保留）
        # ✅ 修正：移除device/dtype参数，interpolate自动继承输入的设备/类型
        y0.tensors = F.interpolate(
            input_tensor,  # 直接使用输入张量，避免重复引用
            size=(h, w),
            mode='bilinear',
            align_corners=False,
            antialias=True  # 仅bilinear模式支持，提升质量且不增加显存
        )

        # 3. 处理mask（避免中间变量，核心优化保留）
        if hasattr(y0, 'mask') and y0.mask is not None:
            # ✅ 修正：移除device参数，mask插值后自动和输入同设备
            y0.mask = F.interpolate(
                y0.mask.float().unsqueeze(0),  # 增加batch维度
                size=(h, w),
                mode='nearest'  # mask用nearest模式，避免模糊
            ).squeeze(0)  # 移除batch维度

            # 直接应用mask，无中间张量（核心优化保留）
            # ✅ 额外优化：确保mask和tensors设备/类型一致
            y0.mask = y0.mask.to(device=device, dtype=dtype, non_blocking=True)
            y0.tensors = y0.tensors * (1 - y0.mask)

        # 4. 释放无用中间张量（核心显存优化）
        del input_tensor
        return y0

    def _initialize_transmission_and_airlight(self, x0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始化传输图和大气光参数"""

        device = x0.tensors.device
        J_init, t_hat_init, A_init = tensor_rank1_ddp(x0.tensors, x0.mask, device)
        # 设备一致性检查
        if t_hat_init.tensors.device != x0.tensors.device:
            t_hat_init = t_hat_init.to(x0.tensors.device)
            A_init = A_init.to(x0.tensors.device)
            J_init = J_init.to(x0.tensors.device)
        del x0  # 释放无用引用
        return J_init, t_hat_init, A_init

    def _get_pyramid_features(self, tensor, pyramid_func):
        """提取金字塔特征，避免内存占用"""
        return tuple(pyramid_func(tensor))

    def _extract_pyramid_features(self, I_tensor, A_tensor, t_hat_tensor):
        """提取多尺度金字塔特征"""
        I_features = self._get_pyramid_features(I_tensor, self.pyramid2)
        A_features = self._get_pyramid_features(A_tensor, self.pyramid2)
        t_hat_features = self._get_pyramid_features(t_hat_tensor, self.pyramid2)
        del I_tensor, A_tensor, t_hat_tensor
        return {
            'I_features': I_features,
            'A_features': A_features,
            't_hat_features': t_hat_features
        }

    def _integrate_high_features(self, multi_scale_feats, high_features):
        """
        将高质量特征整合到多尺度特征中
        优化内存使用，避免重复存储
        """
        integrated_feats = []

        # 假设每个金字塔有相同数量的尺度
        num_scales = len(high_features)

        for i in range(num_scales):
            # 创建尺度特定的特征字典
            scale_data = {
                'scale_index': i,
                'I_feat': multi_scale_feats['I_features'][i] if i < len(multi_scale_feats['I_features']) else None,
                'A_feat': multi_scale_feats['A_features'][i] if i < len(multi_scale_feats['A_features']) else None,
                't_hat_feat': multi_scale_feats['t_hat_features'][i] if i < len(
                    multi_scale_feats['t_hat_features']) else None,
                'high_feat': high_features[i] if i < len(high_features) else None
            }
            integrated_feats.append(scale_data)

        return integrated_feats


class NLRNET_Plus_decoder(nn.Module):
    def __init__(self,
                 in_nc=3,
                 iters=(2, 2, 2, 2),
                 depths=(1, 1, 1, 1),
                 embed_dims=(32, 64, 128, 192, 384),  # 补充第5个维度（匹配编码器）
                 num_heads=(1, 2, 4, 8),
                 sr_ratios=(8, 4, 4, 2),
                 mlp_ratios=(1., 1.5, 1., 1.,),
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super(NLRNET_Plus_decoder, self).__init__()
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)
        assert (len(embed_dims)) == num_stages
        self.aux_train = True

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 上采样层
        # --------------------------------up-level 4---------------------------------------------
        self.upsample4 = upsampling(embed_dims[4], embed_dims[3])
        self.m_up4 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 sr_ratio=sr_ratios[3], transformer=True, CTDM_use=True, depths=depths[3],light=False)
             for _ in range(iters[3])])
        # ---------------------------------up-level 3----------------------------------------------
        self.upsample3 = upsampling(embed_dims[3], embed_dims[2])
        self.m_up3 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 sr_ratio=sr_ratios[2], transformer=True, CTDM_use=True, depths=depths[2],light=False)
             for _ in range(iters[2])])
        # -----------------------------------up-level 2---------------------------------------------
        self.upsample2 = upsampling(embed_dims[2], embed_dims[1])
        self.m_up2 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 sr_ratio=sr_ratios[1], transformer=False, CTDM_use=True, depths=depths[1],light=True)
             for _ in range(iters[1])])
        # -----------------------------------up-level 1--------------------------------------
        self.upsample1 = upsampling(embed_dims[1], embed_dims[0])
        self.m_up1 = nn.ModuleList(
            [Multi_Scaler_Solver(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 sr_ratio=sr_ratios[0], transformer=False, CTDM_use=True, depths=depths[0],light=True)
             for _ in range(iters[0])])
        # -----outpt
        self.m_tail = nn.Conv2d(embed_dims[0], in_nc, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.m_tail.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, multi_scale_out,stage):
        multi_scale_feats = multi_scale_out['multi_scale_feats']
        t_b_J_vars = multi_scale_out['t_b_J_vars']
        mask = multi_scale_out['x0_mask']
        mask_y = multi_scale_out['y0_mask']
        if self.aux_train:
            self.total_loss = multi_scale_out['total_loss']
            self.aux_J = multi_scale_out['aux_J']

        t_b_J_var = t_b_J_vars['4']
        del multi_scale_out, multi_scale_feats[-1], t_b_J_vars['4']
        # 上采样到第四尺度
        t_b_J_var = self._up_process_sigle_scale(self.m_up4, self.upsample4, multi_scale_feats, t_b_J_var, t_b_J_vars,
                                                 mask,  mask_y,stage,scale=4)

        # t_b_J_var = t_b_J_vars['3']
        # del multi_scale_out, multi_scale_feats[-1], t_b_J_vars['3']

        # 第三尺度处理
        t_b_J_var = self._up_process_sigle_scale(self.m_up3, self.upsample3, multi_scale_feats, t_b_J_var, t_b_J_vars,
                                                 mask, mask_y,stage,scale=3)

        # 第二尺度处理
        t_b_J_var = self._up_process_sigle_scale(self.m_up2, self.upsample2, multi_scale_feats, t_b_J_var, t_b_J_vars,
                                                 mask,mask_y,stage, scale=2)

        # 第一尺度处理
        t_b_J_var = self._up_process_sigle_scale(self.m_up1, self.upsample1, multi_scale_feats, t_b_J_var, t_b_J_vars,
                                                 mask, mask_y,stage,scale=1)
        # pred = torch.concat((t_b_J_var[2],t_b_J_vars["-1"][2]),dim=1)
        pred_J = t_b_J_var[2] + t_b_J_vars["-1"][2]
        # 最终卷积+激活（一步到位）
        pred = self.m_tail(pred_J)
        pred = (torch.tanh(pred) + 1.0) / 2.0
        pred = 0.9 * pred + 0.1 * t_b_J_vars["-2"].to(pred.device)

        mask = mask.float().unsqueeze(1)  # [bs,1,h,w]
        pred = pred * (1 - mask)

        # 7. 释放所有无用变量（核心显存优化，保留你原有逻辑）
        # 注意：mask在loss_aux计算完成后再释放，避免提前销毁无法传入loss_aux_J
        del multi_scale_feats, t_b_J_var, pred_J
        del t_b_J_vars["-1"]  # 先不释放t_b_J_vars["-2"]，等loss_aux计算完成后再处理（可选）
        torch.cuda.synchronize()

        loss_aux = None
        if self.aux_train:
            # 核心修改1：调用优化后的loss_aux_J，传入mask（复用现有掩码，强化对齐+缓解噪点）
            loss_aux = loss_aux_J(pred, self.aux_J, mask=mask) if self.aux_J else torch.tensor(0.0, device=pred.device)

            # 核心修改2：loss_aux鲁棒兜底+设备/ dtype 对齐（避免None或格式不匹配报错）
            if loss_aux is None:
                loss_aux = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            else:
                loss_aux = loss_aux.to(pred.device, dtype=pred.dtype)

            # 损失归一化（保留你原有逻辑，避免数值爆炸）
            total_loss = self.total_loss / 4.0
            # 确保total_loss设备/ dtype 对齐（强化鲁棒性）
            total_loss = total_loss.to(pred.device, dtype=pred.dtype)
        else:
            total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 补充：释放剩余无用变量（完成loss_aux计算后，避免显存泄漏）
        del mask, t_b_J_vars["-2"]
        torch.cuda.empty_cache()

        return pred, loss_aux, total_loss

    def _up_process_sigle_scale(self, blocks, upsample, multi_scale_feats, t_b_J_var, t_b_J_vars, mask, mask_y, stage,
                                scale):
        """上采样+单尺度处理（核心优化：no_grad包裹上采样+及时释放）"""
        t_var, b_var, J_var = t_b_J_var
        target_scale = scale - 1

        # 优化：上采样无梯度，包裹no_grad
        with torch.no_grad():
            # 异步上采样，减少显存峰值
            up_t = self.upsample(t_var)
            up_b = self.upsample(b_var)
            # 加权融合（避免中间张量）
            new_t = 0.5 * up_t + 0.5 * t_b_J_vars[str(target_scale)][0]
            new_b = 0.5 * up_b + 0.5 * t_b_J_vars[str(target_scale)][1]
            # 释放上采样中间张量
            del up_t, up_b

        # 上采样J（有梯度，保留计算图）
        new_J = upsample(J_var) + t_b_J_vars[str(target_scale)][2]

        # 释放已用完的变量（核心显存优化）
        del t_var, b_var, J_var, t_b_J_var,t_b_J_vars[str(target_scale)]
        # 移除t_b_J_vars中已用的尺度，减少字典占用

        # 处理当前尺度
        outs = self._process_single_scale(
            blocks, multi_scale_feats, (new_t, new_b, new_J),
            mask, mask_y, stage, target_scale
        )

        # 释放融合后的中间张量
        del new_t, new_b, new_J
        return outs

    def _process_single_scale(self, blocks, multi_scale_feats, t_b_J_var, mask,mask_y,stage, scale):
        """单尺度处理"""
        t_var, b_var, J_var = t_b_J_var
        scale_feat = multi_scale_feats[scale]
        # 提取核心特征（避免重复索引）
        I_feat = scale_feat.get('I_feat', None)
        A_feat = scale_feat.get('A_feat', None)
        t_hat_feat = scale_feat.get('t_hat_feat', None)
        high_feat = scale_feat.get('high_feat', None)

        for i, block in enumerate(blocks):
            # 前向传播（避免原地修改输入）
            outputs = block(
                I_feat, A_feat, t_hat_feat, high_feat,
                t_var, b_var, J_var, mask, mask_y, stage
            )
            # 优化：避免原地赋值，防止DDP梯度错误
            t_var_new, b_var_new, J_var_new, tmp_J, loss_block = outputs
            t_var, b_var, J_var = t_var_new, b_var_new, J_var_new
            # 释放中间张量
            del t_var_new, b_var_new, J_var_new
        del multi_scale_feats[-1]
        # 定期清理GPU缓存
        if scale % 2 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            #torch.cuda.empty_cache()
        if self.aux_train:
            self.aux_J.append(tmp_J)
            self.total_loss = self.total_loss + loss_block
        del scale_feat, I_feat, A_feat, t_hat_feat, high_feat
        return t_var, b_var, J_var