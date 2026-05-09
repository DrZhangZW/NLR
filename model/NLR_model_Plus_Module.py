import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange
from torch.fft import fft2, ifft2
from timm.layers import to_ntuple, trunc_normal_, DropPath, to_2tuple, LayerNorm
from typing import Union, Dict, List, Optional, Tuple
from utils.common_utils import visualize_tensor, show_tensor
from model.rank_GPU import data_fit
from model.loss_functions import loss_CTDM, loss_relight
from utils import logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from torch import tensor


class Multi_Scaler_Solver(nn.Module):
    def __init__(self,
                 dim: int,  # 输入特征维度（通道数），如64/128/256等
                 num_heads: int,  # 注意力头数，控制多头注意力的分组数量
                 mlp_ratio: float = 4.,  # MLP扩展比例（隐藏层维度=dim*mlp_ratio）
                 sr_ratio: int = 1,  # 空间缩减比例（键值对下采样率，1表示无下采样）
                 linear_attn: bool = False,  # 是否启用线性复杂度注意力（降低计算成本）
                 qkv_bias: bool = True,  # 是否在QKV计算中添加偏置项
                 proj_drop: float = 0.,  # 投影层的dropout率（默认0表示不丢弃）
                 drop_path: float = 0.,  # 随机深度衰减率
                 act_layer=nn.GELU,  # 激活函数类型（默认GELU）
                 norm_layer=nn.LayerNorm,  # 归一化层类型（默认LayerNorm）
                 transformer=False,
                 CTDM_use=False,
                 depths: int = 1,
                 light=True
                 ):
        super().__init__()
        self.subproblem_solver = SubproblemSolver(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                  sr_ratio=sr_ratio, linear_attn=linear_attn, qkv_bias=qkv_bias,
                                                  proj_drop=proj_drop, drop_path=drop_path, act_layer=act_layer,
                                                  norm_layer=norm_layer, learnable_params=True, transformer=transformer,
                                                  CTDM_use=CTDM_use, depths=depths, light=True)

    def forward(self, x, A_feat, t_hat_feat, high_feat, t_prev, b_prev, J_prev, mask, mask_y, stage):
        """
        :param x: 当前层的输入特征 [B, C, H, W] (I 的特征)
        :param mask: [B,1,H_ori,W_ori]
        :param A_feat:  当前尺度的 A 特征 [B, C, H, W] (不更新)

        :param t_hat_feat: 当前尺度的 t_hat 特征 [B, C, H, W] (不更新)
        :param t_prev: 前一个块的变量状态
        :param b_prev: 前一个块的变量状态
        :param J_prev: 前一个块的变量状态
        :return:
        """
        t_new, b_new, J_new, tmp_J, total_loss = self.subproblem_solver(
            x, A_feat, t_hat_feat, high_feat, t_prev, b_prev, J_prev, mask, mask_y, stage)

        return t_new, b_new, J_new, tmp_J, total_loss


class SubproblemSolver(nn.Module):
    """Solver for all subproblems"""

    def __init__(self,
                 dim: int,  # 输入特征维度（通道数），如64/128/256等
                 num_heads: int,  # 注意力头数，控制多头注意力的分组数量
                 mlp_ratio: float = 4.,  # MLP扩展比例（隐藏层维度=dim*mlp_ratio）
                 sr_ratio: int = 1,  # 空间缩减比例（键值对下采样率，1表示无下采样）
                 linear_attn: bool = False,  # 是否启用线性复杂度注意力（降低计算成本）
                 qkv_bias: bool = True,  # 是否在QKV计算中添加偏置项
                 proj_drop: float = 0.,  # 投影层的dropout率（默认0表示不丢弃）
                 drop_path: float = 0.,  # 随机深度衰减率
                 act_layer=nn.GELU,  # 激活函数类型（默认GELU）
                 norm_layer=nn.LayerNorm,  # 归一化层类型（默认LayerNorm）
                 transformer=False,
                 CTDM_use=False,
                 learnable_params=True,
                 depths: int = 1,
                 light=True):
        super().__init__()
        # Learnable parameters
        if learnable_params:
            self.alpha = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
            self.beta = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
            self.gamma = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
            self.sigma = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
            self.mu = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
            self.eta = nn.Parameter(torch.tensor(0.5).clamp(min=0.001))
        else:
            self.register_buffer('alpha', torch.tensor(1.0))
            self.register_buffer('beta', torch.tensor(1.0))
            self.register_buffer('mu', torch.tensor(1.0))
            self.register_buffer('h', torch.tensor(1.0))
            self.register_buffer('sigma', torch.tensor(1.0))
            self.register_buffer('eta', torch.tensor(1.0))
        # Gradient calculator
        self.gradient_calculator = GradientCalculator()
        # -----------------------------------------------------
        self.transformer = transformer
        self.CTDM_use = CTDM_use
        if transformer:
            self.block = nn.ModuleList(
                [Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       sr_ratio=sr_ratio, linear_attn=linear_attn, qkv_bias=qkv_bias,
                       drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer) for _ in range(depths)])
        else:
            self.block = nn.ModuleList([
                Res_block(dim, dim) for _ in range(depths)])
        self.Proj_J_channel_up = channel_up(int(dim / 4))
        self.Proj_J_channel_down = channel_down(int(dim / 4))

        if CTDM_use:
            """
            self.CTDM_retinex = Retinex_decom(channels=32, num_heads=num_heads, mlp_ratio=1.5,
                                              sr_ratio=1, linear_attn=linear_attn,
                                              drop_path=drop_path, act_layer=act_layer,
                                              norm_layer=norm_layer,
                                              transformer_use=transformer)
            """
            # self.up = nn.Conv2d(64, dim, 3, 1, 1, bias=False)
            self.relight = Relight(channels=32,depths=1)

    def forward(self, I, A, t_hat, high_feat, t_prev, b_prev, J_prev, mask, mask_y, stage):

        # J subproblem (using MSA module)
        if self.transformer:
            J_new = self.j_subproblem(I, A, t_hat, J_prev, self.block, mask)
        else:
            J_new = self.j_subproblem_conv(I, A, t_hat, J_prev, self.block, mask)
        # ----
        total_loss = torch.tensor(0.0, device=J_new.device)
        tmp_J = self.Proj_J_channel_down(J_new)
        #visualize_tensor(tmp_J)

        if self.CTDM_use:
            tmp_J, total_loss = self.process_ctdm(tmp_J, high_feat, mask, mask_y, stage)
            # tmp_J = 0.5*tmp_J + 0.5*self.Proj_J_channel_down(J_new)
            J_new = 0.8 * J_new + 0.2 * self.Proj_J_channel_up(tmp_J)
        # else:

        # Solve all subproblems in sequence
        with torch.no_grad():
            # u subproblem
            u_new = self.u_subproblem(t_prev, b_prev[:, 0:3], b_prev[:, 3:6])
            # t subproblem
            t_new = self.t_subproblem(t_hat, I, A, u_new[:, 0:1], u_new[:, 1:2], tmp_J,
                                      b_prev[:, 0:3], b_prev[:, 3:6])
            # b subproblem
            b_new = self.b_subproblem(t_new, u_new, b_prev)
        # ========== 新增：删除临时变量（u_new用不到了） ==========
        del u_new
        return t_new, b_new, J_new, tmp_J, total_loss
    # ===================== 适配后的process_ctdm函数（核心：弃用CTDM_retinex）=====================
    def process_ctdm(self, low_feat, high_feat, mask, mask_y, stage):
        device = low_feat.device
        is_paired = False if stage >= 3 else True  # 阶段化配对/非配对
        relight_weight = 0.1 if stage == 2 else 0.2  # Relight损失阶段化权重
        relight_1d_loss_weight = 0.06  # 1维光照对齐损失权重

        # 1. 梯度隔离：保护基础网络，仅用无梯度副本
        low_feat_for_loss = low_feat.detach().clone()
        high_feat_for_loss = high_feat.detach().clone()

        # 2. 纯物理Retinex分解（替代原CTDM_retinex，无参数、无梯度）
        low_R, low_L = self.physical_retinex_decompose(low_feat_for_loss, mask)
        high_R, high_L = self.physical_retinex_decompose(high_feat_for_loss, mask_y)

        # 4. Relight2前向与损失计算（梯度封闭在CTDM内部）
        relight_loss = torch.tensor(0.0, device=device)
        relight_1d_loss = torch.tensor(0.0, device=device)
        ctdm_out_feat = low_feat_for_loss  # 兜底：无Relight时返回原始特征
        if (stage in [1, 2, 3, 4]) and high_L is not None:
            # high_L预处理：梯度隔离+高斯模糊去残差，避免伪影
            high_L_detach = high_L.detach().clone()
            high_L_detach = gaussian_blur_2d(high_L_detach, kernel_size=5, sigma=1.5)
            high_L_detach = high_L_detach.expand_as(low_L)

            # 拼接输入：[low_L.detach(), high_L_detach] → 6通道，适配Relight2
            concat_feat = torch.cat((low_L.detach(), high_L_detach), dim=1)

            # Relight2前向：训练返(优化光照,1维损失)，推理返优化光照
            relight_out, relight_1d_loss = self.relight(concat_feat)

            # 计算Relight损失（仅保留该损失，约束光照优化方向）

            relight_loss = loss_relight(
                low_R.detach(), low_L.detach(), relight_out, high_L_detach,
                paired=is_paired, mask=mask, stage=stage
            ) * relight_weight

            # 生成CTDM输出特征：Retinex物理规律（反射率×优化后光照），无内容伪影
            ctdm_out_feat = low_R.detach() * relight_out

            # 及时释放显存
            del high_L_detach, concat_feat, relight_out

        # ===================== 核心修改：总损失仅融合Relight损失+1维光照对齐损失 =====================
        # 原ctdm_total_loss被移除，总损失无其他成分
        total_loss = relight_loss + relight_1d_loss_weight * relight_1d_loss
        # ==========================================================================================

        # 最终梯度隔离：CTDM输出完全解耦，不污染基础网络
        low_feat = ctdm_out_feat.detach()

        # 释放所有内部张量，优化显存
        del low_feat_for_loss, high_feat_for_loss, low_R, low_L, high_R, high_L, ctdm_out_feat
        return low_feat, total_loss

    def j_subproblem(self, I, A, t, J_prev, msa_module, m):
        """Solve for J subproblem using MSA module"""
        # Convert to [B, H, W, C] format for MSA
        # Convert to [B, H, W, C] format for MSA
        J_prev_permuted = J_prev.permute(0, 2, 3, 1)  # (bs,h,w,c)
        B, H, W, C = J_prev_permuted.shape
        mask = F.interpolate(m[None].float(), size=[H, W]).to(torch.bool)[0]
        mask_flat = mask.flatten(1)  # 重命名，方便后续删除
        J_prev_permuted = J_prev_permuted.reshape(B, -1, C)  # bsx(hw)xC
        # --------------------------------------------------------------
        spatial_shapes = []
        spatial_shape = (H, W)
        spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=J_prev.device)  # [h,w]

        for _, block in enumerate(msa_module):
            J_prev_permuted = block(src=J_prev_permuted,
                                    src_key_padding_mask=mask_flat,
                                    src_spatial_shapes=spatial_shapes)
        J_prev_permuted = J_prev_permuted.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # bsxcxhxw
        # J_prev_permuted = inverse_data_transform(J_prev_permuted)

        data_J = data_fit(I, A, t, mask)
        data_J = data_J.to(I.device)
        data_J = self.Proj_J_channel_up(data_J.tensors)
        out = (self.beta * data_J + J_prev_permuted) / (1 + self.beta)
        # ========== 新增：删除中间张量 ==========
        del J_prev_permuted, mask_flat, spatial_shapes, data_J
        # out = J_prev_permuted
        return out

    def j_subproblem_conv(self, I, A, t, J_prev, module, m):
        """Solve for J subproblem using MSA module"""
        # Convert to [B, H, W, C] format for MSA
        B, C, H, W = I.shape
        mask = F.interpolate(m[None].float(), size=[H, W]).to(torch.bool)[0]
        for _, block in enumerate(module):
            J_prev = block(J_prev)
        # J_prev = inverse_data_transform(J_prev)
        # -------------------
        data_J = data_fit(I, A, t, mask)
        data_J = data_J.to(I.device)
        data_J = self.Proj_J_channel_up(data_J.tensors)
        out = (self.beta * data_J + J_prev) / (1 + self.beta)
        # ========== 新增：删除中间张量 ==========
        del J_prev, mask, data_J
        # out = J_prev
        return out

    def _create_gradient_kernels_freq(self, H, W, device):
        """创建与输入相同尺寸的频域梯度核"""
        # Create spatial gradient kernels
        dx_spatial = torch.tensor([[0, 0, 0],
                                   [0, -1, 1],
                                   [0, 0, 0]], dtype=torch.float32, device=device)

        dy_spatial = torch.tensor([[0, 0, 0],
                                   [0, -1, 0],
                                   [0, 1, 0]], dtype=torch.float32, device=device)

        # 创建与输入相同尺寸的空张量
        dx_full = torch.zeros(1, 1, H, W, device=device)
        dy_full = torch.zeros(1, 1, H, W, device=device)

        # 将梯度核放置在中心位置
        center_h, center_w = H // 2, W // 2
        dx_full[:, :, center_h - 1:center_h + 2, center_w - 1:center_w + 2] = dx_spatial
        dy_full[:, :, center_h - 1:center_h + 2, center_w - 1:center_w + 2] = dy_spatial

        # Take FFT
        F_dx = FourierOperators.fft2(dx_full)
        F_dy = FourierOperators.fft2(dy_full)

        return F_dx, F_dy

    def t_subproblem(self, t_hat, I, A, u_x, u_y, J, b_x, b_y):
        """Solve for t bar subproblem"""
        B, C, H, W = t_hat.shape
        device = t_hat.device

        # 动态创建正确尺寸的梯度核
        F_dx, F_dy = self._create_gradient_kernels_freq(H, W, device)

        # Compute F_N
        F_N = (torch.conj(F_dx) * FourierOperators.fft2(u_x - b_x / self.sigma) +
               torch.conj(F_dy) * FourierOperators.fft2(u_y - b_y / self.sigma))

        # Compute F_D
        F_D = (torch.conj(F_dx) * F_dx + torch.conj(F_dy) * F_dy)

        # Compute numerator and denominator
        numerator = (FourierOperators.fft2(t_hat) +
                     self.beta * FourierOperators.fft2((I - J) / (A - J + 1e-8)) +
                     self.sigma * F_N)

        denominator = 1 + self.beta + self.sigma * F_D
        denominator = torch.where(denominator.abs() < 1e-8,
                                  torch.ones_like(denominator),
                                  denominator)

        t_new = FourierOperators.ifft2(numerator / denominator).real
        t_new = 0.9*t_hat + 0.1*torch.clamp(t_new, 1e-4, 0.999)
        del F_dx, F_dy, F_N, F_D, numerator, denominator
        return t_new

    def compute_gradients(self, x):
        """Compute gradients in x and y directions for multi-channel input"""
        return self.gradient_calculator(x)

    def u_subproblem(self, t, b_x, b_y):
        """Solve for u subproblem — FIXED VERSION"""
        grad_x, grad_y = self.compute_gradients(t)

        # 1. 降低阈值，防止全0！
        sigma = self.sigma + 1e-8
        mu = self.mu * 0.01  # 👈 缩小mu，关键修复
        # mu = self.mu      # 如果你想恢复原版就用这行

        arg_x = grad_x + b_x / sigma
        arg_y = grad_y + b_y / sigma

        # 2. 软阈值
        theta = mu / sigma
        u_x_new = torch.sign(arg_x) * torch.clamp(torch.abs(arg_x) - theta, min=0)
        u_y_new = torch.sign(arg_y) * torch.clamp(torch.abs(arg_y) - theta, min=0)

        # 可视化看看是否还有0
        # print("u_x_new abs mean:", u_x_new.abs().mean().item())

        return torch.cat([u_x_new, u_y_new], dim=1)

    def b_subproblem(self, t, u, b_prev):
        """Solve for b subproblem"""
        grad_x, grad_y = self.compute_gradients(t)
        b_x_new = b_prev[:, 0:1] + self.sigma * (grad_x - u[:, 0:1])
        b_y_new = b_prev[:, 1:2] + self.sigma * (grad_y - u[:, 1:2])
        out = torch.cat([b_x_new, b_y_new], dim=1)
        # ========== 新增：删除中间张量 ==========
        del grad_x, grad_y, b_x_new, b_y_new

        return out


class feature_pyramid2(nn.Module):
    def __init__(self, embed_dims):
        super(feature_pyramid2, self).__init__()
        # ---------------level 0:  1-------------------------
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # conv + downsample
        level0 = x
        level1 = self.downsample(level0)
        level2 = self.downsample(level1)
        level3 = self.downsample(level2)
        level4 = self.downsample(level3)

        return level0, level1, level2, level3, level4


class feature_pyramid(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super(feature_pyramid, self).__init__()
        # ---------------level 0:  1-------------------------
        self.convs = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=(5, 5), stride=(1, 1), padding=2),
                                   nn.Conv2d(channels[0], channels[0], kernel_size=(5, 5), stride=(1, 1), padding=2))
        self.block0 = Res_block(channels[0], channels[0])
        # --------------level 1: 1/2---------------------
        self.down0 = nn.Conv2d(channels[0], channels[0], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.block1 = Res_block(channels[0], channels[1])
        # self.downsample1 = nn.Conv2d(embed_dims[0], embed_dims[1], 2, 2, 0, bias=False)  # 64-->128
        # --------------level 2: 1/4-----------------------
        self.down1 = nn.Conv2d(channels[1], channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.block2 = Res_block(channels[1], channels[2])
        # --------------level 4: 1/8-----------------------
        self.down2 = nn.Conv2d(channels[2], channels[2], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.block3 = Res_block(channels[2], channels[3])
        # --------------level 4: 1/16-----------------------
        self.down3 = nn.Conv2d(channels[3], channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.block4 = Res_block(channels[3], channels[4])

    def forward(self, x):
        #  downsample + conv

        level0 = self.block0(self.convs(x))
        level1 = self.block1(self.down0(level0))
        level2 = self.block2(self.down1(level1))
        level3 = self.block3(self.down2(level2))
        level4 = self.block4(self.down3(level3))
        return level0, level1, level2, level3, level4

    def forward_2(self, x):
        # conv + downsample

        level0 = self.convs(x)
        level1 = self.down0(self.block0(level0))  # 1/2
        level2 = self.down1(self.block1(level1))  # 1/4
        level3 = self.down2(self.block2(level2))  # 1/8

        return level0, level1, level2, level3


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Relight2(nn.Module):
    def __init__(self, channels, depths=1):
        super(Relight2, self).__init__()
        self.conv0 = nn.Conv2d(6, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(*[
            Res_block(channels, channels) for _ in range(depths)
        ])
        self.conv1_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)
                                     )
        self.alpha = nn.Parameter(torch.tensor(0.01).clamp(max=0.5, min=0.01))
        self.alpha_max = 0.15  # 上界更低，限制最大调整幅度
        self.alpha_min = 0.05  # 下界防止为0

    def forward(self, x):
        # x：输入为 [low_L.detach(), high_L_detach] 拼接，shape=[B,6,H,W]
        feats = self.conv0(x)
        feats = self.blocks0(feats)
        outs = self.conv1_1(feats)

        # 【修改2】outs先做sigmoid+数值约束，保证生成的光照符合0.01~1.0的合法范围，避免异常值引入残差
        L_gen = torch.sigmoid(outs)
        L_gen = torch.clamp(L_gen, 0.01, 1.0)  # 与基础模块的光照约束一致

        # 【修改3】对可学习alpha做硬约束，强制限制在[0.0001, 0.2]区间，避免模型学出大幅调整
        alpha_clamped = torch.clamp(self.alpha, self.alpha_min, self.alpha_max)

        # 【修改4】核心融合策略：「原始low_L为主 + 生成光照的残差微调」（替代原有直接加权）
        # 逻辑：x[:, :3, :, :] 是原始low_L.detach()，仅用生成的L_gen做「残差级修正」，而非直接替换
        # 优势：low_L的主体信息完全保留，仅补充极轻微的调整，从根本上避免另一张图的残差被带入
        low_L_ori = x[:, :3, :, :]  # 提取原始low_L（拼接的前3个通道）
        L = low_L_ori + alpha_clamped * (L_gen - low_L_ori)  # 残差微调：L = (1-alpha)*low_L + alpha*L_gen
        # 最终再约束一次，保证光照合法性，避免残差导致的数值溢出
        L = torch.clamp(L, 0.01, 1.0)

        return L

class Relight(nn.Module):
    def __init__(self, channels=32, depths=1, use_1d_loss=True):
        """
        初始化参数说明：
        :param channels: 中间特征通道数，与解码器/CTDM特征维度匹配（建议32，与原配置一致）
        :param depths: 分支内Res_block堆叠数（建议1，与解码器depths匹配，减少计算）
        :param use_1d_loss: 是否开启1维全局光照对齐损失（默认True，强化光照规律学习）
        """
        super(Relight, self).__init__()
        self.use_1d_loss = use_1d_loss  # 1维损失开关（训练开启，推理自动屏蔽）

        # ===================== 核心：高低光照双分支解耦特征提取 =====================
        # low_L分支：仅处理待优化光照（输入前3通道），保留自身局部细节，无high_L干扰
        self.conv0_low = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.blocks0_low = nn.Sequential(*[Res_block(channels, channels) for _ in range(depths)])
        # high_L分支：仅处理光照参考（输入后3通道），专注提取光照特征，后续剥离局部内容
        self.conv0_high = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.blocks0_high = nn.Sequential(*[Res_block(channels, channels) for _ in range(depths)])

        # ===================== 1维全局光照特征提取模块（仅作用于high_L分支）=====================
        # 2D→1维：自适应平均池化，抹平high_L所有局部内容，仅保留全局光照统计
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 1维特征轻量化变换：通道减半→残差学习→通道恢复，强化全局光照规律
        self.one_dim_transform = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Res_block(channels // 2, channels // 2),
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        # 1维→2D：轻量卷积升维，将全局光照特征映射到low_L的空间维度（全局一致，无局部内容）
        self.up_sample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # ===================== 光照生成层：仅融合low_L局部+high_L全局光照 =====================
        # 输入为解耦后的融合特征，输出3通道光照调整图，无任何high_L局部内容
        self.conv1_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # ===================== 可学习alpha：残差微调约束（防止过度光照修正）=====================
        # 初始值0.01，硬约束在0.05~0.15，仅做极轻微残差调整，保证low_L原始光照主导
        self.alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha_min = 0.05
        self.alpha_max = 0.15

    def forward(self, x):
        """
        前向传播说明：
        :param x: 输入张量 [B,6,H,W]，固定格式=拼接[low_L.detach(), high_L.detach()]，前3=待优化光照，后3=光照参考
        :return: 训练阶段返回 (优化后光照L_optim [B,3,H,W], 1维光照对齐损失relight_1d_loss)
                 推理阶段仅返回 优化后光照L_optim [B,3,H,W]
        """
        # 分离输入：严格解耦low_L（待优化）和high_L（仅光照参考），均做detach防止梯度回传
        low_L_ori = x[:, :3, :, :].detach()  # 原始待优化光照 [B,3,H,W]
        high_L_ref = x[:, 3:, :, :].detach() # 纯光照参考 [B,3,H,W]，无任何内容传递

        # ===================== 步骤1：双分支独立特征提取（彻底无交叉干扰）=====================
        # low_L分支：提取自身局部特征，保留原始细节/内容，全程无high_L特征混入
        feats_low = self.conv0_low(low_L_ori)
        feats_low = self.blocks0_low(feats_low)  # [B,channels,H,W] → 仅low_L的局部特征
        # high_L分支：提取自身特征，为后续纯全局光照特征做准备
        feats_high = self.conv0_high(high_L_ref)
        feats_high = self.blocks0_high(feats_high)# [B,channels,H,W] → 仅high_L的特征

        # ===================== 步骤2：high_L→纯全局光照特征（彻底剥离局部内容）=====================
        feats_high_1d = self.global_pool(feats_high)  # [B,channels,H,W] → [B,channels,1,1]，抹平所有局部内容
        feats_high_1d = self.one_dim_transform(feats_high_1d)  # 强化全局光照规律学习
        # 1维特征广播到low_L空间维度 + 轻量卷积细化，生成全局一致的光照特征图（无任何局部细节）
        feats_high_1d_expand = feats_high_1d.expand(-1, -1, low_L_ori.shape[2], low_L_ori.shape[3])
        feats_high_global = self.up_sample(feats_high_1d_expand)  # [B,channels,H,W] → 纯全局光照特征

        # ===================== 步骤3：特征融合（仅low_L局部 + high_L纯全局光照）=====================
        # 核心：无任何high_L局部内容特征，从源头根除非配对伪影
        feats_fuse = feats_low + feats_high_global

        # ===================== 步骤4：1维全局光照对齐损失（仅训练阶段计算）=====================
        relight_1d_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        if self.use_1d_loss and self.training:
            # 构造high_L自拼接输入，提取纯光照参考的1维特征（与训练分支同路径）
            high_L_self = torch.cat([high_L_ref, high_L_ref], dim=1)
            feats_high_ref = self.conv0_high(high_L_self[:, 3:, :, :])
            feats_high_ref = self.blocks0_high(feats_high_ref)
            feats_high_ref_1d = self.global_pool(feats_high_ref)
            # MSE损失：强制训练的high_L全局光照特征向优质参考对齐，强化纯光照学习
            relight_1d_loss = F.mse_loss(feats_high_1d, feats_high_ref_1d.detach())

        # ===================== 步骤5：光照生成 + 极轻微残差微调 =====================
        # 生成光照调整图，sigmoid+数值约束保证光照物理合理性（0.01~1.0，避免过暗/过曝）
        outs = self.conv1_1(feats_fuse)
        L_gen = torch.sigmoid(outs)
        L_gen = torch.clamp(L_gen, 0.01, 1.0)

        # alpha硬约束：仅做5%~15%的残差调整，low_L原始光照占比85%~95%，兜底防止过度修正
        alpha_clamped = torch.clamp(self.alpha, self.alpha_min, self.alpha_max)
        # 残差融合：原始low_L为主，生成光照为辅，无任何外部内容引入
        L_optim = low_L_ori + alpha_clamped * (L_gen - low_L_ori)
        L_optim = torch.clamp(L_optim, 0.01, 1.0)  # 最终光照数值约束

        # 训练/推理分支：训练返优化光照+1维损失，推理仅返优化光照（无额外计算）
        if self.training:
            return L_optim, relight_1d_loss
        else:
            return L_optim, relight_1d_loss

# ========== 先复制这个引导滤波函数（放在模型类前） ==========
def guided_filter(x, guide, kernel_size=5, eps=1e-3):
    """
    引导滤波（保边去噪）：PyTorch实现，适配4维批量张量 [B,C,H,W]
    :param x: 待滤波张量（如你的init_reflectance），shape=[B,C,H,W]
    :param guide: 引导张量（如你的原始x），shape=[B,C,H,W]，需和x同形状
    :param kernel_size: 滤波核大小，建议3/5/7，默认5
    :param eps: 正则化系数，防止分母为0，默认1e-3
    :return: 滤波后的张量，shape和x完全一致
    """
    # 统一通道数：引导图取灰度（若为3通道RGB），保证滤波核与通道匹配
    if guide.size(1) > 1:
        guide = torch.mean(guide, dim=1, keepdim=True)  # RGB→灰度图 [B,1,H,W]
    if x.size(1) > 1:
        x_gray = torch.mean(x, dim=1, keepdim=True)    # 待滤波图取灰度 [B,1,H,W]
    else:
        x_gray = x

    # 定义均值滤波核（替代高斯核，效果接近且计算更快）
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=x.dtype, device=x.device) / (kernel_size ** 2)
    padding = kernel_size // 2  # 等宽填充，保证输出尺寸和输入一致

    # 步骤1：计算均值（引导图、待滤波图、引导图×待滤波图、引导图×引导图）
    mean_guide = F.conv2d(guide, kernel, padding=padding, groups=1)
    mean_x = F.conv2d(x_gray, kernel, padding=padding, groups=1)
    mean_guide_x = F.conv2d(guide * x_gray, kernel, padding=padding, groups=1)
    mean_guide_guide = F.conv2d(guide * guide, kernel, padding=padding, groups=1)

    # 步骤2：计算协方差和方差
    cov_guide_x = mean_guide_x - mean_guide * mean_x  # 协方差
    var_guide = mean_guide_guide - mean_guide * mean_guide + eps  # 方差（加正则化）

    # 步骤3：计算滤波系数
    a = cov_guide_x / var_guide  # 增益系数
    b = mean_x - a * mean_guide  # 偏置系数

    # 步骤4：系数均值化（保证边缘平滑）
    mean_a = F.conv2d(a, kernel, padding=padding, groups=1)
    mean_b = F.conv2d(b, kernel, padding=padding, groups=1)

    # 步骤5：生成滤波结果（适配多通道，广播到原通道数）
    output = mean_a * guide + mean_b
    return output.expand_as(x)  # 扩展通道数和输入x一致 [B,C,H,W]
# ========== 新增：高斯模糊层（可复用，专门用于光照去纹理） ==========

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

class Retinex_decom(nn.Module):
    def __init__(self,
                 channels: int = 1,
                 num_heads: int = 1,  # 注意力头数，控制多头注意力的分组数量
                 depths: int = 1,
                 mlp_ratio: float = 4.,  # MLP扩展比例（隐藏层维度=dim*mlp_ratio）
                 sr_ratio: int = 1,  # 空间缩减比例（键值对下采样率，1表示无下采样）
                 linear_attn: bool = False,  # 是否启用线性复杂度注意力（降低计算成本）
                 qkv_bias: bool = True,  # 是否在QKV计算中添加偏置项
                 proj_drop: float = 0.,  # 投影层的dropout率（默认0表示不丢弃）
                 drop_path: float = 0.,  # 随机深度衰减率
                 act_layer=nn.GELU,  # 激活函数类型（默认GELU）
                 norm_layer=nn.LayerNorm,  # 归一化层类型（默认LayerNorm）
                 transformer_use=False
                 ):
        super(Retinex_decom, self).__init__()

        # self.adjust = nn.Conv2d(channels, 64, 3, 1, 1, bias=False)
        # channels = 64
        self.transformer_use = transformer_use

        # R
        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(*[
            Res_block(channels, channels) for _ in range(depths)
        ])
        # L
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.blocks1 = nn.Sequential(*[
            Res_block(channels, channels) for _ in range(depths)
        ])

        self.conv0_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)
                                     )

        self.conv1_1 = nn.Sequential(Res_block(channels, channels),  # 适配light_channels
                                     nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1))

        if transformer_use:
            self.cross_attention = PVTv2CrossAttention(channels,
                                                       num_heads=num_heads,
                                                       sr_ratio=sr_ratio,
                                                       qkv_bias=qkv_bias,
                                                       dropout=proj_drop,
                                                       )
            self.self_attention = PVTv2CrossAttention(channels,
                                                      num_heads=num_heads,
                                                      sr_ratio=sr_ratio,
                                                      qkv_bias=qkv_bias,
                                                      dropout=proj_drop,
                                                      )
        else:
            self.self_attention = Res_block(channels, channels)



    def forward(self, x, mask):
        B, C, H, W = x.shape  # x: [B,3,H,W]

        if mask is not None:
            valid_mask = 1.0 - F.interpolate(mask.unsqueeze(1).float(), size=[H, W], mode='nearest').repeat(1, 3, 1, 1)
        else:
            valid_mask = torch.ones(B, 3, H, W, device=x.device)

        x = x * valid_mask

        # ========== 1. 保存初始反射率（用于后续加权） ==========
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]  # [B,1,H,W]
        init_reflectance = x / (init_illumination + 1e-6)  # [B,3,H,W]（和x维度一致）
        #init_reflectance = guided_filter(init_reflectance, x, kernel_size=5, eps=1e-3)
        #init_reflectance = torch.clamp(init_reflectance, 0.0, 1.0)  # 约束值域，防止数值失真
        # 不再删除init_reflectance，保留用于加权
        del x  # 仅释放x，保留init_reflectance/init_illumination


        Reflectance  =self.blocks0(self.conv0(init_reflectance))
        Illumination =self.blocks1(self.conv1(init_illumination))

        Illumination = gaussian_blur_2d(Illumination, kernel_size=7, sigma=2)  # 推荐5核，sigma1.5

        if self.transformer_use == False:
            Illumination_content = self.self_attention(Illumination)
            Reflectance_final = self.conv0_1(Reflectance + Illumination_content)
            Illumination_final = self.conv1_1(Illumination - Illumination_content)
            del Illumination_content
        else:
            #Reflectance_final = self.cross_attention_process(Illumination, Reflectance, mask)
            Illumination_content = self.self_attention_process(Illumination, mask)

            Reflectance_final = self.conv0_1(Reflectance + Illumination_content)
            Illumination_final = self.conv1_1(Illumination - Illumination_content)
            del Illumination_content

        # 4. 反射率R生成：核心优化+残差学习
        R = (torch.tanh(Reflectance_final) + 1.0) / 2.0
        R = 0.9*init_reflectance + 0.1*R
        R = torch.clamp(R, 0.0, 1.0)  # 新增：兜底约束，避免数值越界
        # 步骤1：网络学习光照原始输出，约束数值避免sigmoid饱和
        Illumination_final = torch.clamp(Illumination_final, -5.0, 5.0)  # 新增：和R分支对齐，约束网络输出


        L = torch.sigmoid(Illumination_final)  # 网络输出的原始光照
        L = torch.clamp(L, 0.01, 1.0)  # 新增：约束0.01~1.0，避免光照过暗（除0）或过亮
        L = gaussian_blur_2d(L, kernel_size=3, sigma=1.0)  # 轻量模糊，不破坏光照整体亮度
        L = torch.cat([L for _ in range(3)], dim=1)  # 扩展为[B,3,H,W]

        # 仅有效区域保留结果，padding置0
        if mask is not None:
            R = R * valid_mask
            L = L * valid_mask

        return R, L

    def cross_attention_process(self, Illumination, Reflectance, mask):
        Illumination = Illumination.permute(0, 2, 3, 1)  # (bs,h,w,c)
        B, H, W, C = Illumination.shape
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=[H, W]).to(torch.bool)[0]
            mask = mask.flatten(1)  # bsx(hw)

        Illumination = Illumination.reshape(B, -1, C)  # bsx(hw)xC
        Reflectance = Reflectance.permute(0, 2, 3, 1)  # (bs,h,w,c)
        Reflectance = Reflectance.reshape(B, -1, C)  # bsx(hw)xC
        # --------------------------------------------------------------
        spatial_shapes = []
        spatial_shape = (H, W)
        spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=Illumination.device)  # [h,w]
        Reflectance = self.cross_attention(query=Illumination,
                                           key=Reflectance,
                                           value=Reflectance,
                                           key_padding_mask=mask,
                                           feat_size=spatial_shapes.tolist()[0])

        Reflectance = Reflectance.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # bsxcxhxw
        return Reflectance

    def self_attention_process(self, Illumination, mask):
        Illumination = Illumination.permute(0, 2, 3, 1)  # (bs,h,w,c)
        B, H, W, C = Illumination.shape
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=[H, W]).to(torch.bool)[0]
            mask = mask.flatten(1)  # bsx(hw)

        Illumination = Illumination.reshape(B, -1, C)  # bsx(hw)xC

        # --------------------------------------------------------------
        spatial_shapes = []
        spatial_shape = (H, W)
        spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=Illumination.device)  # [h,w]
        Reflectance = self.self_attention(query=Illumination,
                                          key=Illumination,
                                          value=Illumination,
                                          key_padding_mask=mask,
                                          feat_size=spatial_shapes.tolist()[0])
        Reflectance = Reflectance.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # bsxcxhxw
        return Reflectance


class Block(nn.Module):
    def __init__(self,
                 dim: int,  # 输入特征维度（通道数），如64/128/256等
                 num_heads: int,  # 注意力头数，控制多头注意力的分组数量
                 mlp_ratio: float = 4.,  # MLP扩展比例（隐藏层维度=dim*mlp_ratio）
                 sr_ratio: int = 1,  # 空间缩减比例（键值对下采样率，1表示无下采样）
                 linear_attn: bool = False,  # 是否启用线性复杂度注意力（降低计算成本）
                 qkv_bias: bool = True,  # 是否在QKV计算中添加偏置项
                 proj_drop: float = 0.,  # 投影层的dropout率（默认0表示不丢弃）
                 drop_path: float = 0.,  # 随机深度衰减率
                 act_layer=nn.GELU,  # 激活函数类型（默认GELU）
                 norm_layer=nn.LayerNorm  # 归一化层类型（默认LayerNorm）
                 ):
        super().__init__()
        self.hidden_dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = PVTv2CrossAttention(dim,
                                        num_heads=num_heads,
                                        sr_ratio=sr_ratio,
                                        qkv_bias=qkv_bias,
                                        dropout=proj_drop,
                                        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpWithDepthwiseConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            extra_relu=linear_attn,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, src, src_key_padding_mask, src_spatial_shapes=None):
        src2 = self.norm1(src)
        src2 = self.attn(query=src2, key=src2, value=src2, key_padding_mask=src_key_padding_mask,
                         feat_size=src_spatial_shapes.tolist()[0])  # q,k=q+src_pos,k+src_pos
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src), feat_size=src_spatial_shapes.tolist()[0]))
        return src


class PVTv2CrossAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            sr_ratio: int = 1,
            dropout: float = 0.1,
            qkv_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        # 空间降维模块
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.sr = None

        # 交叉注意力核心
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            batch_first=True
        )

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            feat_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """实现带空间降维的交叉注意力
        Args:
            query: [B, N_q, C]
            key: [B, N_k, C]
            key_padding_mask: [B, N_k] (True表示需要mask的位置)
            feat_size: (H, W) key对应的原始特征图尺寸
        """
        # 处理key的空间降维
        if self.sr is not None and feat_size is not None:
            B, N_k, C = key.shape
            H, W = feat_size
            key = key.permute(0, 2, 1).reshape(B, C, H, W)
            key = self.sr(key).reshape(B, C, -1).permute(0, 2, 1)
            key = self.norm(key)

            value = value.permute(0, 2, 1).reshape(B, C, H, W)
            value = self.sr(value).reshape(B, C, -1).permute(0, 2, 1)
            value = self.norm(value)

            # 同步处理mask
            if key_padding_mask is not None:
                """
                key_padding_mask = key_padding_mask.view(B, H, W)

                key_padding_mask = F.max_pool2d(
                    key_padding_mask.float(),
                    kernel_size=self.sr_ratio,
                    stride=self.sr_ratio
                ).flatten(1).bool()
                """
                key_padding_mask = key_padding_mask.reshape(B, H, W)
                key_padding_mask = F.avg_pool2d(
                    key_padding_mask.float().unsqueeze(1),
                    kernel_size=self.sr_ratio,
                    stride=self.sr_ratio
                ).squeeze(1) > 0.5

                key_padding_mask = key_padding_mask.flatten(1)  # 展平为 [B, N_k_reduced]

        # 执行交叉注意力
        attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        return attn_output


class MlpWithDepthwiseConv(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.,
            extra_relu=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU() if extra_relu else nn.Identity()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_size: List[int]):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, feat_size[0], feat_size[1])
        x = self.relu(x)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GradientCalculator(nn.Module):
    """Module to compute gradients for multi-channel inputs"""

    def __init__(self):
        super(GradientCalculator, self).__init__()
        # Create gradient kernels for x and y directions
        grad_x_kernel = torch.tensor([[-1, 0, 1]]).float()
        grad_y_kernel = torch.tensor([[-1], [0], [1]]).float()

        # Expand to handle multiple channels
        self.register_buffer('grad_x_kernel', grad_x_kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grad_y_kernel', grad_y_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        Compute gradients for multi-channel input
        x: [B, C, H, W]
        Returns: grad_x, grad_y: [B, C, H, W] each
        """
        batch_size, channels, height, width = x.shape

        # Expand kernels to match input channels
        grad_x_kernel = self.grad_x_kernel.repeat(channels, 1, 1, 1)
        grad_y_kernel = self.grad_y_kernel.repeat(channels, 1, 1, 1)

        # Use group convolution to compute gradients for each channel separately
        grad_x = nn.functional.conv2d(x, grad_x_kernel, padding=(0, 1), groups=channels)
        grad_y = nn.functional.conv2d(x, grad_y_kernel, padding=(1, 0), groups=channels)

        return grad_x, grad_y


class FourierOperators:
    """FFT operators for the subproblems"""

    @staticmethod
    def fft2(x):
        return fft2(x, dim=(-2, -1))

    @staticmethod
    def ifft2(x):
        return ifft2(x, dim=(-2, -1))


class downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(downsampling, self).__init__()
        if pooling:
            self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, 1),  # 1x1卷积调整通道数
                nn.LeakyReLU(0.2, True)
            )

        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # nn.Tanh()
            )

    def forward(self, x):
        out = self.conv(x)
        return out


class upsampling(nn.Module):
    """可配置的上采样模块"""

    def __init__(self, in_channels, out_channels, method='bilinear'):
        super().__init__()
        self.method = method

        if method == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=method, align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.method == 'transpose':
            return self.act(self.up(x))
        else:
            return self.act(self.conv(self.up(x)))


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=1.5, use_groupnorm=True):
        super().__init__()

        # 扩展维度，确保至少为分组数的倍数
        hidden_dim = max(int(in_channels * expansion_ratio), 32)

        # 统一分组数配置
        groups = 4
        hidden_groups = min(groups, hidden_dim // 4)  # 确保每组至少4个通道
        out_groups = min(groups, out_channels // 4)

        # 主路径：1x1升维 -> 3x3深度可分离卷积 -> 1x1降维
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=hidden_groups, num_channels=hidden_dim)

        # 优化：使用 1x1 + 3x3 替代纯深度卷积（减少内存访问）
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        )
        self.norm2 = nn.GroupNorm(num_groups=hidden_groups, num_channels=hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=out_groups, num_channels=out_channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 残差捷径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=out_groups, num_channels=out_channels)
            )

        # 自适应残差权重
        self.alpha = nn.Parameter(torch.ones(1))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        identity = self.shortcut(x)

        # 主路径前向传播
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leaky_relu(out)

        out = self.depthwise_conv(out)
        out = self.norm2(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.norm3(out)

        # 残差连接，在相加前保持激活状态
        out = self.alpha * out + identity
        out = self.leaky_relu(out)  # 在残差相加后应用激活函数

        return out


class Res_block2(nn.Module):
    def __init__(self, in_channels, out_channels, use_groupnorm=True):
        super().__init__()

        # 主路径卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # 归一化层替代方案
        if use_groupnorm:
            self.norm1 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
            # self.norm1 = nn.BatchNorm2d(out_channels)
            # self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.LayerNorm([out_channels, 1, 1])  # 伪LayerNorm实现
            self.norm2 = nn.LayerNorm([out_channels, 1, 1])

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 残差捷径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.AdaptiveAvgPool2d(1)  # 通道对齐辅助
            )

        # 自适应残差权重
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.alpha * out + identity
        return self.leaky_relu(out)


class channel_down2(nn.Module):
    def __init__(self, channels):
        super(channel_down2, self).__init__()

        # 主路径 - 深度可分离卷积
        self.depthwise_conv0 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1,
                                         groups=channels * 4)
        self.pointwise_conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1)

        # 改进：添加GroupNorm层
        self.norm0 = nn.GroupNorm(num_groups=min(8, channels * 2), num_channels=channels * 2)  # 动态分组

        self.depthwise_conv1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1,
                                         groups=channels * 2)
        self.pointwise_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        # 最终输出层
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=1)
        self.norm_out = nn.GroupNorm(num_groups=1, num_channels=3)  # 输出通道少，用1组

        # 激活函数
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 轻量化的残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels * 4, 3, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=3)
        )

    def forward(self, x):
        # 主路径 - 深度可分离卷积
        out = self.depthwise_conv0(x)
        out = self.pointwise_conv0(out)
        out = self.norm0(out)  # 添加归一化
        out = self.relu(out)

        out = self.depthwise_conv1(out)
        out = self.pointwise_conv1(out)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv_out(out)
        out = self.norm_out(out)

        # 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut

        # 输出激活
        out = (torch.tanh(out) + 1.0) / 2.0

        return out


class channel_down(nn.Module):
    def __init__(self, channels):
        super(channel_down, self).__init__()

        # 主路径卷积层
        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

        # 改进的GroupNorm配置 - 确保每组至少有4个通道
        self.norm0 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels * 2)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels // 2), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=3)  # 输出通道少，用较少分组

        # 激活函数
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 残差连接路径
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels * 4, 3, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=3)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主路径
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = out * self.ca(out)  # 注意力加权

        out = self.conv2(out)
        out = self.norm2(out)

        # 残差连接路径
        shortcut = self.shortcut(x)

        # 残差连接 - 关键改进
        out = out + shortcut
        # out = self.relu(out)  # 残差连接后也应用激活

        # 使用sigmoid确保输出在合理范围
        out = (torch.tanh(out) + 1.0) / 2.0

        return out


class channel_up(nn.Module):
    def __init__(self, channels):
        super(channel_up, self).__init__()

        # 基础卷积层
        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # 激活函数
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 残差连接路径
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, channels * 4, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=8, num_channels=channels * 4)
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 主路径
        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        # 残差连接路径
        shortcut = self.shortcut(x)

        # 相加后激活
        out = self.alpha * out + shortcut
        out = self.relu(out)

        return out


class channel_up2(nn.Module):
    def __init__(self, channels):
        super(channel_up2, self).__init__()

        # 深度可分离卷积层
        self.depthwise_conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        self.pointwise_conv0 = nn.Conv2d(3, channels, kernel_size=1)

        self.depthwise_conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.pointwise_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)

        # 最终输出层
        self.conv_out = nn.Conv2d(channels * 2, channels * 4, kernel_size=1)

        # 激活函数
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 轻量化残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, channels * 4, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=channels * 4)
        )

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels * 4 // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4 // 8, channels * 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主路径 - 深度可分离卷积
        out = self.depthwise_conv0(x)
        out = self.pointwise_conv0(out)
        out = self.relu(out)

        out = self.depthwise_conv1(out)
        out = self.pointwise_conv1(out)
        out = self.relu(out)

        out = self.conv_out(out)

        # 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)

        # 通道注意力
        attention_weights = self.channel_attention(out)
        out = out * attention_weights

        # out = (torch.tanh(out) + 1.0) / 2.0
        return out


def inverse_data_transform(X):
    return (torch.tanh(X) + 1.0) / 2.0
