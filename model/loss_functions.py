import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.common_utils import visualize_tensor
import kornia.color as kc
import kornia.filters as kf  # 新增：导入滤波模块
# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


class Hybrid_denoise_Loss(nn.Module):
    def __init__(self,
                 charbonnier_weight=1.0,
                 tv_init_weight=1.0,
                 decay_rate=0.01,
                 min_tv_weight=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()

        self.tv = TVLoss()
        self.tv_init_weight = tv_init_weight
        self.decay_rate = decay_rate
        self.min_tv_weight = min_tv_weight
        self.charbonnier_weight = charbonnier_weight

    def forward(self, x, y, epoch=None):
        return nn.functional.l1_loss(x, y)
        # 计算基础损失值
        # char_loss = self.charbonnier(x, y) + nn.functional.l1_loss(x,y)
        # tv_loss = self.tv(x)
        # 返回加权总损失
        # return char_loss + 0*tv_loss, char_loss,tv_loss


def compute_L_ref(R_low1, R_low2):
    """
    Compute reflectance consistency loss L_ref.

    Args:
        R_low_1 (torch.Tensor): R_low^1 with shape (B, C, H, W)
        R_low_2 (torch.Tensor): R_low^2 with shape (B, C, H, W)

    Returns:
        torch.Tensor: L_ref value
    """
    #loss = torch.nn.MSELoss()(R_low1, R_low2)
    loss = torch.nn.L1Loss()(R_low1,R_low2)
    return loss


def loss_aux_J(pred_img, tmp_J, mask=None):
    """
    无SSIM优化版：多尺度L1+梯度损失融合，带尺度权重+掩码过滤+设备对齐+鲁棒兜底
    核心：仅用原生PyTorch，强化数值拟合+结构/边缘监督，加速loss_aux_J下降
    Args:
        pred_img: 最终预测图像 [bs,3,h,w] (0-1区间)
        tmp_J: 多尺度目标特征列表 [bsx3xhxw, bsx3x(h/2)x(w/2), ...]
        mask: 无效区域掩码 [bs,1,h,w]（1=无效，0=有效）
    Returns:
        归一化后的多尺度融合损失值（与pred_img同设备、同dtype）
    """
    # 1. 鲁棒兜底：处理空值/格式异常
    if not isinstance(tmp_J, (list, tuple)) or len(tmp_J) == 0:
        return torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)
    if pred_img.dim() != 4 or pred_img.shape[1] != 3:
        return torch.tensor(0.0, device=pred_img.device, dtype=pred_img.dtype)
    
    total_loss = 0.0
    current_img = pred_img

    # 2. 强化尺度权重：提升大尺寸的监督占比，更注重细节拟合（比原权重更聚焦）
    scale_weights = [1.2, 1.0, 0.8, 0.5, 0.8, 1.0, 1.2]  # 两端大尺寸权重提升至1.2
    if len(scale_weights) != len(tmp_J):
        scale_weights = [1.2 for _ in range(len(tmp_J))]  # 兜底也用强化权重

    # 3. 定义原生梯度计算函数（Sobel算子，捕捉边缘/纹理，无新依赖）
    def compute_image_gradient(img):
        """
        计算图像梯度（单通道→扩展为3通道，保持维度一致）
        Args:
            img: [bs,3,h,w] 输入图像
        Returns:
            grad_img: [bs,3,h,w] 图像梯度（边缘信息）
        """
        # Sobel算子核（x方向+y方向，原生tensor，无需新包）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device)
        # 扩展维度适配卷积：[3,3] → [1,1,3,3]（适配groups卷积）
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # 卷积计算x/y方向梯度（groups=3，保持3通道独立计算，无信息丢失）
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=3)
        
        # 合并梯度（L2范数，更稳定）
        grad_img = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad_img

    # 4. 遍历多尺度tmp_J计算融合损失（L1+梯度损失）
    for i, target in enumerate(tmp_J):
        # 目标特征鲁棒兜底
        if target is None or target.dim() != 4 or target.shape[1] != 3:
            continue
        target = target.to(pred_img.device, dtype=pred_img.dtype)
        target_h, target_w = target.size(-2), target.size(-1)
        current_weight = scale_weights[i]

        # 5. 尺寸插值匹配（双线性+align_corners=False，缓解锯齿，与原逻辑一致）
        if target_h == current_img.size(-2) and target_w == current_img.size(-1):
            adjusted_img = current_img
        else:
            adjusted_img = F.interpolate(
                current_img, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )

        # 6. 掩码过滤有效区域（与原逻辑一致，保证兼容性）
        mask_scaled = None
        if mask is not None and mask.dim() == 4:
            mask_scaled = F.interpolate(
                mask.to(pred_img.device, dtype=pred_img.dtype),
                size=(target_h, target_w),
                mode='nearest'
            )
            mask_scaled = mask_scaled.expand_as(adjusted_img)

        # 7. 核心1：强化L1损失（仅有效区域，数值拟合）
        if mask_scaled is not None:
            l1_loss_per_pixel = F.l1_loss(adjusted_img, target, reduction='none')
            valid_l1_sum = (l1_loss_per_pixel * (1 - mask_scaled)).sum()
            valid_pixel = (1 - mask_scaled).sum() + 1e-8
            l1_loss = valid_l1_sum / valid_pixel
        else:
            l1_loss = F.l1_loss(adjusted_img, target, reduction='mean')

        # 8. 核心2：原生梯度损失（结构/边缘监督，替代SSIM，无新包）
        adjusted_grad = compute_image_gradient(adjusted_img)
        target_grad = compute_image_gradient(target)
        if mask_scaled is not None:
            grad_loss_per_pixel = F.l1_loss(adjusted_grad, target_grad, reduction='none')
            valid_grad_sum = (grad_loss_per_pixel * (1 - mask_scaled)).sum()
            grad_loss = valid_grad_sum / valid_pixel
        else:
            grad_loss = F.l1_loss(adjusted_grad, target_grad, reduction='mean')

        # 9. L1+梯度损失融合（8:2），兼顾数值稳定和结构拟合（无需调整）
        current_loss = 0.8 * l1_loss + 0.2 * grad_loss
        total_loss += current_weight * current_loss

    # 10. 损失归一化，保证数值稳定（与原逻辑一致）
    valid_tmp_j_count = max(1, len([t for t in tmp_J if t is not None]))
    return total_loss / valid_tmp_j_count

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:, :, 2:, :] - img[:, :, :height - 2, :]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width - 2]).abs()
    return gradient_h, gradient_w


def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss


def R_loss(L1, R1, im1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1)
    loss1 = torch.nn.MSELoss()(L1 * R1, im1) + torch.nn.MSELoss()(R1, im1 / L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2


def gradient_1(input_tensor, direction):
    device = input_tensor.device
    input_channels = input_tensor.shape[1]  # 获取输入张量的通道数
    # 创建对应通道数的卷积核
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
    smooth_kernel_x = smooth_kernel_x.repeat(1, input_channels, 1, 1).to(device)

    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)


    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel.to(device),
                                  stride=1, padding=1))
    return grad_out
def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient_1(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)


def smooth_light(input_L, input_R):
    """
    多尺度引导式光照平滑损失（原自适应核心+多尺度扩展）
    核心：R引导L的自适应平滑 + 原尺度+0.5倍下采样尺度，覆盖所有大小纹理
    """

    def single_scale_smooth(L, R):
        # 单尺度：原自适应平滑逻辑（保留核心）
        # R转灰度单通道
        R_gray = 0.299 * R[:, 0, :, :] + 0.587 * R[:, 1, :, :] + 0.114 * R[:, 2, :, :]
        R_gray = torch.unsqueeze(R_gray, dim=1)  # [B,1,H,W]
        # 计算x/y方向自适应平滑损失
        loss_x = gradient_1(L, "x") * torch.exp(-10 * ave_gradient(R_gray, "x"))
        loss_y = gradient_1(L, "y") * torch.exp(-10 * ave_gradient(R_gray, "y"))
        return (loss_x + loss_y).mean()

    # 多尺度融合：原尺度 + 0.5倍下采样尺度
    # 原尺度损失
    loss_smooth = single_scale_smooth(input_L, input_R)
    # 下采样尺度损失（覆盖大尺度纹理）
    L_down = F.interpolate(input_L, scale_factor=0.5, mode='bilinear', align_corners=False)
    R_down = F.interpolate(input_R, scale_factor=0.5, mode='bilinear', align_corners=False)
    loss_smooth += single_scale_smooth(L_down, R_down)

    # 平均多尺度损失，避免尺度叠加导致损失值过大
    return loss_smooth / 2

# ========== 核心修改：loss_CTDM内部计算初始值 ==========
def loss_CTDM(I, R, L, mask):
    """
    1. 完全在损失函数内部计算init_R/init_L（基于输入特征I）
    2. 保留原有所有损失逻辑 + 初始值回归损失
    3. 无需外部传参init_R/init_L，调用更简洁
    """
    # ========== 1. 基础配置：mask适配 + 设备/尺寸获取 ==========
    device = I.device
    B, C, H, W = I.shape
    # 计算有效mask（屏蔽padding区域）
    if mask is not None:
        valid_mask = 1.0 - F.interpolate(mask.unsqueeze(1).float(), size=[H, W], mode='nearest').repeat(1, 3, 1, 1)
    else:
        valid_mask = torch.ones(B, 3, H, W, device=device)

    # ========== 2. 新增：在loss内部计算物理初始值（核心调整） ==========
    # 基于输入特征I计算初始反射率/光照（和CTDM_retinex逻辑一致）
    init_L = torch.max(I, dim=1, keepdim=True)[0].expand_as(I)  # 初始光照：通道维度最大值
    init_R = I / (init_L + 1e-6)                               # 初始反射率：I / (L + 极小值)

    # ========== 3. 原有核心损失：重构+反射率一致性+亮度平滑 ==========
    # 3.1 物理约束：R×L = I（仅有效区域）
    recon_pred = R * L * valid_mask
    recon_gt = I * valid_mask
    loss_recon = F.l1_loss(recon_pred, recon_gt)

    # 3.2 反射率一致性监督（仅bs=2时生效）
    loss_ref_consist = torch.tensor(0.0, device=device)
    bs = R.shape[0]
    if bs == 2:
        r1 = R[0:1] * valid_mask
        r2 = R[1:2] * valid_mask
        loss_ref_consist = compute_L_ref(r1, r2)  # 你的原有函数

    # 3.3 亮度平滑损失（抑噪声）
    loss_light = smooth_light(L*valid_mask, R*valid_mask)  # 你的原有函数

    # ========== 4. 初始值回归损失（基于内部计算的init_R/init_L） ==========
    init_loss_weight = 0.5  # 权重0.2，避免过度约束解空间
    # 仅有效区域约束：网络输出的R/L贴近物理初始值
    init_R_loss = F.l1_loss(R * valid_mask, init_R * valid_mask)

    loss_init = (init_R_loss ) * init_loss_weight

    # ========== 5. 损失权重融合 ==========
    #total_loss =0.5* (1.0 * loss_recon + 1.0 * loss_ref_consist + 0.1 * loss_light) + loss_init
    total_loss = 7*loss_recon +loss_light+ 3*loss_init
    return total_loss

# 需确保smooth_light函数已定义（多尺度引导式自适应平滑）
def loss_relight(low_R, low_L, low_L_tiled, high_L, paired, mask, stage=None):
    """
    最终优化版：
    1. 强化「基础光照保真」，限制Relight过度调整（核心）
    2. 阶段化损失配比，stage2保守拟合、stage3适度放开
    3. 增加数值约束，避免光照异常值引入伪影
    4. 平衡各损失占比，适配Relight残差微调策略
    5. 优化非配对场景约束，降低伪影风险
    """
    # ========== 1. 适配mask尺寸和通道 + 数值安全约束（新增） ==========
    B, C, H, W = low_R.shape
    valid_mask = 1.0 - F.interpolate(mask.unsqueeze(1).float(), size=[H, W], mode='nearest').repeat(1, 3, 1, 1)
    # 强制所有光照张量在合法范围（0.01~1.0），避免异常值导致损失震荡/伪影
    low_L = torch.clamp(low_L, 0.01, 1.0)
    low_L_tiled = torch.clamp(low_L_tiled, 0.01, 1.0)
    high_L = torch.clamp(high_L, 0.01, 1.0)

    # ========== 2. 基础光照保真损失（新增核心！限制Relight过度调整） ==========
    # 作用：强制Relight输出（low_L_tiled）贴合原始基础光照（low_L），仅做轻量调整
    # 权重最高，确保「基础光照为主，Relight微调为辅」，从损失层抑制残差
    loss_fidelity = F.l1_loss(low_L_tiled * valid_mask, low_L * valid_mask)

    # ========== 4. Relight核心损失（优化：阶段化幂次+权重，更保守） ==========
    # 动态幂次：stage2极保守（0.1→0.08），stage3适度放开（0.25→0.15）；配对/非配对差异化
    if stage is not None:
        if paired:
            power = 0.08 if stage == 2 else 0.12  # stage2仿真数据，幂次更低更保守
        else:
            power = 0.10 if stage == 2 else 0.15  # 非配对场景略放开，避免欠拟合
    else:
        power = 0.08 if paired else 0.10  # 无stage时，保留保守逻辑

    # 核心损失权重：全场景降权，适配残差微调；非配对场景大幅降低high_L拟合权重
    if paired:
        # 配对场景：high_L拟合（1.2→0.8）+ 幂次约束（1.2→0.9），双降权更保守
        loss_core = 0.8 * F.l1_loss(low_L_tiled * valid_mask, high_L * valid_mask) + \
                    0.9 * F.l1_loss(low_L_tiled * valid_mask, torch.pow(low_L, power) * valid_mask)
    else:
        # 非配对场景：high_L拟合（0.5→0.2），避免强制拟合引入伪影；幂次约束（1.0→0.7）
        loss_core = 0.7 * F.l1_loss(low_L_tiled * valid_mask, torch.pow(low_L, power) * valid_mask)

    # ========== 5. 亮度平滑损失（优化：轻量约束，仅抑制高频残差） ==========
    # 平滑损失仅针对Relight输出，权重保持极低（0.1→0.05），避免过度平滑丢失亮度渐变
    loss_smooth = 0.05*smooth_light(low_L_tiled * valid_mask, low_R * valid_mask)

    # ========== 6. 总损失（阶段化配比，核心：保真损失占比最高） ==========
    # stage2：极致保守，优先保基础光照，次要拟合high_L，轻微平滑
    # stage3：适度放开，保真仍为主，核心损失占比提升，平滑不变
    if stage == 2:
        total_loss = 1.5 * loss_fidelity + 0.8 * loss_core + loss_smooth
    elif stage in [3, 4]:  # 新增stage4适配，与解码器4-3-2-1层的stage对应
        total_loss = 1.2 * loss_fidelity + 1.0 * loss_core + loss_smooth
    else:
        total_loss = 1.3 * loss_fidelity + 0.9 * loss_core + loss_smooth

    return total_loss


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
def grad_denoise_loss(pred_img, sigma=1.0, kernel_size=3):
    """
    自监督梯度降噪损失：仅依赖模型输出，抑制无规律噪声，保留正常边缘
    原理：通过高斯模糊分离「真实边缘梯度」和「噪声梯度」，仅惩罚噪声部分
    输入：pred_img - [B,3,H,W]，模型输出的真实图像增强结果（支持float16/float32）
    输出：标量损失值（值越小，噪声越少）
    """
    device = pred_img.device
    # 1. 核心替换：用自定义gaussian_blur_2d替代kf.gaussian_blur2d（兼容所有PyTorch版本）
    blur_pred = gaussian_blur_2d(pred_img, kernel_size=kernel_size)  # 直接调用你的模糊函数

    # 2. 定义梯度核（修复：显式指定float32+动态匹配输入dtype，避免Long类型冲突）
    # 基础核显式设为float32，避免默认整型
    kernel_x = torch.tensor([[-1.0, 1.0]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)
    # 重塑并扩展为3通道，适配RGB图像
    kernel_x = kernel_x.view(1, 1, 1, 2).repeat(3, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 2, 1).repeat(3, 1, 1, 1)
    # 动态转换为输入图像的dtype（兼容float16/半精度训练）
    kernel_x = kernel_x.to(dtype=pred_img.dtype)
    kernel_y = kernel_y.to(dtype=pred_img.dtype)

    # 3. 计算梯度（修复：添加groups=3，实现3通道分组卷积，解决通道不匹配）
    pred_gx = torch.abs(F.conv2d(pred_img, kernel_x, padding='same', groups=3))
    pred_gy = torch.abs(F.conv2d(pred_img, kernel_y, padding='same', groups=3))
    # 4. 计算模糊后的梯度（仅含真实边缘，噪声被平滑）
    blur_gx = torch.abs(F.conv2d(blur_pred, kernel_x, padding='same', groups=3))
    blur_gy = torch.abs(F.conv2d(blur_pred, kernel_y, padding='same', groups=3))

    # 5. 仅惩罚噪声梯度（原始梯度 - 边缘梯度，小于0则置0，不惩罚正常边缘）
    noise_gx = torch.clamp(pred_gx - blur_gx, min=0.0)
    noise_gy = torch.clamp(pred_gy - blur_gy, min=0.0)

    # 6. 平滑L1损失（抗异常值，避免过度降噪导致图像过平滑）
    loss = F.smooth_l1_loss(noise_gx, torch.zeros_like(noise_gx)) + \
           F.smooth_l1_loss(noise_gy, torch.zeros_like(noise_gy))
    return loss


def gradient_loss(pred, target):
    """
    高频梯度损失：计算水平+垂直梯度，惩罚边缘模糊，专门提升细节PSNR
    输入：pred/target - [B,3,H,W]，支持float16/float32，取值[0,1]
    输出：梯度损失值（标量），与输入同精度
    """
    device = pred.device
    # 核心修改1：定义核时显式用float32，避免默认Long整型
    # 水平梯度核（1x2）、垂直梯度核（2x1）
    kernel_x = torch.tensor([[-1.0, 1.0]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)

    # 重塑核形状并扩展为3通道，适配RGB图像
    kernel_x = kernel_x.view(1, 1, 1, 2).repeat(3, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 2, 1).repeat(3, 1, 1, 1)

    # 核心修改2：将核转换为与输入pred完全一致的数值类型（兼容float16/float32）
    kernel_x = kernel_x.to(dtype=pred.dtype)
    kernel_y = kernel_y.to(dtype=pred.dtype)

    # 计算梯度（groups=3保证3通道分组卷积，padding='same'保持尺寸不变）
    pred_gx = torch.abs(F.conv2d(pred, kernel_x, padding='same', groups=3))
    pred_gy = torch.abs(F.conv2d(pred, kernel_y, padding='same', groups=3))
    target_gx = torch.abs(F.conv2d(target, kernel_x, padding='same', groups=3))
    target_gy = torch.abs(F.conv2d(target, kernel_y, padding='same', groups=3))

    # L1梯度损失（鲁棒性强，不放大噪声，与输入同精度计算）
    loss_gx = F.l1_loss(pred_gx, target_gx)
    loss_gy = F.l1_loss(pred_gy, target_gy)
    return loss_gx + loss_gy
# -------------------------- 简化版RGB转LAB（保留核心，移除冗余归一化） --------------------------
def rgb_to_lab(img_tensor):
    """
    简化版RGB转LAB：仅做维度适配和范围转换，保留原始LAB数值（避免归一化损失精度）
    输入：img_tensor - [B,3,H,W] 或 [3,H,W]，取值[0,1]
    输出：lab_tensor - 同维度，L(0-100)、a(-128~127)、b(-128~127)
    """
    is_batch = len(img_tensor.shape) == 4
    if not is_batch:
        img_tensor = img_tensor.unsqueeze(0)

    # RGB范围转换：[0,1] → [-1,1]（适配kornia）
    img_11 = (img_tensor * 2) - 1
    lab = kc.rgb_to_lab(img_11)

    if not is_batch:
        lab = lab.squeeze(0)
    return lab


# -------------------------- 聚焦核心的颜色一致性损失（移除冗余约束） --------------------------
def color_consistency_loss(pred_img, refer_img):
    """
    简化版颜色损失：聚焦L通道（亮度）+ ab通道（色度）的核心匹配，移除协方差等复杂约束
    优先级：亮度 > 色度，使用平滑L1替代L1（抗异常值）
    """
    # 1. 转换到LAB空间（保留原始数值）
    pred_lab = rgb_to_lab(pred_img)
    refer_lab = rgb_to_lab(refer_img)

    # 2. 分离L/a/b通道（亮度是视觉核心，加权优先）
    pred_L, pred_a, pred_b = pred_lab[:, 0:1], pred_lab[:, 1:2], pred_lab[:, 2:3]
    refer_L, refer_a, refer_b = refer_lab[:, 0:1], refer_lab[:, 1:2], refer_lab[:, 2:3]

    # 3. 平滑L1损失（替代L1，减少异常值影响）
    L_loss = F.smooth_l1_loss(pred_L, refer_L)  # 亮度损失（权重最高）
    ab_loss = F.smooth_l1_loss(torch.cat([pred_a, pred_b], dim=1),
                               torch.cat([refer_a, refer_b], dim=1))  # 色度损失

    # 4. 加权求和（亮度重要性是色度的2倍）
    total_loss = 2 * L_loss + ab_loss
    return total_loss


# -------------------------- 简化版通道损失（聚焦高频+空间注意力） --------------------------
def channel_loss(pred_img, refer_img):
    """
    简化版通道损失：移除复杂权重，聚焦边缘/高频区域的颜色匹配，提升梯度效率
    """
    # 1. 基础通道差异（保留原始数值）
    channel_diff = torch.abs(pred_img - refer_img)  # [B,3,H,W]

    # 2. 高频区域提取（替代Sobel，使用高斯模糊差分，更稳定）
    def gaussian_blur(x, kernel_size=5, sigma=2):
        # 修正：使用kornia.filters的gaussian_blur2d
        return kf.gaussian_blur2d(x, (kernel_size, kernel_size), (sigma, sigma))

    # 高频 = 原图 - 模糊图（保留边缘/细节）
    pred_high = pred_img - gaussian_blur(pred_img)
    refer_high = refer_img - gaussian_blur(refer_img)
    high_weight = torch.abs(pred_high) + torch.abs(refer_high)  # 高频区域权重
    high_weight = F.softmax(high_weight.flatten(2), dim=2).reshape(high_weight.shape)  # 归一化权重

    # 3. 加权通道损失（仅聚焦高频区域）
    weighted_diff = channel_diff * high_weight
    high_loss = torch.mean(weighted_diff)

    # 4. 全局基础损失（低权重，保证整体颜色）
    base_loss = torch.mean(channel_diff)

    # 5. 总损失（高频优先，权重3倍）
    total_loss = 3 * high_loss + base_loss
    return total_loss


# -------------------------- 组合损失（最终训练用） --------------------------
def total_color_loss(pred_img, refer_img):
    """
    组合颜色损失：简化+聚焦，保证梯度可优化
    """
    # 颜色一致性损失（LAB空间） + 通道损失（RGB高频）
    loss1 = color_consistency_loss(pred_img, refer_img)
    loss2 = channel_loss(pred_img, refer_img)
    # 平衡权重（根据任务调整，初始1:1）
    total_loss = loss1 + loss2
    return total_loss
from model.rank_GPU import tensor_rank1_ddp


def physical_loss(pred_img, x0):
    """
    物理模型伪监督损失（低干扰版）
    :param pred_img: 模型输出图像 [B, C, H, W]
    :param x0: 输入真实水下图像的nested_tensor对象（含tensors和mask）
    :return: 物理约束L1损失（已做mask过滤，数值稳定）
    """
    device = x0.tensors.device
    # 1. 从输入图像自身估计物理参数（J:光照, t:透射率, A:背景光）
    J_init, t_hat_init, A_init = tensor_rank1_ddp(x0.tensors, x0.mask, device)

    # 2. 强制设备一致性+维度适配（避免广播/设备错误）
    if t_hat_init.tensors.device != x0.tensors.device:
        t_hat_init = t_hat_init.to(x0.tensors.device)
        A_init = A_init.to(x0.tensors.device)
        J_init = J_init.to(x0.tensors.device)
    if hasattr(J_init, 'tensors'):
        J_init = J_init.tensors
        t_hat_init = t_hat_init.tensors
        A_init = A_init.tensors

    # 透射率t强制裁剪在[0.05, 0.95]，避免过暗/过曝，保证物理合理性
    t_hat_init = torch.clamp(t_hat_init, 0.05, 0.95)
    # 背景光A适配pred_img通道数（防止通道维度不匹配）
    if A_init.dim() == 1:
        A_init = A_init.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]

    # 3. 物理模型重建输入图像：phyI = pred*T + A*(1-T)（修正原公式符号，符合水下成像物理模型）
    phyI = pred_img * (1-t_hat_init) + A_init *  t_hat_init
    # 重建结果裁剪，避免数值溢出导致损失突变
    phyI = torch.clamp(phyI, 0.0, 1.0)

    l1_loss = nn.functional.l1_loss(phyI, x0.tensors)




    return l1_loss


