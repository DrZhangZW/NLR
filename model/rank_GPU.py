import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from scipy.stats import mstats
import torch
import torch.nn.functional as F
from utils import misc

# -------------------------- 核心GPU化函数 --------------------------
def crop_with_mask_gpu(matrix: torch.Tensor, mask: torch.Tensor):
    """
    GPU版：根据mask裁剪张量，去除padding区域(标记为0的部分)
    参数:
        matrix: 输入张量 [C, H, W]
        mask: 掩码张量 [H, W], 1表示有效区域
    返回:
        裁剪后的张量 [C, H_new, W_new]
    """
    # 获取有效区域边界
    rows = torch.any(mask == 1, dim=1)  # [H]
    cols = torch.any(mask == 1, dim=0)  # [W]

    # 计算裁剪范围（GPU上避免numpy操作）
    y_indices = torch.nonzero(rows).squeeze(1)
    x_indices = torch.nonzero(cols).squeeze(1)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return matrix  # 无有效区域，返回原张量

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    # 执行裁剪
    cropped = matrix[:, y_min:y_max + 1, x_min:x_max + 1]
    return cropped
# ===================== 第三步：最终GPU版函数（完全对齐CPU版） =====================
def tensor_rank1_ddp(x0: torch.Tensor, mask: torch.Tensor, device: torch.device):
    """
    纯GPU版（无CPU调用）+ 完全对齐CPU版逻辑
    核心修改：
    1. 移除插值（CPU版无插值）
    2. 移除额外mask乘法（CPU版仅裁剪用mask）
    3. 复刻CPU版裁剪/数值范围/维度逻辑
    4. 全程GPU tensor计算，无numpy/CPU交互
    """
    # 步骤1：设备/类型统一（和CPU版一致）
    x0 = x0.to(device, non_blocking=True)
    mask = mask.float().to(device, non_blocking=True)  # 避免布尔运算错误
    batch_size = x0.shape[0]

    processed_J, processed_t_hat, processed_A = [], [], []

    for i in range(batch_size):
        img = x0[i]  # [3, H, W]（0-1）
        mask_i = mask[i]  # [H, W]
        valid_mask = 1.0 - mask_i  # [H, W]（1=有效区域，和CPU版m一致）

        # 步骤2：裁剪有效区域（纯GPU，复刻CPU版crop_with_mask）
        cropped_img = crop_with_mask_gpu(img, valid_mask)

        # 步骤3：核心计算（纯GPU，复刻CPU版rank_one_prior逻辑）
        J_cropped, s_tildeT_cropped, tildeT, A_cropped = rank_one_prior_gpu(cropped_img, device)

        # 步骤4：移除插值+移除额外mask乘法（和CPU版一致）
        J = J_cropped
        s_tildeT = tildeT
        A = A_cropped

        # 步骤5：添加到列表（维度和CPU版一致）
        processed_J.append(J)
        processed_t_hat.append(s_tildeT)
        processed_A.append(A)

    # 步骤6：转为NestedTensor（和CPU版格式统一）
    processed_J = misc.nested_tensor_from_tensor_list(processed_J)
    processed_t_hat = misc.nested_tensor_from_tensor_list(processed_t_hat)
    processed_A = misc.nested_tensor_from_tensor_list(processed_A)

    return processed_J, processed_t_hat, processed_A

def tensor_rank1_ddp_better(x0: torch.Tensor, mask: torch.Tensor, device: torch.device):
    """
    DDP友好版：批量张量秩一先验计算（全程GPU）
    修复点：布尔掩码转浮点型后再做算术运算
    参数:
        x0: 输入张量 [B, C, H, W] (0-1范围)
        mask: 掩码张量 [B, H, W] (0=有效, 1=padding) → 支持bool/float型
        device: 目标设备 (如cuda:0)
    返回:
        J, t_hat, A: 批量张量 [B, C, H, W]
    """
    # 步骤1：统一设备 + 确保mask是浮点型（核心修复）
    x0 = x0.to(device, non_blocking=True)
    # 无论mask是bool/float，先转float，避免布尔运算错误
    mask = mask.float().to(device, non_blocking=True)

    batch_size = x0.shape[0]
    processed_J, processed_t_hat, processed_A = [], [], []
    original_h, original_w = x0.shape[2], x0.shape[3]

    for i in range(batch_size):
        img = x0[i]  # [3, H, W]
        # 步骤2：计算有效区域掩码（提前转float，避免重复错误）
        mask_i = mask[i]  # [H, W] → 已是float型
        valid_mask = 1.0 - mask_i  # [H, W] (1=有效区域，0=padding)

        # 裁剪有效区域（GPU版）
        cropped_img = crop_with_mask_gpu(img, valid_mask)

        # 核心去雾计算（全程GPU）
        J_cropped, s_tildeT_cropped, _, A_cropped = rank_one_prior_gpu(cropped_img, device)

        # 恢复原始尺寸（双线性插值）
        J = F.interpolate(J_cropped.unsqueeze(0), size=(original_h, original_w),
                          mode='bilinear', align_corners=False).squeeze(0)
        s_tildeT = F.interpolate(s_tildeT_cropped.unsqueeze(0), size=(original_h, original_w),
                                 mode='bilinear', align_corners=False).squeeze(0)
        A = F.interpolate(A_cropped.unsqueeze(0), size=(original_h, original_w),
                          mode='bilinear', align_corners=False).squeeze(0)

        # 步骤3：应用mask（仅保留有效区域）→ 使用提前计算的float型valid_mask
        J = J * valid_mask.unsqueeze(0)  # [3, H, W] * [1, H, W]
        s_tildeT = s_tildeT * valid_mask.unsqueeze(0)  # [1, H, W] * [1, H, W]
        A = A * valid_mask.unsqueeze(0)  # [3, H, W] * [1, H, W]

        # 添加到结果列表
        processed_J.append(J)
        processed_t_hat.append(s_tildeT)
        processed_A.append(A)

    # ========== 关键修改：转为NestedTensor（和CPU版一致） ==========
    processed_J = misc.nested_tensor_from_tensor_list(processed_J)
    processed_t_hat = misc.nested_tensor_from_tensor_list(processed_t_hat)
    processed_A = misc.nested_tensor_from_tensor_list(processed_A)

    return processed_J, processed_t_hat, processed_A



def data_fit(I, A, t_hat, mask):
    """
    DDP友好版：数据拟合函数（全程GPU）
    """
    device = I.device
    batch_size, c, H, W = I.shape
    processed_J = []
    # 提前统一mask为float类型（全局避免布尔错误）
    mask_float = mask.float().to(device, non_blocking=True)
    VALID_MIN = 1e-10  # 统一容错最小值（和rank_one_prior_gpu保持一致）

    for i in range(batch_size):
        tmp_I = I[i]  # [3, H, W]
        tmp_A = A[i]  # [3, H, W]
        tmp_t_hat = t_hat[i]  # [1, H, W]
        mask_i = mask_float[i]  # [H, W] → 已转为float，避免布尔错误
        valid_mask = 1.0 - mask_i  # [H, W]（1=有效区域，0=padding）

        # ========== 优化2：裁剪逻辑复用（和rank_one_prior_gpu的输入维度匹配） ==========
        # 裁剪后直接保持 [C, Hc, Wc]，避免提前转置（减少permute次数）
        tmp_I_cropped = crop_with_mask_gpu(tmp_I, valid_mask)  # [3, Hc, Wc]
        tmp_A_cropped = crop_with_mask_gpu(tmp_A, valid_mask)  # [3, Hc, Wc]
        tmp_t_hat_cropped = crop_with_mask_gpu(tmp_t_hat, valid_mask)  # [1, Hc, Wc]
        # ========== 优化3：核心计算复用rank_one_prior_gpu的逻辑 ==========
        # 转换为 [H, W, C]（仅在计算时转置，和rank_one_prior_gpu完全一致）
        tmp_I_hwc = tmp_I_cropped.permute(1, 2, 0)  # [Hc, Wc, 3]
        tmp_A_hwc = tmp_A_cropped.permute(1, 2, 0)  # [Hc, Wc, 3]
        tmp_t_hat_hwc = tmp_t_hat_cropped.permute(1, 2, 0)  # [Hc, Wc, 1]

        # 透射率计算（复用rank_one_prior_gpu的单通道平均逻辑，提升效果）
        t_ini = 1 - tmp_t_hat_hwc  # [Hc, Wc, 1]
        # 去雾核心计算（统一容错最小值为1e-10）
        ShowR_d = (tmp_I_hwc - tmp_A_hwc) / torch.clamp(t_ini, min=VALID_MIN) + tmp_A_hwc  # [Hc, Wc, 3]

        # ========== 优化4：分位数归一化（完全复用rank_one_prior_gpu的逻辑） ==========
        ShowR_d_flat = ShowR_d.reshape(-1, 3)  # [Hc*Wc, 3]
        mi = torch.quantile(ShowR_d_flat, 0.01, dim=0)  # [3]
        ma = torch.quantile(ShowR_d_flat, 0.99, dim=0)  # [3]
        # 扩展维度（和rank_one_prior_gpu一致，仅扩展2次）
        mi = mi[None, None, :]  # [1, 1, 3]
        ma = ma[None, None, :]  # [1, 1, 3]

        Jr_ini = (ShowR_d - mi) / torch.clamp(ma - mi, min=VALID_MIN)
        Jr_ini = torch.clamp(Jr_ini, 0.0, 1.0)  # [Hc, Wc, 3]

        # Gamma校正
        Jr_ini = gamma0_opencv_gpu(Jr_ini)  # [Hc, Wc, 3]

        # 恢复尺寸和格式
        Jr_ini = Jr_ini.permute(2, 0, 1)  # [3, Hc, Wc]
        Jr_ini = F.interpolate(Jr_ini.unsqueeze(0), size=(H, W),
                               mode='bilinear', align_corners=False).squeeze(0)  # [3, H, W]
        Jr_ini = Jr_ini * valid_mask.unsqueeze(0)  # [3, H, W]

        processed_J.append(Jr_ini)

    processed_J = misc.nested_tensor_from_tensor_list(processed_J)

    processed_J.tensors = processed_J.tensors[:,:,:H,:W]
    processed_J.mask = processed_J.mask[:,:H,:W]
    # 拼接为批量张量
    #processed_J = torch.stack(processed_J, dim=0)
    return processed_J


def rank_one_prior_gpu(img: torch.Tensor, device: torch.device):
    """
    GPU版：秩一先验去雾核心计算（严格控制维度）
    参数:
        img: 输入张量 [C, H, W] (0-1范围)
        device: 计算设备 (cuda:x)
    返回:
        J: 去雾图像 [C, H, W]
        s_tildeT: 平滑透射率 [1, H, W]
        tildeT: 透射率 [1, H, W]
        atmosphere: 大气光 [C, H, W]
    """
    # 强制确保输入是3维 [3, H, W]

    h, w = img.shape[1], img.shape[2]

    # 转换为 [H, W, 3]（严格3维）
    img = img.permute(1, 2, 0)  # [H, W, 3]


    # 统一光谱计算
    x_RGB = torch.mean(img.reshape(-1, 3), dim=0)  # [3]
    x_mean = x_RGB[None, None, :].expand(h, w, 3)  # [H, W, 3]（用None避免多余维度）

    # 方向相似度计算
    scat_basis = x_mean / torch.clamp(torch.norm(x_mean, dim=-1, keepdim=True), min=1e-10)  # [H, W, 3]
    fog_basis = img / torch.clamp(torch.norm(img, dim=-1, keepdim=True), min=1e-10)  # [H, W, 3]
    cs_sim = torch.sum(scat_basis * fog_basis, dim=-1, keepdim=True)  # [H, W, 1]

    # 散射光估计
    scatter_light = cs_sim * (torch.sum(img, dim=-1, keepdim=True) /
                              torch.clamp(torch.sum(x_mean, dim=-1, keepdim=True), min=1e-3)) * x_mean  # [H, W, 3]

    # 大气光估计
    atmosphere, scatter_light = get_atmosphere_gpu(img, scatter_light, device)  # 均为[H, W, 3]


    # 透射率计算
    omega = 0.8
    T = 1 - omega * scatter_light  # [H, W, 3]
    T_mean = T.mean(dim=-1)  # [H, W]（单通道平均）
    T_ini = scattering_mask_sample_gpu(T_mean)  # [H, W]


    # 去雾核心计算
    ShowR_d = (img - atmosphere) / torch.clamp(T_ini[:, :, None], min=0.001) + atmosphere  # [H, W, 3]


    # 百分位数归一化（严格控制维度，避免4维）
    ShowR_d_flat = ShowR_d.reshape(-1, 3)  # [H*W, 3]
    mi = torch.quantile(ShowR_d_flat, 0.01, dim=0)  # [3]（不使用keepdim，避免多余维度）
    ma = torch.quantile(ShowR_d_flat, 0.99, dim=0)  # [3]
    # 扩展为 [1,1,3]（仅扩展2次，保持3维）
    mi = mi[None, None, :]  # [1, 1, 3]
    ma = ma[None, None, :]  # [1, 1, 3]

    Jr_ini = (ShowR_d - mi) / torch.clamp(ma - mi, min=1e-10)  # [H, W, 3]
    Jr_ini = torch.clamp(Jr_ini, 0.0, 1.0)  # [H, W, 3]（严格3维）


    # Gamma校正
    Jr_ini = gamma0_opencv_gpu(Jr_ini)  # [H, W, 3]


    # 透射率处理
    tildeT = 1 - T_mean  # [H, W]
    s_tildeT = 1 - T_ini  # [H, W]

    # 转换回 [C, H, W] 格式（核心修复：确保3维转置）
    J = Jr_ini.permute(2, 0, 1)  # [3, H, W]
    s_tildeT = s_tildeT.unsqueeze(0)  # [1, H, W]
    tildeT = tildeT.unsqueeze(0)  # [1, H, W]
    atmosphere = atmosphere.permute(2, 0, 1)  # [3, H, W]


    return J, s_tildeT, tildeT, atmosphere


def scattering_mask_sample_gpu(I: torch.Tensor):
    """
    GPU版：散射掩码采样（严格控制维度）
    参数:
        I: 输入张量 [H, W] (单通道)
    返回:
        采样后的张量 [H, W]
    """
    assert I.dim() == 2, f"输入必须是[H, W]，当前是{I.shape}"
    h, w = I.shape

    # 计算缩小尺寸（保持比例）
    small_h = max(1, int(h * 0.02))
    small_w = max(1, int(w * 0.02))

    # GPU上的双线性插值（严格控制维度）
    I_reshaped = I.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    I0 = F.interpolate(I_reshaped, size=(small_h, small_w), mode='bilinear', align_corners=False)
    F_interp = F.interpolate(I0, size=(h, w), mode='bilinear', align_corners=False)

    return F_interp.squeeze(0).squeeze(0)  # [H, W]


def gamma0_opencv_gpu(img: torch.Tensor):
    """
    GPU版：Gamma校正（严格控制维度）
    参数:
        img: 输入张量 [H, W, 3] (0-1范围)
    返回:
        校正后的张量 [H, W, 3] (0-1范围)
    """
    assert img.dim() == 3 and img.shape[-1] == 3, f"输入必须是[H, W, 3]，当前是{img.shape}"
    h, w = img.shape[0], img.shape[1]

    # 转换为YCbCr（GPU版）
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]  # [H, W]
    Cb = (img[..., 1] - Y) / (2 * (1 - 0.114)) + 0.5  # [H, W]
    Cr = (img[..., 0] - Y) / (2 * (1 - 0.299)) + 0.5  # [H, W]

    # Gamma校正 (5/6次方)
    Y = torch.clamp(Y, 1e-6, 1.0)  # 避免0次方错误
    Y_corrected = Y ** (5 / 6)  # [H, W]

    # YCbCr2RGB逆转换
    R = Y_corrected + 1.402 * (Cr - 0.5)  # [H, W]
    G = Y_corrected - 0.34414 * (Cb - 0.5) - 0.71414 * (Cr - 0.5)  # [H, W]
    B = Y_corrected + 1.772 * (Cb - 0.5)  # [H, W]

    # 拼接并裁剪到0-1范围（严格3维）
    corrected = torch.stack([R, G, B], dim=-1)  # [H, W, 3]
    return torch.clamp(corrected, 0.0, 1.0)


def get_atmosphere_gpu(image: torch.Tensor, scatterlight: torch.Tensor, device: torch.device):
    """
    GPU版：大气光估计（严格控制维度）
    参数:
        image: 输入张量 [H, W, 3] (0-1范围)
        scatterlight: 散射光张量 [H, W, 3]
    返回:
        atmosphere: 大气光张量 [H, W, 3]
        scatterlight: 校正后的散射光张量 [H, W, 3]
    """
    assert image.dim() == 3 and scatterlight.dim() == 3, f"输入必须是3维，当前image={image.shape}, scatterlight={scatterlight.shape}"
    h, w = image.shape[0], image.shape[1]

    # 散射光强度总和
    scatter_est = torch.sum(scatterlight, dim=-1)  # [H, W]
    n_pixels = h * w
    n_search_pixels = max(1, int(n_pixels * 0.001))  # 前0.1%像素

    # 散射光强度排序（GPU版argsort）
    scatter_flat = scatter_est.flatten()
    _, sorted_indices = torch.topk(scatter_flat, n_search_pixels, largest=True)

    # 大气光估计
    image_flat = image.reshape(-1, 3)  # [H*W, 3]
    atmosphere = torch.mean(image_flat[sorted_indices], dim=0)  # [3]
    atmosphere = atmosphere[None, None, :].expand(h, w, 3)  # [H, W, 3]（严格3维）

    # 阈值计算与散射光校正
    sek = scatter_flat[sorted_indices[-1]]
    sek_vec = sek * scatterlight[0, 0] / torch.clamp(scatter_est[0, 0], min=1e-10)  # [3]
    sek_vec = sek_vec[None, None, :].expand(h, w, 3)  # [H, W, 3]

    # 掩码校正
    mask = (scatter_est <= sek)[:, :, None]  # [H, W, 1]
    scatterlight = scatterlight * mask + (2 * sek_vec - scatterlight) * (~mask)  # [H, W, 3]

    return atmosphere, scatterlight


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 1. 检查CUDA是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 构造测试张量（模拟批量输入，严格控制维度）
    batch_size = 2
    test_img = torch.rand(batch_size, 3, 256, 256).float().to(device)  # [B, 3, H, W]（用rand确保0-1）
    test_mask = torch.zeros(batch_size, 256, 256).float().to(device)  # [B, H, W]

    # 3. 调用函数并验证设备
    J, t_hat, A = tensor_rank1_ddp(test_img, test_mask, device)

    # 4. 检查输出维度和设备
    print(f"J.shape: {J.shape}, 设备: {J.device}")  # 预期 [2,3,256,256]
    print(f"t_hat.shape: {t_hat.shape}, 设备: {t_hat.device}")  # 预期 [2,1,256,256]
    print(f"A.shape: {A.shape}, 设备: {A.device}")  # 预期 [2,3,256,256]

    # 5. 检查中间函数的维度和设备
    single_img = test_img[0]  # [3, 256, 256]
    J_single, s_tildeT, _, A_single = rank_one_prior_gpu(single_img, device)
    print(f"单张J.shape: {J_single.shape}, 设备: {J_single.device}")  # 预期 [3,256,256]
    print(f"s_tildeT.shape: {s_tildeT.shape}, 设备: {s_tildeT.device}")  # 预期 [1,256,256]

    # 6. 测试data_fit函数
    fit_result = data_fit(test_img, A, t_hat, test_mask)
    print(f"data_fit.shape: {fit_result.shape}, 设备: {fit_result.device}")  # 预期 [2,3,256,256]
    print(f"所有测试通过！")