import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from scipy.stats import mstats
import torch
from utils import misc
from utils.common_utils import visualize_tensor


def crop_with_mask(matrix, mask):
    """
    根据mask裁剪矩阵，去除padding区域(标记为0的部分)
    参数:
        matrix: 输入矩阵 (h×w×3)
        mask: 掩码矩阵 (h×w), 1表示有效区域
    返回:
        裁剪后的新矩阵 (h_new×w_new×3)
    """
    # 获取有效区域边界（值为1的区域）
    rows = np.any(mask == 1, axis=1)  # 每行是否有有效像素
    cols = np.any(mask == 1, axis=0)  # 每列是否有有效像素

    # 计算裁剪范围
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 执行裁剪（确保在矩阵范围内）
    cropped = matrix[y_min:y_max + 1, x_min:x_max + 1, :]
    return cropped


def tensor_rank1(x0, mask):
    """
            遍历处理批量张量图像：
            x0: input tensor [B, C, H, W] in range [0, 1]
            mask: 0=true, 1=padding
            Returns: J, s_tildeT, tildeT, atmosphere all in range [0, 1]
            """
    # Convert to [B, H, W, C] for processing
    x0 = x0.to('cpu')
    mask = mask.to('cpu')

    batch_size = x0.shape[0]
    processed_J, processed_tildeT, processed_A = [], [], []
    for i in range(0, batch_size):
        img = x0[i]
        m = (1 - mask[i].float()).numpy()
        # 转置为h x w x c并转为numpy
        img_np = img.permute(1, 2, 0).numpy()  # h x w x c
        # 获取有效像素坐标
        valid_img = crop_with_mask(img_np, m)
        J, s_tildeT, tildeT, atmosphere = rank_one_prior(valid_img)
        # plt.imshow(s_tildeT)
        # plt.show(block=True)
        processed_J.append(torch.from_numpy(J).float().permute(2, 0, 1) / 255.0)  # HxWxC -> CxHxW
        processed_tildeT.append(torch.from_numpy(s_tildeT).permute(2, 0, 1))  # HxW -> 1xHxW
        processed_A.append(torch.from_numpy(atmosphere).permute(2, 0, 1))  # HxWxC -> CxHxW
    processed_J = misc.nested_tensor_from_tensor_list(processed_J)
    processed_tildeT = misc.nested_tensor_from_tensor_list(processed_tildeT)
    processed_A = misc.nested_tensor_from_tensor_list(processed_A)
    # visualize_tensor(processed_J.tensors)
    # visualize_tensor(processed_tildeT.tensors)
    # visualize_tensor(processed_A.tensors)
    return  processed_J,processed_tildeT, processed_A

def data_fit(I, A, t_hat, mask):
    """
    I,A,t_hat: bsxcxhxw
    """

    I = I.to('cpu')
    A = A.to('cpu')
    t_hat = t_hat.to('cpu')
    mask = mask.to('cpu')

    batch_size,c,H,W = I.shape
    processed_J = []
    for i in range(0, batch_size):
        tmp_I, tmp_A, tmp_t_hat = I[i], A[i], t_hat[i]
        m = (1 - mask[i].float()).numpy()
        # 转置为h x w x c并转为numpy
        tmp_I = tmp_I.permute(1, 2, 0).numpy()  # h x w x c
        tmp_A = tmp_A.permute(1, 2, 0).numpy()  # h x w x c
        tmp_t_hat = tmp_t_hat.permute(1, 2, 0).detach().numpy()  # h x w x c
        # 获取有效像素坐标
        tmp_I = crop_with_mask(tmp_I, m)
        tmp_A = crop_with_mask(tmp_A, m)
        tmp_t_hat = crop_with_mask(tmp_t_hat, m)
        # -
        t_ini = 1 - tmp_t_hat
        ShowR_d = (tmp_I - tmp_A) / np.maximum(t_ini, 0.001) + tmp_A
        # 注意：厚雾场景不建议进行亮度调整
        # 计算百分位数
        mi = prctile20192(ShowR_d, 1, [0, 1])  # 1%分位数
        ma = prctile20192(ShowR_d, 99, [0, 1])  # 99%分位数
        mi = np.reshape(mi, (1, 1, -1))
        ma = np.reshape(ma, (1, 1, -1))
        # 通道归一化
        Jr_ini = np.zeros_like(ShowR_d)
        for c in range(3):
            Jr_ini[..., c] = (ShowR_d[..., c] - mi[0, 0, c]) / (ma[0, 0, c] - mi[0, 0, c])

        # Gamma校正
        Jr_ini = gamma0_opencv(Jr_ini * 255)

        # plt.imshow(s_tildeT)
        # plt.show(block=True)
        processed_J.append(torch.from_numpy(Jr_ini).float().permute(2, 0, 1) / 255.0)  # HxWxC -> CxHxW
    processed_J = misc.nested_tensor_from_tensor_list(processed_J)

    processed_J.tensors = processed_J.tensors[:,:,:H,:W]
    processed_J.mask = processed_J.mask[:,:H,:W]
    return processed_J




def rank_one_prior(img, omega=0.8):
    # img:  Image.open(img_path)
    # return:
    # Jr_ini=recoverd images
    # s_tildeT: smoothed \tilde{T} which we use for final computation
    # tildeT: \tilde{t} in equation (1)
    h, w = img.shape[:2]
    # ================== 向量化处理=====================================
    img_vec = img.reshape(-1, 3)
    # 统一光谱计算（原MATLAB代码中x_RGB被简化为均值）
    x_RGB = np.mean(img_vec, 0)  # unified spectrum
    # ====================direction difference==============================
    # 扩展为图像尺寸
    x_mean = np.tile(x_RGB[np.newaxis, np.newaxis, :], (h, w, 1))
    scat_basis = x_mean / np.maximum(np.sqrt(np.sum(x_mean ** 2, axis=2, keepdims=True)), 0.001)  # normalization
    fog_basis = img / np.maximum(np.sqrt(np.sum(img ** 2, axis=2, keepdims=True)), 0.001)  # normalization
    cs_sim = np.sum(scat_basis * fog_basis, axis=2)  # cos similarity
    cs_sim = np.tile(cs_sim[:, :, np.newaxis], (1, 1, 3))
    # ===================scattering_light_estimation===================

    scattering_light = cs_sim * (np.sum(img, axis=2) /
                                 np.maximum(np.sum(x_mean, axis=2), 1e-3))[..., np.newaxis] * x_mean

    intial_img = img
    # ====================get atmosphere==========================
    [atmosphere, scattering_light] = get_atmosphere(intial_img, scattering_light)
    T = 1 - omega * scattering_light
    T_ini = scattering_mask_sample(T)  # 需实现该函数
    # 去雾核心计算
    ShowR_d = (intial_img - atmosphere) / np.maximum(T_ini, 0.001) + atmosphere

    # 注意：厚雾场景不建议进行亮度调整
    # 计算百分位数
    mi = prctile20192(ShowR_d, 1, [0, 1])  # 1%分位数
    ma = prctile20192(ShowR_d, 99, [0, 1])  # 99%分位数
    mi = np.reshape(mi, (1, 1, -1))
    ma = np.reshape(ma, (1, 1, -1))
    # 通道归一化
    Jr_ini = np.zeros_like(ShowR_d)
    for c in range(3):
        Jr_ini[..., c] = (ShowR_d[..., c] - mi[0, 0, c]) / (ma[0, 0, c] - mi[0, 0, c])

    # Gamma校正
    Jr_ini = gamma0_opencv(Jr_ini * 255)

    # 透射率计算
    tildeT = 1 - T
    s_tildeT = 1 - T_ini
    return Jr_ini, s_tildeT, tildeT, atmosphere





def scattering_mask_sample(I):
    # 方案1：使用scikit-image（最接近MATLAB行为）
    small_size = (max(1, int(I.shape[0] * 0.02)), max(1, int(I.shape[1] * 0.02)))
    I0 = resize(I, small_size, order=3, mode='reflect', anti_aliasing=True)
    F = resize(I0, I.shape[:2], order=3, mode='reflect', anti_aliasing=False)
    return F


def gamma0_opencv(img):
    # uint8[0,255]
    # 使用OpenCV的色彩空间转换
    if img.dtype != np.uint8:
        img = np.nan_to_num(img)  # 替换NaN为0，inf为有限值
        img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.ascontiguousarray(img, dtype=np.uint8)
    lut = np.clip(((np.arange(256) + 0.5) / 256) ** (5 / 6) * 255 + 0.5, 0, 255).astype(np.uint8)

    # OpenCV的RGB2YCrCb转换
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycbcr[..., 0] = lut[ycbcr[..., 0]]
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)


def get_atmosphere(image, scatterlight):
    """
    散射光校正与大气光估计函数
    参数:
        image: 输入图像数组(H×W×3)
        scatterlight: 散射光矩阵(H×W×3)
    返回:
        tuple: (大气光矩阵, 校正后的散射光)
    """
    # 散射光强度总和计算
    scatter_est = np.sum(scatterlight, axis=2)
    n_pixels = scatter_est.size
    # 确定搜索像素数量
    n_search_pixels = int(n_pixels * 0.001)  # 取前0.1%的像素

    # 将图像展平为2D数组
    image_vec = image.reshape(-1, 3)

    # 散射光强度排序
    # sorted_indices = np.argsort(scatter_est.flatten())
    sorted_indices = np.argsort(scatter_est.flatten())[::-1][:n_search_pixels]

    # 大气光估计

    atmosphere = np.mean(image_vec[sorted_indices], axis=0)
    # 大气光矩阵扩展
    atmosphere = np.tile(
        atmosphere[np.newaxis, np.newaxis, :],
        (scatter_est.shape[0], scatter_est.shape[1], 1)
    )

    # 计算阈值
    sek = scatter_est.ravel()[sorted_indices[-1]]
    # 计算修正向量
    scatter_est = np.nan_to_num(scatter_est)
    scatter_est = np.where(scatter_est == 0, 1e-10, scatter_est)
    sek_vec = np.tile(sek * scatterlight[0, 0, :] / scatter_est[0, 0], (scatter_est.shape[0], scatter_est.shape[1], 1))

    # sek_vec = np.tile(sek * np.divide(scatterlight[0, 0, :], scatter_est[0, 0],
    #                                  where=scatter_est[0, 0] != 0),
    #                  (scatter_est.shape[0], scatter_est.shape[1], 1))
    # 更新散射光
    mask = np.expand_dims(scatter_est <= sek, axis=2)
    scatterlight = scatterlight * mask + (2 * sek_vec - scatterlight) * (~mask)

    return atmosphere, scatterlight


def prctile20192(x, p, dim=None, method='exact'):
    """MATLAB 2019b prctile函数等效实现"""
    x = np.asarray(x)
    p = np.atleast_1d(p)

    if dim is None:
        dim = np.argmax(x.shape)

    if method == 'exact':
        return np.percentile(x, p, axis=dim, method='linear')
    elif method == 'approximate':
        return mstats.mquantiles(x, p, axis=dim)
    else:
        raise ValueError("Invalid method")


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    img_path = r"/underwater17.png"
    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float64) / 255.0  # 对应im2double(imread())
    plt.imshow(img)
    plt.show()
    # =================hyper-parameter============================
    # I = Jxt+A(1-t) =
    J, s_tildeT, tildeT, atmosphere = rank_one_prior(img)
