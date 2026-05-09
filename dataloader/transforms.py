"""
Transforms and data augmentation for both image + bbox.
"""
import random
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    """
    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]
    """

    return cropped_image, target


def hflip(image, target):
    if type(image) is tuple:
        flipped_image = tuple(F.hflip(tmp_image) for tmp_image in image)
        w, h = image[0].size
    else:
        flipped_image = F.hflip(image)
        w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

# -------------------------- 2. 优化 resize 函数（支持只缩小、不放大） --------------------------
def resize(image, target, size, max_size=None, min_scale_only: bool = False):
    """
    对输入图像（或图像元组）进行等比例缩放（新增：只缩小、不放大逻辑）
    核心功能：保持图像长宽比的前提下缩放，同时修正标注的空间信息，保证标注与图像对齐
    新增功能：min_scale_only=True 时，仅当原始图像大于目标尺寸才缩放，否则保留原始尺寸
    Args:
        image: 输入图像（PIL.Image 或 图像元组）
        target: 图像对应的标注信息（字典格式，包含boxes/area/size/masks等键）
        size: 缩放目标尺寸（两种格式：1. 标量=最小边尺寸；2. (w, h) 元组=固定宽高）
        max_size: 最大边尺寸限制（可选，避免图像缩放后过大）
        min_scale_only: 仅缩小、不放大（True=不放大，False=正常缩放/放大）
    Returns:
        rescaled_image: 缩放后的图像（或图像元组）
        target: 同步更新后的标注信息（无标注时返回None）
    """
    # -------------------------- 内部辅助函数1：计算等比例缩放后的尺寸（新增不放大逻辑） --------------------------
    def get_size_with_aspect_ratio(image_size, size, max_size=None, min_scale_only=False):
        """
        保持图像长宽比，计算合理的缩放尺寸（核心：新增不放大逻辑，避免放大低分辨率图像）
        Args:
            image_size: 原始图像尺寸 (w, h)
            size: 目标最小边尺寸（标量）
            max_size: 目标最大边尺寸限制（标量）
            min_scale_only: 仅缩小、不放大
        Returns:
            (oh, ow): 缩放后的图像尺寸（高、宽），对应PIL.Image的尺寸格式
        """
        w, h = image_size  # 提取原始图像的宽、高
        original_min_size = float(min((w, h)))  # 原始图像最小边
        original_max_size = float(max((w, h)))  # 原始图像最大边

        # 新增：min_scale_only=True 时，若原始最小边 <= 目标尺寸，直接返回原始尺寸（不放大）
        if min_scale_only and original_min_size <= float(size):
            return (h, w)

        if max_size is not None:
            # 计算原始图像的长宽比（最小边/最大边）
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            # 防止缩放后最大边超过限制：若按最小边缩放后最大边超界，重新计算最小边尺寸
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        # 若图像已符合目标尺寸，直接返回原始尺寸（避免重复缩放）
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        # 等比例计算缩放后的宽、高（保证长宽比不变，无拉伸）
        if w < h:
            # 宽 < 高：以宽为基准缩放，高按比例计算
            ow = size  # 缩放后宽 = 目标最小边
            oh = int(size * h / w)  # 缩放后高 = 目标边 * 原始高/原始宽
        else:
            # 高 <= 宽：以高为基准缩放，宽按比例计算
            oh = size  # 缩放后高 = 目标最小边
            ow = int(size * w / h)  # 缩放后宽 = 目标边 * 原始宽/原始高

        return (oh, ow)  # 返回 (高, 宽)，适配torchvision的resize接口

    # -------------------------- 内部辅助函数2：解析缩放尺寸格式（传递 min_scale_only 参数） --------------------------
    def get_size(image_size, size, max_size=None, min_scale_only=False):
        """
        解析输入的size格式，返回最终的缩放尺寸
        支持两种size格式：1. 标量（等比例缩放）；2. (w, h)元组（固定尺寸缩放）
        Args:
            image_size: 原始图像尺寸 (w, h)
            size: 输入的目标尺寸（标量 或 (w, h)元组）
            max_size: 最大边尺寸限制（仅对标量size有效）
            min_scale_only: 仅缩小、不放大
        Returns:
            最终缩放尺寸（(h, w) 格式，适配torchvision.resize）
        """
        if isinstance(size, (list, tuple)):
            # 若size是列表/元组（固定宽高），反转顺序返回 (h, w)（适配PIL和torchvision的尺寸格式）
            return size[::-1]
        else:
            # 若size是标量，调用等比例缩放函数计算尺寸（传递不放大参数）
            return get_size_with_aspect_ratio(image_size, size, max_size, min_scale_only)

    # -------------------------- 主逻辑：处理图像缩放（传递 min_scale_only 参数） --------------------------
    if type(image) is tuple:
        # 情况1：输入是图像元组（如同时处理原图和增强图），批量缩放所有图像
        # 以元组中第一张图像为基准计算缩放尺寸（保证所有图像缩放比例一致）
        size = get_size(image[0].size, size, max_size, min_scale_only)
        # 对元组中每一张图像执行缩放，返回缩放后的图像元组
        rescaled_image = tuple(F.resize(tmp_image, size) for tmp_image in image)
        # 计算缩放比例（宽缩放比、高缩放比），用于后续更新标注信息
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, image[0].size))
        ratio_width, ratio_height = ratios
    else:
        # 情况2：输入是单张图像，直接计算缩放尺寸并执行缩放
        size = get_size(image.size, size, max_size, min_scale_only)
        rescaled_image = F.resize(image, size)
        # 计算缩放比例：缩放后尺寸 / 原始尺寸（宽、高分别计算）
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

    # -------------------------- 主逻辑：同步更新标注信息（target）--------------------------
    if target is None:
        # 无标注信息时，直接返回缩放后的图像和None
        return rescaled_image, None

    # 复制标注信息（避免修改原始标注数据，产生副作用）
    target = target.copy()

    if "boxes" in target:
        # 更新检测框坐标：将原始坐标按缩放比例映射到缩放后的图像上
        # boxes格式：(x1, y1, x2, y2) （左上角x/y，右下角x/y）
        boxes = target["boxes"]
        # 按宽、高缩放比分别更新四个坐标值，保持检测框与目标物体对齐
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        # 更新目标区域面积：原始面积 * 宽缩放比 * 高缩放比（面积是二维缩放，需乘两个方向的比例）
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "size" in target:
        # 更新标注中的图像尺寸信息，记录缩放后的图像高、宽
        h, w = size
        target["size"] = torch.tensor([h, w])

    if "masks" in target:
        # 更新掩码（mask）：对掩码张量进行插值缩放，保持掩码与图像对齐
        # 步骤：1. 增加维度（适配interpolate接口）；2. 浮点型插值；3. 去除多余维度；4. 二值化（保持掩码的0/1特性）
        target['masks'] = interpolate(
            target['masks'][:, None].float(),  # 增加通道维度 (N, 1, H, W)
            size,  # 缩放目标尺寸
            mode="nearest"  # 最近邻插值：保持掩码的像素类别（0/1）不模糊，适合分割任务
        )[:, 0] > 0.5  # 去除通道维度，二值化（>0.5保证掩码为布尔型/0-1张量）

    # 返回缩放后的图像和更新后的标注信息
    return rescaled_image, target

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(img.width if img.width < self.min_size else self.min_size, min(img.width, self.max_size))
        h = random.randint(img.height if img.height < self.min_size else self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None, min_scale_only: bool = False):
        """
        随机缩放（新增 min_scale_only 开关，控制是否只缩小、不放大）
        Args:
            sizes: 缩放目标尺寸列表（标量，最小边尺寸）
            max_size: 最大边尺寸限制
            min_scale_only: 若为True，仅当原始图像大于目标尺寸时才缩放（只缩小、不放大）
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.min_scale_only = min_scale_only  # 新增：不放大开关

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        # 调用 resize 函数，传递 min_scale_only 参数
        return resize(img, target, size, self.max_size, self.min_scale_only)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        if type(img) is tuple:
            Tensor_image = tuple(F.to_tensor(tmp_image) for tmp_image in img)

        else:
            Tensor_image = F.to_tensor(img)

        return Tensor_image, target




class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class RandomColorJitter(object):
    def __init__(self, p=0.5):
        self.p = p
        self.colorjitter = T.ColorJitter(brightness=0.40, contrast=0.40, saturation=0.40, hue=0.20)

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.colorjitter(img), target
        else:
            return img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        #image = F.normalize(image, mean=self.mean, std=self.std)
        if target == []:
            return image, []
        else:
            target = target.copy()
            h, w = image.shape[-2:]
            thres = 3 * (h * w) * 0.01 * 0.01

            if "boxes" in target:
                area = target['area']
                boxes = target['boxes']
                labels = target['labels']
                keep = area>thres
                target['area'] = area[keep]
                target['boxes'] = boxes[keep]
                target['labels'] = labels[keep]

                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string








