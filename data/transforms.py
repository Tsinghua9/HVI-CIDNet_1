import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda


def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def transform_mask1(size=256):
    """
    与 transform1 相同的裁剪/翻转，保持标签为 int64。
    """
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Lambda(lambda pic: torch.from_numpy(np.array(pic, dtype=np.int64)))
    ])


def transform2():
    return Compose([ToTensor()])
