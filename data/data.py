import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *

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



def get_lol_training_set(data_dir,size,label_dir=None,use_prior=False):
    mask_tf = transform_mask1(size) if use_prior else None
    return LOLDatasetFromFolder(
        data_dir,
        transform=transform1(size),
        mask_transform=mask_tf,
        label_dir=label_dir,
        return_index_map=use_prior
    )

def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())
