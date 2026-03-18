import torch
import numpy as np
from data.transforms import transform1, transform_mask1, transform2
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *



def get_lol_training_set(data_dir, size, label_dir=None, use_prior=False, max_regions: int = 16):
    mask_tf = transform_mask1(size) if use_prior else None
    return LOLDatasetFromFolder(
        data_dir,
        transform=transform1(size),
        mask_transform=mask_tf,
        label_dir=label_dir,
        return_index_map=use_prior,
        max_regions=max_regions,
    )

def get_lol_v2_training_set(data_dir, size, label_dir=None, use_prior=False, max_regions: int = 16):
    mask_tf = transform_mask1(size) if use_prior else None
    return LOLv2DatasetFromFolder(
        data_dir,
        transform=transform1(size),
        mask_transform=mask_tf,
        label_dir=label_dir,
        return_index_map=use_prior,
        max_regions=max_regions,
    )


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir, size, label_dir=None, use_prior=False, max_regions: int = 16):
    mask_tf = transform_mask1(size) if use_prior else None
    return LOLv2SynDatasetFromFolder(
        data_dir,
        transform=transform1(size),
        mask_transform=mask_tf,
        label_dir=label_dir,
        return_index_map=use_prior,
        max_regions=max_regions,
    )


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size,label_dir=None,use_prior=False,max_regions: int = 16):
    return SICEDatasetFromFolder(
        data_dir,
        size=size,
        label_dir=label_dir,
        return_index_map=use_prior,
        max_regions=max_regions,
    )

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())
