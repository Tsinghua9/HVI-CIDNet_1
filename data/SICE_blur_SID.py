
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from data.util import *
from data.transforms import transform1, transform_mask1
from torchvision import transforms as t
import torch.nn.functional as F

class LOLBlurDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLBlurDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed) 
            index = random.randint(0, 259)
            fill_index = str(index+1).zfill(4)
            folder = join(self.data_dir+'/low_blur', fill_index)
            folder2 = join(self.data_dir+'/high_sharp_scaled', fill_index)
            if  not os.path.exists(folder):
                continue
            data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
            data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
            num = len(data_filenames)
            if num != 0: break
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[index1-1])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, data_filenames[index1-1], data_filenames2[index1-1]

    def __len__(self):
        return 10200
    

class SIDDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, size=256, transform=None, mask_transform=None, label_dir=None,
                 return_index_map=False, max_regions: int = 16):
        super(SIDDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.size = size
        self.transform = transform or transform1(size)
        self.mask_transform = mask_transform if mask_transform is not None else transform_mask1(size)
        self.label_dir = label_dir
        self.return_index_map = return_index_map
        self.max_regions = int(max_regions)

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed) 
            index = random.randint(0, 233)
            fill_index = str(index+1).zfill(5)
            folder = join(self.data_dir+'/short', fill_index)
            folder2 = join(self.data_dir+'/long', fill_index)
            if os.path.exists(folder): 
                data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
                data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
                num = len(data_filenames)
                break
            else:
                continue
        index1 = random.randint(1,num)


        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[0])
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_filenames2[0])
        seed = np.random.randint(random.randint(1, 1000000)) # make a seed with numpy generator 
        label = None
        if self.return_index_map:
            label = self._resolve_label(file1, fill_index)
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
            if self.return_index_map:
                if self.mask_transform is None:
                    raise RuntimeError("mask_transform must be provided when return_index_map=True")
                random.seed(seed)
                torch.manual_seed(seed)
                label = self.mask_transform(label)
                label = self._remap_index_map(label, max_regions=self.max_regions)
        if self.return_index_map:
            return im1, im2, label, file1, file2
        return im1, im2, file1, file2

    def __len__(self):
        return 2099

    def _resolve_label(self, file_name, folder_index):
        if self.label_dir:
            base, _ = os.path.splitext(os.path.basename(file_name))
            folder_path = join(self.label_dir, folder_index)
            candidates = [
                join(folder_path, f"{base}_labels.png"),
                join(folder_path, f"{base}_labels.jpg"),
                join(folder_path, f"{base}_labels.JPG"),
            ]
            for path in candidates:
                if os.path.exists(path) and os.path.isfile(path):
                    return Image.open(path).convert('L')
            raise FileNotFoundError(f"Label file not found for {file_name} in {self.label_dir}")
        raise FileNotFoundError("label_dir is required for SID when return_index_map=True")

    @staticmethod
    def _remap_index_map(mask_tensor, max_regions: int = 16):
        mask_np = mask_tensor.numpy()
        uniq = np.unique(mask_np)
        remapped = np.searchsorted(uniq, mask_np)
        if int(max_regions) > 0:
            k = int(remapped.max()) + 1 if remapped.size else 0
            if k > max_regions:
                counts = np.bincount(remapped.reshape(-1), minlength=k)
                keep_n = max(int(max_regions) - 1, 1)
                keep_ids = np.argsort(counts)[::-1][:keep_n]
                mapping = np.zeros((k,), dtype=np.int64)
                for new_id, old_id in enumerate(keep_ids, start=1):
                    mapping[int(old_id)] = int(new_id)
                remapped = mapping[remapped]
        return torch.from_numpy(remapped.astype(np.int64))
    
    
    
class SICEDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, size=160, transform=None, mask_transform=None, label_dir=None,
                 return_index_map=False, max_regions: int = 16):
        super(SICEDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.size = size
        self.transform = transform or transform1(size)
        self.mask_transform = mask_transform if mask_transform is not None else transform_mask1(size)
        self.label_dir = label_dir
        self.return_index_map = return_index_map
        self.max_regions = int(max_regions)

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed)
            index = random.randint(0, 590)
            fill_index = str(index+1)
            train, _ = os.path.split(self.data_dir)
            folder = join(self.data_dir, fill_index)
            data_gt = join(train+'/label', fill_index+'.JPG')
            if os.path.exists(folder):
                data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
                num = len(data_filenames)
                break
            else:
                continue
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_gt)
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_gt)
        seed = np.random.randint(random.randint(1, 1000000))
        label = None
        if self.return_index_map:
            label = self._resolve_label(file1, data_gt, fill_index)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
            if self.return_index_map:
                if self.mask_transform is None:
                    raise RuntimeError("mask_transform must be provided when return_index_map=True")
                random.seed(seed)
                torch.manual_seed(seed)
                label = self.mask_transform(label)
                label = self._remap_index_map(label, max_regions=self.max_regions)
        if self.return_index_map:
            return im1, im2, label, file1, file2
        return im1, im2, file1, file2

    def __len__(self):
        return 4803

    def _resolve_label(self, file_name, default_gt, folder_index=None):
        if self.label_dir:
            base, _ = os.path.splitext(os.path.basename(file_name))
            candidates = []
            if folder_index:
                folder_path = join(self.label_dir, folder_index)
                candidates.extend([
                    join(folder_path, f"{base}_labels.png"),
                    join(folder_path, f"{base}_labels.jpg"),
                    join(folder_path, f"{base}_labels.JPG"),
                ])
            candidates.extend([
                join(self.label_dir, base),
                join(self.label_dir, base + '.png'),
                join(self.label_dir, base + '.jpg'),
                join(self.label_dir, base + '.JPG'),
            ])
            for path in candidates:
                if os.path.exists(path) and os.path.isfile(path):
                    return Image.open(path).convert('L')
            raise FileNotFoundError(f"Label file not found for {file_name} in {self.label_dir}")
        if os.path.exists(default_gt):
            return Image.open(default_gt).convert('L')
        raise FileNotFoundError(f"Default GT not found for {file_name}")

    @staticmethod
    def _remap_index_map(mask_tensor, max_regions: int = 16):
        mask_np = mask_tensor.numpy()
        uniq = np.unique(mask_np)
        remapped = np.searchsorted(uniq, mask_np)
        if int(max_regions) > 0:
            k = int(remapped.max()) + 1 if remapped.size else 0
            if k > max_regions:
                counts = np.bincount(remapped.reshape(-1), minlength=k)
                keep_n = max(int(max_regions) - 1, 1)
                keep_ids = np.argsort(counts)[::-1][:keep_n]
                mapping = np.zeros((k,), dtype=np.int64)
                for new_id, old_id in enumerate(keep_ids, start=1):
                    mapping[int(old_id)] = int(new_id)
                remapped = mapping[remapped]
        return torch.from_numpy(remapped.astype(np.int64))
