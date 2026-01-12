
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from data.util import *
from torchvision import transforms as t
from data.options import option

# opt = option().parse_args()
opt, _ = option().parse_known_args()

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        mask_transform=None,
        label_dir=None,
        return_index_map=False,
        max_regions: int = 32,
    ):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_dir = label_dir
        self.return_index_map = return_index_map
        self.max_regions = int(max_regions)
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._init_pairs()

    def _init_pairs(self):
        low_dir = join(self.data_dir, 'low')
        high_dir = join(self.data_dir, 'high')
        low_files = sorted([join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)])
        high_files = sorted([join(high_dir, x) for x in listdir(high_dir) if is_image_file(x)])
        high_by_name = {os.path.basename(p): p for p in high_files}
        pairs = []
        for low_path in low_files:
            name = os.path.basename(low_path)
            high_path = high_by_name.get(name)
            if high_path is None:
                raise FileNotFoundError(f"Missing GT for low image: {name}")
            pairs.append((low_path, high_path, name))
        if len(pairs) == 0:
            raise RuntimeError(f"No image pairs found in {low_dir} and {high_dir}")
        self.pairs = pairs

    def __getitem__(self, index):

        low_path, high_path, file_name = self.pairs[index]
        im1 = load_img(low_path)
        im2 = load_img(high_path)
        file1 = file_name
        file2 = file_name
        label = None
        if self.return_index_map:
            label_dir = self.label_dir or join(self.data_dir, 'label')
            base, ext = os.path.splitext(file1)
            candidates = [
                join(label_dir, file1),
                join(label_dir, f"{base}_labels{ext}"),
                join(label_dir, f"{base.replace('_low','')}_labels{ext}")
            ]
            label_path = next((p for p in candidates if os.path.exists(p)), None)
            if label_path is None:
                raise FileNotFoundError(f"Label file not found for {file1} in {label_dir}")
            label = Image.open(label_path).convert('L')
        # seed = random.randint(1, 1000000)
        seed = opt.seed
        seed = np.random.randint(seed) # make a seed with numpy generator 
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
        return len(self.pairs)

    @staticmethod
    def _remap_index_map(mask_tensor, max_regions: int = 16):
        """
        把任意灰度值映射成连续 0..K-1 的 index_map。
        """
        mask_np = mask_tensor.numpy()
        uniq = np.unique(mask_np)
        remapped = np.searchsorted(uniq, mask_np)
        max_regions = int(max_regions)
        if max_regions > 0:
            k = int(remapped.max()) + 1 if remapped.size else 0
            # If too many regions, keep the largest (max_regions-1) and merge the rest into 0 ("other").
            if k > max_regions:
                counts = np.bincount(remapped.reshape(-1), minlength=k)
                keep_n = max_regions - 1
                # Always keep at least 1 region besides "other"
                if keep_n < 1:
                    keep_n = 1
                keep_ids = np.argsort(counts)[::-1][:keep_n]
                mapping = np.zeros((k,), dtype=np.int64)
                for new_id, old_id in enumerate(keep_ids, start=1):
                    mapping[int(old_id)] = int(new_id)
                remapped = mapping[remapped]
        return torch.from_numpy(remapped.astype(np.int64))


    
# class LOLDatasetFromFolder(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(LOLDatasetFromFolder, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform
#         self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def __getitem__(self, index):

#         folder = self.data_dir+'/low'
#         folder2= self.data_dir+'/high'
#         data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
#         data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
#         num = len(data_filenames)

#         im1 = load_img(data_filenames[index])
#         im2 = load_img(data_filenames2[index])
#         _, file1 = os.path.split(data_filenames[index])
#         _, file2 = os.path.split(data_filenames2[index])
#         # seed = random.randint(1, 1000000)
#         seed = opt.seed
#         seed = np.random.randint(seed) # make a seed with numpy generator 
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranfsorms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             im1 = self.transform(im1)
#             random.seed(seed)
#             torch.manual_seed(seed)         
#             im2 = self.transform(im2) 
#         return im1, im2, file1, file2

#     def __len__(self):
#         return 485
    
class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        mask_transform=None,
        label_dir=None,
        return_index_map=False,
        max_regions: int = 32,
    ):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_dir = label_dir
        self.return_index_map = return_index_map
        self.max_regions = int(max_regions)
        self._init_pairs()

    def _init_pairs(self):
        low_dir = join(self.data_dir, 'Low')
        high_dir = join(self.data_dir, 'Normal')
        low_files = sorted([join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)])
        high_files = sorted([join(high_dir, x) for x in listdir(high_dir) if is_image_file(x)])
        high_by_name = {os.path.basename(p): p for p in high_files}
        pairs = []
        for low_path in low_files:
            name = os.path.basename(low_path)
            high_path = high_by_name.get(name)
            if high_path is None:
                raise FileNotFoundError(f"Missing GT for low image: {name}")
            pairs.append((low_path, high_path, name))
        if len(pairs) == 0:
            raise RuntimeError(f"No image pairs found in {low_dir} and {high_dir}")
        self.pairs = pairs

    def __getitem__(self, index):
        low_path, high_path, file_name = self.pairs[index]
        im1 = load_img(low_path)
        im2 = load_img(high_path)
        file1 = file_name
        file2 = file_name
        label = None
        if self.return_index_map:
            if self.label_dir is None:
                raise RuntimeError("prior_label_dir (label_dir) must be provided for lolv2_real when use_region_prior=True")
            base, ext = os.path.splitext(file1)
            candidates = [
                join(self.label_dir, file1),
                join(self.label_dir, f"{base}_labels{ext}"),
                join(self.label_dir, f"{base}_label{ext}"),
            ]
            label_path = next((p for p in candidates if os.path.exists(p)), None)
            if label_path is None:
                raise FileNotFoundError(f"Label file not found for {file1} in {self.label_dir}")
            label = Image.open(label_path).convert('L')
        # seed = random.randint(1, 1000000)
        seed = opt.seed
        seed = np.random.randint(seed) # make a seed with numpy generator
        if self.transform:
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7 
            im2 = self.transform(im2)
            if self.return_index_map:
                if self.mask_transform is None:
                    raise RuntimeError("mask_transform must be provided when return_index_map=True")
                random.seed(seed)
                torch.manual_seed(seed)
                label = self.mask_transform(label)
                label = LOLDatasetFromFolder._remap_index_map(label, max_regions=self.max_regions)
        if self.return_index_map:
            return im1, im2, label, file1, file2
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.pairs)



class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        mask_transform=None,
        label_dir=None,
        return_index_map=False,
        max_regions: int = 16,
    ):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_dir = label_dir
        self.return_index_map = return_index_map
        self.max_regions = int(max_regions)
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._init_pairs()

    def _init_pairs(self):
        low_dir = join(self.data_dir, 'Low')
        high_dir = join(self.data_dir, 'Normal')
        low_files = sorted([join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)])
        high_files = sorted([join(high_dir, x) for x in listdir(high_dir) if is_image_file(x)])
        high_by_name = {os.path.basename(p): p for p in high_files}
        pairs = []
        for low_path in low_files:
            name = os.path.basename(low_path)
            high_path = high_by_name.get(name)
            if high_path is None:
                raise FileNotFoundError(f"Missing GT for low image: {name}")
            pairs.append((low_path, high_path, name))
        if len(pairs) == 0:
            raise RuntimeError(f"No image pairs found in {low_dir} and {high_dir}")
        self.pairs = pairs

    def __getitem__(self, index):
        low_path, high_path, file_name = self.pairs[index]
        im1 = load_img(low_path)
        im2 = load_img(high_path)
        file1 = file_name
        file2 = file_name
        label = None
        if self.return_index_map:
            if self.label_dir is None:
                raise RuntimeError("prior_label_dir (label_dir) must be provided for lolv2_syn when use_region_prior=True")
            base, ext = os.path.splitext(file1)
            candidates = [
                join(self.label_dir, file1),
                join(self.label_dir, f"{base}_labels{ext}"),
                join(self.label_dir, f"{base}_label{ext}"),
            ]
            label_path = next((p for p in candidates if os.path.exists(p)), None)
            if label_path is None:
                raise FileNotFoundError(f"Label file not found for {file1} in {self.label_dir}")
            label = Image.open(label_path).convert('L')
        # seed = random.randint(1, 1000000)
        seed = opt.seed
        seed = np.random.randint(seed) # make a seed with numpy generator 
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
                label = LOLDatasetFromFolder._remap_index_map(label, max_regions=self.max_regions)
        if self.return_index_map:
            return im1, im2, label, file1, file2
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.pairs)



    
