"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506060417
"""

import os
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
from typing import Dict
from PIL import Image
import cv2
import random
import torchvision.transforms.functional as F

from ..common.utils import log_print, load_json_data


class cleanPathVQAset(Dataset): # (sample_idx-image-prior-query-answer-split)
    def __init__(self, 
        meta_path: str, 
        data_path: str, 
        split: str,
        img_processor, 
        txt_processor, 
        croped_data: bool = False, 
        testing: int = None,
    ):
        log_print(f"Building...", head=True)

        self.meta_path = meta_path
        self.data_path = data_path
        self.croped_data = croped_data
        self.img_processor = img_processor
        self.txt_processor = txt_processor
        self.split = split

        self.meta_list = []
        _meta_list = load_json_data(self.meta_path)
        for single_data_dict in tqdm(_meta_list):
            if single_data_dict['split'] == split:
                self.meta_list += [single_data_dict]

        if testing is not None:
            log_print(f"using testing size: {testing}")
            self.meta_list = self.meta_list[:testing]

        log_print(f"using img size: {self.img_processor.image_size}")
        log_print(f"using split: {self.split}")
        log_print(f"data len: {len(self.meta_list)}")
        log_print("...Done\n")
        
    def __len__(self):
        return len(self.meta_list)
    
    def __getitem__(self, idx):

        global_idx = self.meta_list[idx]['sample_idx']

        if not self.croped_data:
            image_path = os.path.join(self.data_path, self.meta_list[idx]['image_path'])
            # image = Image.open(image_path).convert("RGB")
        else:
            image_path = os.path.join(self.data_path, self.meta_list[idx]['cleaned_image_path'])
        image = cv2.imread(image_path)

        if self.split == 'train':
            input_image = self.img_processor(image)
        else:
            input_image = self.img_processor.testing(image)

        caption = self.meta_list[idx]['caption']
        input_caption = self.txt_processor(caption)

        question = self.meta_list[idx]['question']
        input_question = self.txt_processor(question)

        prior = self.meta_list[idx]['prior']
        input_prior = self.txt_processor(prior)

        return {
            'idx': global_idx,
            'image': input_image,
            # 'caption': input_caption,
            'text_input': input_caption,
            'question': input_question,
            'prior': input_prior,
        }
    
    @classmethod
    def from_config(cls, 
        cfg, 
        split: str,
        img_processor,
        txt_processor,
    ):
        meta_path = str(cfg.get("meta_path"))
        data_path = str(cfg.get("data_path"))

        testing = cfg.get("test_data_num")
        if testing < 0:
            testing = None

        dataset = cls(
            meta_path=meta_path, 
            data_path=data_path, 
            split=split,
            img_processor=img_processor, 
            txt_processor=txt_processor, 
            testing=testing,
        )

        return dataset
    
    @classmethod
    def get_loader_from_config(cls, 
        cfg, 
        split: str,
        img_processor,
        txt_processor,
    ):
        meta_path = str(cfg.get("meta_path"))
        data_path = str(cfg.get("data_path"))
        croped_data = bool(cfg.get("croped_data"))

        multi_res_method = str(cfg.get("multi_res_method"))
        if multi_res_method == 'SimpleResize':
            pass
        elif multi_res_method == 'RatioPadInterp':
            pass
        elif multi_res_method == 'PatchPadBucket':
            bucket_size_multiplier = cfg.get("bucket_size_multiplier")

        testing = cfg.get("test_data_num")
        if type(testing) == dict:
            testing = testing.get(split)
        else:
            testing = None

        batch_size = int(cfg.get("batch_size"))
        num_workers = int(cfg.get("num_workers", 0))
        pin_memory = bool(cfg.get("pin_memory", True))
        shuffle = split == 'train'

        dataset = cls(
            meta_path=meta_path, 
            data_path=data_path, 
            croped_data=croped_data, 
            split=split,
            img_processor=img_processor, 
            txt_processor=txt_processor, 
            testing=testing,
        )

        if multi_res_method == 'PatchPadBucket':
            sampler = BucketBatchSampler(
                dataset=dataset, 
                batch_size=batch_size, 
                bucket_size_multiplier=bucket_size_multiplier, 
                shuffle=shuffle,
            )

            loader = DataLoader(
                dataset, 
                batch_sampler=sampler,
                collate_fn=center_pad_multimodal_collate,
                num_workers=num_workers, 
                pin_memory=pin_memory,
            )
        else:
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers, 
                pin_memory=pin_memory
            )

        return loader


def center_pad_multimodal_collate(
    batch, 
    patch_size: int = 32, 
    pad_value: float = 1.0,
):
    idxs        = [item['idx']        for item in batch]
    imgs        = [item['image']      for item in batch]
    captions    = [item['text_input'] for item in batch]
    questions   = [item['question']   for item in batch]
    priors      = [item['prior']      for item in batch]

    max_h = max(img.shape[1] for img in imgs)
    max_h = (max_h // patch_size + 1) * patch_size
    max_w = max(img.shape[2] for img in imgs)
    max_w = (max_w // patch_size + 1) * patch_size

    padded_imgs = []
    for img in imgs:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        img_pad = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                        fill=pad_value)
        padded_imgs.append(img_pad)

    images_tensor = torch.stack(padded_imgs, dim=0)          # (B, C, H_max, W_max)
    idx_tensor    = torch.as_tensor(idxs, dtype=torch.long)  # (B,)

    return {
        'idx'       : idx_tensor,
        'image'     : images_tensor,
        'text_input': captions,
        'question'  : questions,
        'prior'     : priors,
    }

class BucketBatchSampler(Sampler):
    def __init__(self, 
        dataset: Dataset, 
        batch_size: int, 
        bucket_size_multiplier: int = 100, 
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        
        indexed_resolutions = []
        for idx, sample in enumerate(self.dataset):
            _, h, w = sample['image'].shape
            area = h * w
            indexed_resolutions.append((idx, area))
        
        indexed_resolutions.sort(key=lambda x: x[1])
        self.sorted_indices = [idx for idx, _ in indexed_resolutions]
    
    def __iter__(self
    ):
        buckets = [
            self.sorted_indices[i : i + self.bucket_size]
            for i in range(0, len(self.sorted_indices), self.bucket_size)
        ]
        
        if self.shuffle:
            random.shuffle(buckets)
        
        batch_indices = []
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batch_indices.append(batch)
        
        if self.shuffle:
            random.shuffle(batch_indices)
        
        for b in batch_indices:
            yield b
    
    def __len__(self
    ):
        return len(self.dataset) // self.batch_size


















