"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2505272219
"""

import cv2
import numpy as np
import random
import albumentations as A
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode


class ImgProcessor():
    def __init__(self, 
        image_size: int = None, 
        mean: tuple = None, 
        std: tuple = None, 
        min_scale: float = 0.5, 
        max_scale: float = 1.0
    ):

        self.image_size = image_size
        self.min_scale = min_scale
        self.max_scale = max_scale

        if mean is None:
            self.mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            self.std = (0.26862954, 0.26130258, 0.27577711)

        self.aug_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),

            # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.GaussNoise(std_range=(0.3, 0.7), mean_range=(0.0, 0.0), p=0.3),

            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.1), 

            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.1), 
            A.ElasticTransform(alpha=1, sigma=50, interpolation=cv2.INTER_CUBIC, p=0.1), 

            # A.CoarseDropout(max_holes=3, max_height=20, max_width=20, fill_value=random.randint(100, 200), p=0.1),
            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(5, 20), hole_width_range=(5, 20), fill=random.randint(100, 200), p=0.1),

        ])
        self.pre_transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     image_size,
            #     scale=(self.min_scale, self.max_scale),
            #     interpolation=InterpolationMode.BICUBIC,
            # ),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=self.mean, 
            #     std=self.std
            # ),
        ])
        
    @staticmethod
    def macenko_normalization(
        I: np.ndarray, 
        Io: int = 240, 
        alpha: int = 1, 
        beta: float = 0.15, 
        target_max=None
    ):
        I = I.astype(np.float32)
        OD = -np.log((I + 1) / Io)
        
        mask = (I < Io).all(axis=2)
        OD_hat = OD[mask].reshape(-1, 3)
        OD_hat = OD_hat[np.max(OD_hat, axis=1) > beta]
        
        if OD_hat.shape[0] == 0:
            return I.astype(np.uint8)
        
        U, S, V = np.linalg.svd(OD_hat, full_matrices=False)
        V = V[:2, :].T
        
        That = np.dot(OD_hat, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        stain_matrix = np.array([v1, v2]).T
        stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
        
        OD_flat = OD.reshape((-1, 3)).T
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, OD_flat, rcond=None)
        
        maxC = np.percentile(concentrations, 99, axis=1)
        if target_max is None:
            target_max = maxC
        
        norm_concentrations = concentrations * (target_max[:, None] / maxC[:, None])
        OD_normalized = np.dot(stain_matrix, norm_concentrations)
        
        I_normalized = Io * np.exp(-OD_normalized)
        I_normalized = I_normalized.T.reshape(I.shape)
        I_normalized = np.clip(I_normalized, 0, 255).astype(np.uint8)
        
        return I_normalized
    
    def __call__(self, 
        image
    ):
        aug_image = self.aug_pipeline(image=image)['image']
        aug_image = self.macenko_normalization(aug_image)
        aug_image = transforms.ToPILImage()(aug_image)
        aug_image = self.pre_transform(aug_image)

        if self.image_size is not None:
            aug_image = F.interpolate(aug_image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        return aug_image

    def testing(self, 
        image
    ):
        # aug_image = self.aug_pipeline(image=image)['image']
        aug_image = self.macenko_normalization(aug_image)
        aug_image = transforms.ToPILImage()(aug_image)
        aug_image = self.pre_transform(aug_image)

        if self.image_size is not None:
            aug_image = F.interpolate(aug_image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        return aug_image

    @classmethod
    def from_config(cls, 
        cfg
    ):
        image_size = cfg.get("image_size", None)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        processor = cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        return processor










