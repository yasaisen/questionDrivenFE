"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506132139
"""

import cv2
import numpy as np
import random
import albumentations as A
from torchvision import transforms
import torch.nn.functional as F
from typing import Dict
import torch

from ..common.utils import log_print, get_trainable_params, highlight, highlight_show


class ImgProcessor():
    def __init__(self, 
        image_size: int = None, 
        center_pad_dict: Dict = None, 

    ):
        self.center_pad_dict = center_pad_dict
        self.image_size = image_size

        self.aug_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),

            A.GaussNoise(std_range=(0.3, 0.7), mean_range=(0.0, 0.0), p=0.3),

            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.1), 

            A.ElasticTransform(alpha=1, sigma=50, interpolation=cv2.INTER_CUBIC, p=0.1), 

            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(5, 20), hole_width_range=(5, 20), fill=random.randint(100, 200), p=0.1),

        ])
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __call__(self, 
        image
    ):
        image = self.aug_pipeline(image=image)['image']
        if self.center_pad_dict is None:
            aug_image = macenko_normalization(image)
        else:
            aug_image = macenko_normalization_manyWhite(image)

            if np.array_equal(aug_image, image):
                log_print("[Warning] Skipped Macenko regularization")
                aug_image = statistical_normalization(
                    image, 
                    target_mean=128, 
                    target_std=40
                )
        aug_image = transforms.ToPILImage()(aug_image)
        aug_image = self.pre_transform(aug_image)

        if self.image_size is not None:
            aug_image = F.interpolate(
                aug_image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)

        if self.center_pad_dict is not None:
            aug_image = center_padding(
                image=aug_image, 
                size_ratio=self.center_pad_dict['size_ratio'], 
                new_img_height=self.center_pad_dict['new_img_height'], 
                pad_value=self.center_pad_dict['pad_value'], 
            )

        return aug_image

    def testing(self, 
        image
    ):
        if self.center_pad_dict is None:
            aug_image = macenko_normalization(image)
        else:
            aug_image = macenko_normalization_manyWhite(image)

            if np.array_equal(aug_image, image):
                log_print("[Warning] Skipped Macenko regularization")
                aug_image = statistical_normalization(
                    image, 
                    target_mean=128, 
                    target_std=40
                )

        aug_image = transforms.ToPILImage()(aug_image)
        aug_image = self.pre_transform(aug_image)

        if self.image_size is not None:
            aug_image = F.interpolate(
                aug_image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
        
        if self.center_pad_dict is not None:
            aug_image = center_padding(
                image=aug_image, 
                size_ratio=self.center_pad_dict['size_ratio'], 
                new_img_height=self.center_pad_dict['new_img_height'], 
                pad_value=self.center_pad_dict['pad_value'], 
            )

        return aug_image

    @classmethod
    def from_config(cls, 
        cfg
    ):
        multi_res_method = str(cfg.get("multi_res_method"))
        if multi_res_method == 'SimpleResize':
            image_size = cfg.get("image_size")
            center_pad_dict = None
        elif multi_res_method == 'RatioPadInterp':
            image_size = None
            center_pad_dict = {
                'size_ratio': cfg.get("size_ratio"), 
                'new_img_height': cfg.get("new_img_height"), 
                'pad_value': cfg.get("pad_value"), 
            }
        elif multi_res_method == 'PatchPadBucket':
            image_size = None
            center_pad_dict = None

        processor = cls(
            image_size=image_size,
            center_pad_dict=center_pad_dict, 
        )

        return processor

def statistical_normalization(image, target_mean=128, target_std=50, mask=None):
    image = image.astype(np.float32)
    
    if mask is None:
        mask = (image < 220).all(axis=2)
    
    normalized = image.copy()
    
    for i in range(3):
        channel = image[:, :, i]
        if np.sum(mask) > 0:
            valid_pixels = channel[mask]
            if len(valid_pixels) > 0 and np.std(valid_pixels) > 0:
                current_mean = np.mean(valid_pixels)
                current_std = np.std(valid_pixels)
                
                normalized_channel = (channel - current_mean) / current_std
                normalized_channel = normalized_channel * target_std + target_mean
                
                normalized[:, :, i] = channel.copy()
                normalized[mask, i] = normalized_channel[mask]
            else:
                normalized[:, :, i] = channel
        else:
            normalized[:, :, i] = channel
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

def macenko_normalization_manyWhite(
    I: np.ndarray, 
    Io: int = 240, 
    alpha: int = 1, 
    beta: float = 0.15, 
    target_max=None,
    white_threshold: int = 220
):
    I = I.astype(np.float32)
    
    non_white_mask = (I < white_threshold).all(axis=2)
    tissue_mask = (I < Io).all(axis=2)
    
    valid_mask = non_white_mask & tissue_mask
    
    valid_pixel_count = np.sum(valid_mask)
    total_pixels = I.shape[0] * I.shape[1]
    valid_ratio = valid_pixel_count / total_pixels
    
    if valid_ratio < 0.01:
        # print("警告：有效像素太少，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    OD = -np.log((I + 1) / Io)
    OD_hat = OD[valid_mask].reshape(-1, 3)
    
    OD_hat = OD_hat[np.max(OD_hat, axis=1) > beta]
    
    if OD_hat.shape[0] < 10:
        # print("警告：有效光密度點太少，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    try:
        U, S, V = np.linalg.svd(OD_hat, full_matrices=False)
        
        if len(S) < 2 or S[1] < 1e-6:
            # print("警告：SVD 結果不穩定，跳過 Macenko 正規化")
            return I.astype(np.uint8)
            
        V = V[:2, :].T
        
    except np.linalg.LinAlgError:
        # print("警告：SVD 分解失敗，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    That = np.dot(OD_hat, V)
    
    if That.shape[1] < 2:
        # print("警告：投影結果維度不足，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    
    stain_matrix = np.array([v1, v2]).T
    
    if np.any(np.linalg.norm(stain_matrix, axis=0) < 1e-6):
        # print("警告：染色矩陣退化，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
    
    OD_flat = OD.reshape((-1, 3)).T
    try:
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, OD_flat, rcond=None)
    except np.linalg.LinAlgError:
        # print("警告：濃度計算失敗，跳過 Macenko 正規化")
        return I.astype(np.uint8)
    
    valid_concentrations = concentrations[:, valid_mask.flatten()]
    if valid_concentrations.shape[1] > 0:
        maxC = np.percentile(valid_concentrations, 99, axis=1)
    else:
        maxC = np.percentile(concentrations, 99, axis=1)
    
    if target_max is None:
        target_max = maxC.copy()
    
    epsilon = 1e-6
    
    if np.any(np.abs(maxC) < epsilon) or np.any(~np.isfinite(maxC)):
        norm_concentrations = concentrations.copy()
    else:
        norm_concentrations = concentrations * (target_max[:, None] / maxC[:, None])
    
    OD_normalized = np.dot(stain_matrix, norm_concentrations)
    I_normalized = Io * np.exp(-OD_normalized)
    I_normalized = I_normalized.T.reshape(I.shape)
    
    result = I_normalized.copy()
    white_pixels = ~valid_mask
    result[white_pixels] = I[white_pixels]
    
    result = np.nan_to_num(result, nan=0.0, posinf=255.0, neginf=0.0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

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
        target_max = maxC.copy()
    
    epsilon = 1e-6
    
    if np.any(np.abs(maxC) < epsilon) or np.any(~np.isfinite(maxC)):
        norm_concentrations = concentrations.copy()
    else:
        norm_concentrations = concentrations * (target_max[:, None] / maxC[:, None])
    
    OD_normalized = np.dot(stain_matrix, norm_concentrations)
    
    I_normalized = Io * np.exp(-OD_normalized)
    I_normalized = I_normalized.T.reshape(I.shape)
    
    I_normalized = np.nan_to_num(I_normalized, nan=0.0, posinf=255.0, neginf=0.0)
    I_normalized = np.clip(I_normalized, 0, 255).astype(np.uint8)
    
    return I_normalized

def center_padding(
    image: torch.Tensor, 
    size_ratio: float, 
    new_img_height: int = None, 
    pad_value: float = 1.0,
):
    # _, old_height, old_width = image.shape
    
    # new_height = int(old_height)
    # new_width = int(new_height * size_ratio)

    # pad_h = new_height - old_height
    # pad_w = new_width - old_width

    # pad_top    = pad_h // 2
    # pad_bottom = pad_h - pad_top
    # pad_left   = pad_w // 2
    # pad_right  = pad_w - pad_left

    # image = F.pad(
    #     image, 
    #     (pad_left, pad_top, pad_right, pad_bottom),
    #     value=pad_value
    # )

    if new_img_height is not None:
        image = F.interpolate(
            image.unsqueeze(0), 
            size=(new_img_height, int(new_img_height * size_ratio)), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

    return image








