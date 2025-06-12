"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506122347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from __future__ import annotations
from typing import List, Sequence, Union, Iterable, Dict


def evenly_pick_indices(
    A: int, 
    B: int
) -> list[int]:
    if B <= 0:
        raise ValueError("B must be a positive integer")
    if A < B:
        raise ValueError("A ≥ B must be satisfied")

    if B == 1:
        return [(A - 1) // 2]

    step = (A - 1) / (B - 1)
    indices = [round(i * step) for i in range(B)]

    for i in range(1, B):
        if indices[i] <= indices[i - 1]:
            indices[i] = indices[i - 1] + 1

    indices[-1] = A - 1
    return indices

def amad_kl_loss(
    teacher_attns, 
    student_attns, 
    lambda_weight: int = 1.0, 
    teacher_regtok: int = 0, 
    student_regtok: int = 0, 
):
    total_loss = 0.0
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    tea_layer_idx_list = evenly_pick_indices(len(teacher_attns), len(student_attns))
    
    for stu_layer_idx in range(len(student_attns)):
        t_attn = teacher_attns[tea_layer_idx_list[stu_layer_idx]]  # (B, H_t, Q, K)
        s_attn = student_attns[stu_layer_idx]  # (B, H_s, Q, K)

        t_attn = t_attn[:, :, (1 + teacher_regtok):, (1 + teacher_regtok):]
        s_attn = s_attn[:, :, (1 + student_regtok):, (1 + student_regtok):]
        
        B, Ht, Q, K = t_attn.shape
        _, Hs, _, _ = s_attn.shape
        
        if t_attn.shape[2:] != s_attn.shape[2:]:
            s_attn = F.interpolate(
                s_attn, 
                size=(Q, K), 
                mode='bilinear', 
                align_corners=False
            )
        n = Q * K
        
        T_flat = t_attn.reshape(B, Ht, n)
        S_flat = s_attn.reshape(B, Hs, n)
        
        T_dist = T_flat / (T_flat.sum(dim=-1, keepdim=True) + 1e-8)
        S_dist = S_flat / (S_flat.sum(dim=-1, keepdim=True) + 1e-8)
        
        T_norm = F.normalize(T_dist, p=2, dim=-1)
        S_norm = F.normalize(S_dist, p=2, dim=-1)
        sim = torch.matmul(T_norm, S_norm.transpose(-1, -2))
        
        weight = F.softmax(sim / 0.1, dim=-1)
        
        S_aligned = torch.matmul(weight, S_dist)
        
        T_stable = T_dist + 1e-8
        S_stable = S_aligned + 1e-8
        
        T_kl = T_stable.view(-1, n)
        S_kl = S_stable.view(-1, n)

        eps = 1e-8
        T_kl = T_kl.clamp(min=eps)
        
        layer_loss = kl_div(S_kl.log(), T_kl)
        total_loss += lambda_weight * layer_loss
    
    return total_loss / len(student_attns)

class ViTAttentionExtractor:
    _HOOK_KEY = "attn_drop"
    def __init__(self,
        model: nn.Module,
        layers: Union[str, int, Sequence[int]] = "all",
        store_cpu: bool = True,
        keep_fused: bool = False,
    ):
        self.model = model
        self.store_cpu = store_cpu
        self.keep_fused = keep_fused
        self.maps: List[torch.Tensor] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._orig_fused: List[bool] = []

        if layers == "all":
            self._target_idx: Iterable[int] = range(len(model.blocks))
        elif isinstance(layers, int):
            self._target_idx = [layers]
        else:
            self._target_idx = layers

    def __enter__(self
    ):
        for blk in self.model.blocks:
            self._orig_fused.append(blk.attn.fused_attn)
            if not self.keep_fused:
                blk.attn.fused_attn = False

        for idx in self._target_idx:
            attn_drop = getattr(self.model.blocks[idx].attn, self._HOOK_KEY)
            h = attn_drop.register_forward_hook(self._make_hook(idx))
            self._handles.append(h)
        return self

    def __exit__(self, 
        exc_type, 
        exc_val, 
        exc_tb
    ):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for blk, orig in zip(self.model.blocks, self._orig_fused):
            blk.attn.fused_attn = orig

    def _make_hook(self, 
        layer_idx: int
    ):
        def _hook(
            module: nn.Module, 
            inputs, 
            outputs
        ):
            attn = inputs[0]
            if self.store_cpu:
                attn = attn.cpu()
            self.maps.append(attn)
        return _hook
    











    