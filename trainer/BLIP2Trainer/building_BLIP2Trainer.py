"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506130419
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ...common.utils import (
    log_print, 
    highlight_show, 
    highlight, 
    sample_device_adjust, 
    ConfigHandler,
    outputClass2lossDict, 
)

from ..AMADKDTrainer.building_AMADKDTrainer import (
    amad_kl_loss, 
    ViTAttentionExtractor, 
    get_teacher_model, 
    sample_patch_adjust,
)


GRAD_CLIP = 1.0
LR_DICT = {
    'pretrain': 1e-3, 
    'finetune': 5e-5, 
    'default': 2e-4, 
}
SCHEDULER = {
    'warmup_steps': 500, 
}


class BLIP2Trainer:
    def __init__(self,
        model: nn.Module, 
        cfg_handler: ConfigHandler,
        kd_teacher_name: str, 
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4, 
        num_epoch: int = 30, 
        steps_per_epoch: int = 30, 
        use_amp: bool = True, 
        device: str = "cuda",
    ):
        print()
        log_print(f"Building...")

        self.device = device
        self.cuda_enabled = self.device == "cuda"
        self.model = model.to(self.device)
        self.cfg_handler = cfg_handler

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.use_amp = use_amp

        # self.optimizer = optim.AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()), 
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        # )
        param_groups = [
            {"params": [], "lr": LR_DICT['finetune'], "weight_decay": self.weight_decay, "name": "visual_encoder"},
            {"params": [], "lr": LR_DICT['pretrain'], "weight_decay": self.weight_decay, "name": "cross_attention"},
            {"params": [], "lr": LR_DICT['finetune'], "weight_decay": self.weight_decay, "name": "qformer_other"},
            {"params": [], "lr": LR_DICT['pretrain'], "weight_decay": self.weight_decay, "name": "projections"},
            {"params": [], "lr": LR_DICT['default'], "weight_decay": self.weight_decay, "name": "default"}
        ]
        for name, params in model.named_parameters():
            if not params.requires_grad:
                continue
            if "visual_encoder" in name:
                param_groups[0]["params"].append(params)
            elif "Qformer" in name and "crossattention" in name:
                param_groups[1]["params"].append(params)
            elif "Qformer" in name:
                param_groups[2]["params"].append(params)
            elif any(proj in name for proj in ["vision_proj", "text_proj"]):
                param_groups[3]["params"].append(params)
            else:
                param_groups[4]["params"].append(params)
        self.optimizer = torch.optim.AdamW(
            [group for group in param_groups if group["params"]], 
            betas=(0.9, 0.999),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=SCHEDULER['warmup_steps'], 
            num_training_steps=self.steps_per_epoch * self.num_epoch,
        )

        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.do_kd = kd_teacher_name != 'NoDistill'
        if self.do_kd:
            self.teacher_model, self.teacher_patch_size = get_teacher_model(
                model_name=kd_teacher_name,
                weight_path=self.model.weight_path, 
                device=self.device,
            )
        log_print(f"...Done\n")

    @staticmethod
    def loader_checker(
        data_loader: DataLoader,
    ):
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)
        iters_per_epoch = len(data_loader)

        return data_loader, iters_per_epoch

    def train_inner_loop(self,
        cur_epoch: int,
        data_loader: DataLoader,
    ):
        self.model.train()
        metrics_list = []

        log_print(f'\nStart training epoch {cur_epoch}, {len(data_loader)} iters per inner epoch.')
        # for idx, samples in enumerate(data_loader):
        for idx, samples in tqdm(enumerate(data_loader), total=len(data_loader)):
            samples = sample_device_adjust(samples, cuda_enabled=self.cuda_enabled)

            if self.do_kd:
                with torch.no_grad():
                    with ViTAttentionExtractor(self.teacher_model) as ex_t:
                        teacher_samples = sample_patch_adjust(
                            samples=samples['image'], 
                            patch_size=self.teacher_patch_size,
                        )
                        _ = self.teacher_model(teacher_samples)
                        teacher_attns = ex_t.maps

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    with ViTAttentionExtractor(self.model.visual_encoder) as ex_s:
                        output = self.model(samples)
                        loss, metrics = outputClass2lossDict(output)

                        student_attns = ex_s.maps

                attn_loss_kl = amad_kl_loss(
                    teacher_attns, 
                    student_attns, 
                    lambda_weight=0.5, 
                    teacher_regtok=self.teacher_model.num_reg_tokens, 
                    student_regtok=self.model.visual_encoder.num_reg_tokens, 
                )
                metrics.update({
                    'loss_kl': attn_loss_kl.item(), 
                })
                total_loss = loss + attn_loss_kl

            else:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(samples)
                    loss, metrics = outputClass2lossDict(output)
                total_loss = loss

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(total_loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                self.optimizer.step()

            self.scheduler.step()
            # self.optimizer.zero_grad()

            metrics.update({
                'epoch': cur_epoch,
                'iter': idx + 1,
                'lr': self.scheduler.get_last_lr()[0],
                'stage': 'train',
            })
            self.cfg_handler.save_result(
                result=metrics, 
                update=True,
            )
            metrics_list.append(metrics)

        avg_dict = self.cfg_handler.get_update_avg()
        log_print(f'[train] [{cur_epoch}] {avg_dict}')

        return metrics_list

    @torch.no_grad()
    def valid_inner_loop(self,
        cur_epoch: int,
        data_loader: DataLoader,
    ):
        self.model.eval()

        log_print(f'Start validing epoch {cur_epoch}, {len(data_loader)} iters per inner epoch.')
        # for idx, samples in enumerate(data_loader):
        for idx, samples in tqdm(enumerate(data_loader), total=len(data_loader)):
            samples = sample_device_adjust(samples, cuda_enabled=self.cuda_enabled)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(samples)
                loss, metrics = outputClass2lossDict(output)

            metrics.update({
                'epoch': cur_epoch,
                'iter': idx + 1,
                'stage': 'valid',
            })
            self.cfg_handler.save_result(
                result=metrics, 
                update=True,
            )

        avg_dict = self.cfg_handler.get_update_avg()
        log_print(f'[valid] [{cur_epoch}] {avg_dict}')

        return avg_dict

    @classmethod
    def from_config(cls, 
        cfg, 
        model: nn.Module, 
        steps_per_epoch: int, 
        cfg_handler: ConfigHandler,
    ):
        kd_teacher_name = cfg.get("kd_teacher_name")
        num_epoch = int(cfg.get("num_epoch"))
        learning_rate = float(cfg.get("learning_rate"))
        weight_decay = float(cfg.get("weight_decay"))
        device = str(cfg.get("device"))

        trainer = cls(
            model=model, 
            cfg_handler=cfg_handler, 
            kd_teacher_name=kd_teacher_name, 
            learning_rate=learning_rate,
            weight_decay=weight_decay, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            device=device,
        )
        return trainer
    









