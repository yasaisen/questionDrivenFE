"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506111714
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common.utils import (
    log_print, 
    highlight_show, 
    highlight, 
    sample_device_adjust, 
    ConfigHandler,
    outputClass2lossDict, 
)


class BLIP2Trainer:
    def __init__(self,
        model: nn.Module, 
        cfg_handler: ConfigHandler,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4, 
        max_lr: float = 1e-3, 
        num_epoch: int = 30, 
        steps_per_epoch: int = 30, 
        pct_start: float = 0.2, 
        anneal_strategy: str = 'cos', 
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
        self.max_lr = max_lr
        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.use_amp = use_amp

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

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

        # if is_main_process():
        # log_print(f'Start training epoch {cur_epoch}, {iters_per_epoch} iters per inner epoch.')
        for idx, samples in enumerate(data_loader):
            samples = sample_device_adjust(samples, cuda_enabled=self.cuda_enabled)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(samples)
                loss, metrics = outputClass2lossDict(output)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

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
        log_print(f'[train] {cur_epoch} {avg_dict}')

        return metrics_list

    @torch.no_grad()
    def valid_inner_loop(self,
        cur_epoch: int,
        data_loader: DataLoader,
    ):
        self.model.eval()

        # if is_main_process():
        #     log_print(f'Start validing epoch {cur_epoch}, {iters_per_epoch} iters per inner epoch.')
        for idx, samples in enumerate(data_loader):
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
        log_print(f'[valid] {cur_epoch} {avg_dict}')

        return avg_dict

    @classmethod
    def from_config(cls, 
        cfg, 
        model: nn.Module, 
        steps_per_epoch: int, 
        cfg_handler: ConfigHandler,
    ):
        num_epoch = int(cfg.get("num_epoch"))
        learning_rate = float(cfg.get("learning_rate"))
        weight_decay = float(cfg.get("weight_decay"))
        max_lr = float(cfg.get("max_lr"))
        pct_start = float(cfg.get("pct_start"))
        anneal_strategy = str(cfg.get("anneal_strategy"))
        device = str(cfg.get("device"))

        trainer = cls(
            model=model, 
            cfg_handler=cfg_handler,
            learning_rate=learning_rate,
            weight_decay=weight_decay, 
            max_lr=max_lr, 
            num_epoch=num_epoch, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 
            device=device,
        )
        return trainer
    









