"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506060435
"""

import json
from typing import Dict, List
from datetime import datetime
import torch
import numpy as np
import random
import argparse
import yaml
from pprint import pprint
import matplotlib.pyplot as plt
import inspect
import os

from ..models.blip2Model.dist_utils import is_main_process

        
def log_print(
    text: str = '', 
    head: bool = False,
    newline: bool = True,
    traceback: bool = False,
):
    if is_main_process():
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        nowtime = datetime.now().strftime('%H:%M:%S')

        cls_name = None
        if 'self' in frame.f_locals:
            cls_name = frame.f_locals['self'].__class__.__name__
        elif 'cls' in frame.f_locals:
            cls_name = frame.f_locals['cls'].__name__

        if head:
            print()
        if cls_name:
            print(f"[{nowtime}] [{cls_name}.{func_name}] {text}", end='')
        else:
            print(f"[{nowtime}] [{func_name}] {text}", end='')
        if newline:
            print()
        if traceback:
            for line in inspect.stack():
                if line.function == func_name:
                    continue
                print(f"  -> {line.filename}:{line.lineno} in {line.function} on line {line.lineno}")

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class ConfigHandler:
    def __init__(self, 
        cfg, 
        best_metrics_list: List[Dict], 
        default_save_filename: str = 'best_model',
    ):
        log_print(f"Building...", head=True)

        self.cfg = cfg
        self.default_save_filename = default_save_filename

        if cfg['env'].get("root_path") == "":
            pwd = os.getcwd()
            cfg['env']['root_path'] = pwd
            log_print(f"Automatically set path on {pwd}")
        if cfg['env'].get("output_path") == "":
            cfg['env']['output_path'] = "output"
            log_print(f"Automatically set output path to 'output'")


        log_print(f"Loaded config:")
        pprint(self.cfg)


        self.nowtime = datetime.now().strftime("%y%m%d%H%M")
        self.save_path = os.path.join(self.cfg['env'].get("root_path"), self.cfg['env'].get("output_path"), self.nowtime)
        checkpath(self.save_path)


        self.log_save_path = os.path.join(self.save_path, self.nowtime + '_result.log')
        with open(self.log_save_path, "w") as file:
            # file.write(self.cfg + "\n")
            yaml.safe_dump(self.cfg, file, default_flow_style=False)
            file.write(f"\n===FOLLOWING_IS_RESULTS===\n[\n")

        self.update_temp_list = []
        self.best_metrics_list = best_metrics_list

        log_print(f"Saved config to {self.log_save_path}")
        log_print(f"...Done\n")

    def save_result(self, 
        result: Dict,
        update: bool = False,
        print_log: bool = False,
    ):
        def _apply(x):
            if torch.is_tensor(x):
                return x.item()
            elif isinstance(x, dict):
                return {key: _apply(value) for key, value in x.items()}
            elif isinstance(x, list):
                return [_apply(x) for x in x]
            else:
                return x
        result = _apply(result)

        if update:
            self.update_temp_list.append(result)

        with open(self.log_save_path, "a") as f:
            f.write(f"{str(result)}, \n")

        if print_log:
            log_print(f"Saved result to {self.log_save_path}")

    def get_update_avg(self, 
        print_log: bool = False,
    ):
        if len(self.update_temp_list) == 0:
            return None
        
        data_temp_dict = {}
        keys = []
        for key, value in self.update_temp_list[0].items():
            if isinstance(value, (int, float)):
                keys.append(key)

        data_temp_dict = {key: 0 for key in keys}
        for key in keys:
            for d in self.update_temp_list:
                data_temp_dict[key] += d.get(key, 0)
            data_temp_dict[key] /= len(self.update_temp_list)

        if print_log:
            log_print(f"Average {data_temp_dict}")

        self.update_temp_list = []
        return data_temp_dict

    def best_metrics_cond(self,
        metrics_dict: Dict, 
        weight_dict: Dict,
    ):
        for idx in range(len(self.best_metrics_list)):
            if self.best_metrics_list[idx]['cond'] == 'cur>best':
                if metrics_dict[self.best_metrics_list[idx]['key']] >= self.best_metrics_list[idx]['best']:
                    self.best_metrics_list[idx]['best'] = metrics_dict[self.best_metrics_list[idx]['key']]
                    weight_dict['metric_key'] = self.best_metrics_list[idx]['key']
                    weight_dict['metric_value'] = self.best_metrics_list[idx]['best']
                    self.save_weight(
                        weight_dict=weight_dict, 
                        save_filename=f"{self.best_metrics_list[idx]['key']}_best_model",
                        print_log=True,
                    )

            elif self.best_metrics_list[idx]['cond'] == 'cur<best':
                if metrics_dict[self.best_metrics_list[idx]['key']] <= self.best_metrics_list[idx]['best']:
                    self.best_metrics_list[idx]['best'] = metrics_dict[self.best_metrics_list[idx]['key']]
                    weight_dict['metric_key'] = self.best_metrics_list[idx]['key']
                    weight_dict['metric_value'] = self.best_metrics_list[idx]['best']
                    self.save_weight(
                        weight_dict=weight_dict, 
                        save_filename=f"{self.best_metrics_list[idx]['key']}_best_model",
                        print_log=True,
                    )

    def save_weight(self, 
        weight_dict: Dict, 
        save_filename: str = None, 
        print_log: bool = False,
    ):
        if is_main_process():
            if save_filename is None:
                save_filename = self.default_save_filename
            file_save_path = os.path.join(self.save_path, f'{self.nowtime}_{save_filename}.pth')
            torch.save(
                weight_dict, 
                file_save_path
            )

            if print_log:
                log_print(f"Saved weight to {file_save_path}")

    def close_result(self, 
        ):
        with open(self.log_save_path, "a") as f:
            f.write("]\n")

    @classmethod
    def get_cfg(cls,
        best_metrics_list: List[Dict], 
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg-path", required=True)
        args = parser.parse_args()

        with open(args.cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)

        cfg_handler = cls(
            cfg=cfg, 
            best_metrics_list=best_metrics_list,
        )
        return cfg_handler

def sample_device_adjust(
    sample, 
    cuda_enabled: bool = True
):
    if sample == None or len(sample) == 0:
        sample = {}

    if cuda_enabled:
        def _apply(x):
            if torch.is_tensor(x):
                return x.cuda()
            elif isinstance(x, dict):
                return {key: _apply(value) for key, value in x.items()}
            elif isinstance(x, list):
                return [_apply(x) for x in x]
            else:
                return x
        sample = _apply(sample)
    
    if not isinstance(sample, dict):
        sample = {"is_empty":True}

    return sample

class Config:
    def __init__(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def save_list2json(self,
    meta_list: List[Dict[str, str]], 
    save_filename: str = 'meta_list', 
):
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    file_save_path = os.path.join(self.save_path, f'{self.nowtime}_{save_filename}.json')
    with open(file_save_path, "w") as file:
        json.dump(meta_list, file, indent=4, default=convert)

    log_print(f"Saved list to {file_save_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def highlight(
    text: str = 'debug',
):
    return f"\033[1;31;40m{text}\033[0m"

def highlight_show(
        key: str, 
        text: str,
        bar: str = '=',
    ):
    print(f'[{key}]', bar * 43)
    print(text)
    print(f'[{key}]', bar * 43)
    
def calu_dict_avg(
    local_metrics_list: List[Dict[str, float]], 
    epoch_idx: int,
    state: str, 
    show: bool = False,
    avoid_key_list: List[str] = [],
) -> Dict[str, float]:
    local_metrics = {}
    for key in local_metrics_list[0]:
        if key not in avoid_key_list:
            local_metrics[key] = 0
        
    for single_dict in local_metrics_list:
        for key in local_metrics:
            local_metrics[key] += single_dict[key]

    if show:
        print('\n', f'[{state}_{str(epoch_idx)}_Results]', '=' * 43)
        for key in local_metrics:
            local_metrics[key] /=  len(local_metrics_list)
            print(key, local_metrics[key])
        print(f'[{state}_{str(epoch_idx)}_Results]', '=' * 43, '\n')

    return local_metrics

def load_result_logs(
    result_log_path: str,
) -> dict:
    log_print(f"Loading from {result_log_path}.log")
    data = []
    with open(result_log_path + ".log", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    d = json.loads(line)
                    data.append(d)
                except Exception as e:
                    d = eval(line)
                    data.append(d)
    log_print(f"Number of data: {len(data)}")

    unique_states = []
    for single_dict in data: 
        if single_dict.get('state') not in unique_states:
            unique_states.append(single_dict.get('state'))
    log_print(f"Loaded states: {unique_states}")    

    data_dict = {}
    for state in unique_states:
        state_data_list = []
        for d in data:
            if d['state'] == state:
                state_data_list.append(d)
        print(f"-> [{state}]: {len(state_data_list)}")

        keys = list(state_data_list[0].keys())
        data_dict[state] = {key: [] for key in keys}
        for d in state_data_list:
            for key in keys:
                data_dict[state][key].append(d.get(key, None))

        for key in data_dict[state]:
            print(f"----> [{key}]")

    log_print(f"...Done\n")

    return data_dict

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_data_list(
    data_list=None,
    key_list=None,
    diff_list=None,
    data_min:int=0,
    data_max:int=-1,
    title:str=None,
    ymin:int=None,
    ymax:int=None,
    smoothed:bool=True,
    window_size:int=51,
    save:str=None,
):
    if diff_list is not None:
        key_list = ['diff']
        data_list = {
            'diff': diff_list
        }
    data_len = len(data_list[key_list[0]][data_min:data_max])
    smoothed_values = moving_average(data_list[key_list[0]][data_min:data_max], window_size)
    smoothed_steps = list(range(1 + window_size // 2, data_len - window_size // 2 + 1))
    x_axis = list(range(0, data_len))

    plt.figure(figsize=(15, 7))
    for key in key_list:
        plt.plot(x_axis, data_list[key][data_min:data_max], marker='o', label=key)
    if smoothed:
        plt.plot(smoothed_steps, smoothed_values, label=f'Smoothed', color='red', linewidth=2)
    if diff_list is not None:
        ymax = plt.ylim()[1]
        ymin = plt.ylim()[0]
        plt.axhspan(ymin=ymin, ymax=0, facecolor='red', alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Value")
    if title is None:
        title = str(key_list)
    plt.title(title)
    plt.xlim(0, data_len)
    if ymax is not None and ymin is not None:
        plt.ylim(ymin, ymax)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(os.getcwd(), save + '.png'))
    plt.show()

def outputClass2lossDict(
    output
):
    loss_dict = {}
    for k,v in output.items():
        # if "loss" in k:
            loss_dict[k] = v

    return output["loss"], loss_dict






