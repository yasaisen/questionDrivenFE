"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502251532
"""

import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from ..common.utils import log_print


class cleanPCset_v1_Dataset(Dataset):
    def __init__(self, 
        meta_list, 
        data_path, 
        split,
        img_processor, 
        text_processor=None, 
        logits_path=None,
        device='cuda',
    ):
        self.state_name = 'cleanPCset_v1_Dataset'
        self.device = device
        print()
        log_print("Building...")

        self.data_path = data_path
        self.logits_path = logits_path
        self.img_processor = img_processor
        self.text_processor = text_processor
        self.split = split

        self.meta_list = []
        for single_data_dict in tqdm(meta_list):
            if single_data_dict['split'] == split:
                self.meta_list += [single_data_dict]

        log_print(f"using size: {self.img_processor.size}")
        log_print(f"using split: {self.split}")
        log_print(f"data len: {len(self.meta_list)}")
        log_print("...Done\n")
        
    def __len__(self):
        return len(self.meta_list)
    
    def __getitem__(self, idx):

        global_idx = self.meta_list[idx]['question_idx']

        image_path = os.path.join(self.data_path, self.meta_list[idx]['image_path'])
        image = Image.open(image_path).convert("RGB")
        input_image = self.img_processor(image).to(self.device)

        # input_caption = self.meta_list[idx]['caption']
        if self.logits_path is None:
        #     input_caption_ids = self.text_processor(input_caption).to(self.device)

            return {
                'image': input_image,
                # 'caption': input_caption,
                # 'caption_ids': input_caption_ids,
                'idx': global_idx
            }

        else:
            logit_file_name = 'text_logits_' + str(global_idx).zfill(7) + '.pt'
            logit_file_path = os.path.join(self.data_path, self.logits_path, logit_file_name)
            text_logits = torch.load(logit_file_path).squeeze(0).to(torch.float32).to(self.device)

            return {
                'image': input_image,
                # 'caption': input_caption,
                # 'text_logits': text_logits,
                'idx': global_idx
            }
        











