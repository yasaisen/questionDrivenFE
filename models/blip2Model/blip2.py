"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 This file is modified from Salesforce (BSD-3-Clause):
   Copyright (c) 2023, salesforce.com, inc.
   SPDX-License-Identifier: BSD-3-Clause
   See: https://opensource.org/licenses/BSD-3-Clause

 last modified in 2506080043
"""

import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from ..blip2Model.dist_utils import (
    get_world_size, 
    get_rank, 
    is_dist_avail_and_initialized,
)
from ..blip2Model.base_model import BaseModel
from ..blip2Model.Qformer import BertConfig, BertLMHeadModel


WEIGHT_MAPPING_DICT = {
    'biogpt_embedding': 'biogpt_embedding.pth', 

    'prov-gigapath': 'tile_encoder.pth', 
    'H-optimus-0': 'H-optimus-0.pth', 
    'vit_base': '.pth', 
    'vit_large': '.pth', 

    'histogpt-config': 'biogpt-large-config.json', 
    'histogpt-3b': 'histogpt-3b-6k-pruned.pth', 
    'histogpt-1b': 'histogpt-1b-6k-pruned.pth', 
}

import timm
from transformers import BioGptConfig, BioGptTokenizer
from ...models.histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
from ...models.histogpt.helpers.inference import generate


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):

        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer = BioGptTokenizer.from_pretrained(
            "microsoft/biogpt", 
            truncation_side=truncation_side
        )
        
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, 
        weight_path: str,
        num_query_token: int, 
        vision_width: int, 
        cross_attention_freq: int = 2,
    ):
        bert_name = "bert-large-uncased" # "bert-base-uncased"
        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.vocab_size = 42384

        Qformer = BertLMHeadModel.from_pretrained(
            bert_name, 
            config=encoder_config, 
            ignore_mismatched_sizes=True
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


        partial_state_dict = torch.load(os.path.join(weight_path, WEIGHT_MAPPING_DICT['biogpt_embedding']))
        Qformer.bert.embeddings.word_embeddings.load_state_dict(partial_state_dict['biogpt_embedding'])
        Qformer.cls.predictions.decoder.load_state_dict(partial_state_dict['biogpt_output_proj'])

        for param in Qformer.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = False
        for param in Qformer.cls.predictions.decoder.parameters():
            param.requires_grad = False

        return Qformer, query_tokens

    def init_vision_encoder(self, 
        weight_path: str,
        model_name, 
        img_size, 
        drop_path_rate, 
        use_grad_checkpoint
    ):
        if model_name == 'prov-gigapath': # 1,134,953,984
            visual_encoder = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", 
                pretrained=False, 
                checkpoint_path=os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name]),
                dynamic_img_size=True
            )
        elif model_name == 'H-optimus-0': # 1,134,774,272
            visual_encoder = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", 
                pretrained=False, 
                checkpoint_path=os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name]),
                init_values=1e-5, 
                dynamic_img_size=True
            )
        elif model_name == 'vit_base': # 87,849,728
            visual_encoder = timm.create_model(
                "vit_base_patch32_224_clip_laion2b", 
                pretrained=False, 
                checkpoint_path=os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name]),
                dynamic_img_size=True
            )
        elif model_name == 'vit_large': # 303,966,976
            visual_encoder = timm.create_model(
                "vit_large_patch14_224_clip_laion2b", 
                pretrained=False, 
                checkpoint_path=os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name]),
                dynamic_img_size=True
            )

        visual_encoder.set_grad_checkpointing(use_grad_checkpoint)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def init_language_model(self, 
        weight_path: str,
        model_name, 
    ):
        histogpt_generate = generate
        if model_name == 'histogpt-3b':
            biogpt_config = BioGptConfig.from_pretrained(os.path.join(weight_path, WEIGHT_MAPPING_DICT['histogpt-config']))
            biogpt_weight_path = os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name])
        elif model_name == 'histogpt-1b':
            biogpt_config = BioGptConfig()
            biogpt_weight_path = os.path.join(weight_path, WEIGHT_MAPPING_DICT[model_name])
        language_model = HistoGPTForCausalLM(biogpt_config, PerceiverResamplerConfig())
        state_dict = torch.load(biogpt_weight_path, map_location='cpu', weights_only=False)
        msg = language_model.load_state_dict(state_dict)

        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

        return language_model, tokenizer, histogpt_generate, msg


    def load_from_pretrained(self, url_or_filename):
        if USE_ORIGINAL_PRETRAIN:
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            state_dict = checkpoint["model"]

        else:
            print('>>>Loading stage1 pretrained weight from:', PRETRAIN_WEIGHT_PATH)
            checkpoint = torch.load(PRETRAIN_WEIGHT_PATH, map_location="cpu", weights_only=False)

            state_dict = checkpoint["state_dict"]


        msg = self.load_state_dict(state_dict, strict=False)

        PassOrNot = True
        for got_key in msg.missing_keys:
            if 'visual_encoder' not in got_key:
                print(">>>Missing keys {}".format(got_key))
                PassOrNot = False
        if PassOrNot:
            print('>>>Successful loaded pretrained weight')
        else:
            print('>>>Something went wrong')

            logging.info("Missing keys {}".format(msg.missing_keys))
            logging.info("load checkpoint from %s" % url_or_filename)

        return msg


    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer() #???
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
