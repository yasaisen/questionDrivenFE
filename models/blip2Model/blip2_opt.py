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

import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..blip2Model.blip2 import Blip2Base, disabled_train
from ...common.utils import log_print, get_trainable_params, highlight, highlight_show


class Blip2OPT(Blip2Base):
    def __init__(self, 
        weight_path: str,

        vit_model: str = "prov-gigapath",
        lm_model: str = "histogpt-3b",

        img_size: int = None,
        freeze_vit: bool = True,
        cross_attention_freq: int = 2,
        num_query_token: int = 32,
        max_txt_len: int = 32,
        prompt: str = "",

        drop_path_rate: int = 0,
        use_grad_checkpoint: bool = False,
        apply_lemmatizer: bool = False,
    ):
        super().__init__()
        log_print(f"Building...", head=True)
        self.weight_path = weight_path

        # self.tokenizer = self.init_tokenizer()

        log_print(f"Loading visual_encoder({vit_model})...", newline=False)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            weight_path=self.weight_path,
            model_name=vit_model, 
            img_size=img_size, 
            drop_path_rate=drop_path_rate, 
            use_grad_checkpoint=use_grad_checkpoint, 
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train # ???
            logging.info("freeze vision encoder")
        print("...Done")
        log_print(f"visual_encoder trainable params: {get_trainable_params(self.visual_encoder)}")


        log_print(f"Loading Qformer...", newline=False)
        self.Qformer, self.query_tokens = self.init_Qformer(
            weight_path=self.weight_path,
            num_query_token=num_query_token, 
            vision_width=self.visual_encoder.num_features,
            cross_attention_freq=cross_attention_freq
        )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        print("...Done")
        log_print(f"Qformer trainable params: {get_trainable_params(self.Qformer)}")


        log_print(f"Loading language_model({lm_model})...", newline=False)
        self.lm_model, self.lm_tokenizer, self.histogpt_generate, msg = self.init_language_model(
            weight_path=self.weight_path,
            model_name=lm_model, 
        )
        print(msg, end="")
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.lm_tokenizer(
            "\n" # , add_special_tokens=False
        ).input_ids[0]
        print("...Done")
        log_print(f"lm_model trainable params: {get_trainable_params(self.lm_model)}")


        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.lm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.lm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None     

        log_print(f"ln_vision trainable params: {get_trainable_params(self.ln_vision)}")
        log_print(f"opt_proj trainable params: {get_trainable_params(self.opt_proj)}")
        log_print(f"query_tokens trainable params: {sum(p.numel() for p in self.query_tokens if p.requires_grad)}")
        
        log_print(f"model trainable params: {get_trainable_params(self)}")
        log_print(f"...Done\n")


    def forward(self, 
        samples
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder.forward_features(image)) # [bsz patches_len + cls 1536]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )


        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.lm_tokenizer.padding_side = "right"


        text = [q + " " + a + "\n" for q, a in zip(samples["question"], samples["text_input"])]
        question_tokens = self.lm_tokenizer(
            [q + " " for q in samples["question"]], 
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        
        lm_tokens = self.lm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = lm_tokens.input_ids.masked_fill(
            lm_tokens.input_ids == self.lm_tokenizer.pad_token_id, -100
        )
        
        for i in range(targets.size(0)):
            question_length = (question_tokens.attention_mask[i] == 1).sum().item()
            targets[i, :question_length] = -100
        
        if self.prompt:
            targets[:, : self.prompt_length] = -100

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)


        inputs_embeds = self.lm_model.histogpt.embed_tokens(lm_tokens.input_ids)


        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, lm_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder.forward_features(image)) # [bsz patches_len + cls 1536]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )


            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            lm_tokens = self.lm_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            inputs_embeds = self.lm_model.get_input_embeddings()(lm_tokens.input_ids)
            attention_mask = lm_tokens.attention_mask


            inputs_embeds = torch.cat([inputs_opt, inputs_embeds],dim=1)
            attention_mask = torch.cat([atts_opt, attention_mask], dim=1)


            outputs = self.histogpt_generate(
                model=self.lm_model,
                prompt=prompt,
                # image=features,
                length=256,
                top_k=40,
                top_p=0.95,
                temp=0.7,
                device=image.device
            )


            output_text = self.lm_tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text

        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.lm_tokenizer.padding_side = "left"
            lm_tokens = self.lm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, lm_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.lm_model.get_input_embeddings()(lm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.lm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.lm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
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
        

    @classmethod
    def from_config(cls, cfg):
        weight_path = cfg.get("weight_path")

        vit_model = cfg.get("vit_model", "prov-gigapath")
        lm_model = cfg.get("lm_model", "histogpt-3b")

        img_size = cfg.get("image_size", None)
        freeze_vit = cfg.get("freeze_vit", True)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        num_query_token = cfg.get("num_query_token")
        max_txt_len = cfg.get("max_txt_len", 32)
        prompt = cfg.get("prompt", "")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            weight_path=weight_path,

            vit_model=vit_model,
            lm_model=lm_model,

            img_size=img_size,
            freeze_vit=freeze_vit,
            cross_attention_freq=cross_attention_freq,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            prompt=prompt,

            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            apply_lemmatizer=apply_lemmatizer,
        )
        # model.load_checkpoint_from_config(cfg)

        return model












