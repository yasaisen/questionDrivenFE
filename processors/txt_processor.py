"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 This file includes portions of code from Salesforce (BSD-3-Clause):
   Copyright (c) 2022, salesforce.com, inc.
   SPDX-License-Identifier: BSD-3-Clause
   See: https://opensource.org/licenses/BSD-3-Clause

 last modified in 2505192050
"""

import re


class TxtProcessor():
    def __init__(self, 
        prompt: str = "", 
        max_words: int = None,
    ):
        self.prompt = prompt
        self.max_words = max_words

    def pre_caption(self, 
        caption: str
    ):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        caption_words = caption.split(" ")
        if self.max_words is not None:
            if len(caption_words) > self.max_words:
                caption = " ".join(caption_words[: self.max_words])

        return caption

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, 
        cfg
    ):
        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", None)

        processor = cls(
            prompt=prompt, 
            max_words=max_words
        )

        return processor










