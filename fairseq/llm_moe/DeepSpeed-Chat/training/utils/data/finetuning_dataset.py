#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
part of this code is adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
"""

import copy
from typing import List

import jsonlines
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def pad_sequence(inputs: List[torch.LongTensor], padding_value: int = 0, padding_side="left", batch_first=True):
    max_len = max(seq.size(0) for seq in inputs)

    padding_inputs = []

    for seq in inputs:
        if seq.size(0) < max_len:
            if padding_side == "left":
                seq = torch.cat([torch.ones((max_len - seq.size(0),)).to(seq) * padding_value, seq])
            else:
                seq = torch.cat([seq, torch.ones((max_len - seq.size(0),)).to(seq) * padding_value])
        padding_inputs.append(seq)

    return_tensor = torch.stack(padding_inputs)

    if not batch_first:
        return_tensor = return_tensor.transpose(0, 1)

    return return_tensor


def build_input_and_label_ids(samples, tokenizer, eval_mode=True, compute_target_only=False):
    """
    从符合SFT格式的数据中构造 input_ids 和 label_ids
    """
    if "input" in samples[0]:
        INPUT_KEY = "input"
    elif "inputs" in samples[0]:
        INPUT_KEY = "inputs"
    else:
        raise ValueError("The input key should be either `input` or `inputs`")

    if "target" in samples[0]:
        TARGET_KEY = "target"
    elif "targets" in samples[0]:
        TARGET_KEY = "targets"
    else:
        TARGET_KEY = None  # no target side, all the data are in the `input`

    if "prompt" in samples[0]:
        WITH_PROMPT_TEMPLATE = True
    else:
        WITH_PROMPT_TEMPLATE = False

    def _build_source(item):
        inputs = item[INPUT_KEY]

        if isinstance(inputs, str):
            inputs = [inputs, ]

        if WITH_PROMPT_TEMPLATE:
            inputs = item['prompt'].format(*inputs)
        else:
            inputs = inputs[0]

        return inputs

    def _tokenize_fn(strings, tokenizer, add_eos=True):
        """Tokenize a list of strings."""

        tokenized_list = [
            tokenizer(
                text,
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]

        input_ids = [tokenized.input_ids for tokenized in tokenized_list]

        if add_eos and getattr(tokenizer, 'add_eos_token', False) is False:
            input_ids = [sample + [tokenizer.eos_token_id, ] for sample in input_ids]

        input_ids = [torch.tensor(sample) for sample in input_ids]
        labels = copy.deepcopy(input_ids)

        input_ids_lens = labels_lens = [
            sample_pt.ne(tokenizer.pad_token_id).sum().item() for sample_pt in input_ids
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    sources = [_build_source(item) for item in samples]

    # 是否需要将target拼入最终的样本
    # eval时target是需要我们生成的部分，所以不需要拼入
    if not eval_mode and TARGET_KEY is not None:
        targets = [item[TARGET_KEY] for item in samples]
    else:
        targets = []

    if len(targets) > 0:
        examples = [s + t for s, t in zip(sources, targets)]
    else:
        examples = sources

    examples_tokenized = _tokenize_fn(examples, tokenizer, add_eos=True if not eval_mode else False)

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    if len(targets) > 0 and compute_target_only:
        sources_tokenized = _tokenize_fn(sources, tokenizer, add_eos=False)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

    return input_ids, labels


class FinetuningDataset(Dataset):
    def __init__(self, data_path, tokenizer, compute_target_only=False, eval_mode=False):
        """Finetuning dataset format
        The file should be in jsonlines format, and each line should contain keys below:
            - input (or inputs): The input of finetuning data.
        And could contain these keys:
            - prompt: instruction template to fill the input in

        TODO:
        - [ ] We could move data preparation and padding outside this function and use MMF Dataset instead.
        """
        super(FinetuningDataset, self).__init__()

        self.tokenizer = tokenizer
        self.eval_mode = eval_mode

        with jsonlines.open(data_path) as f:
            all_data = [item for item in f]

        input_ids, labels = build_input_and_label_ids(all_data, tokenizer, eval_mode, compute_target_only)

        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
        }

    def collect_fn(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = pad_sequence(input_ids, padding_value=self.tokenizer.pad_token_id,
                                 padding_side=self.tokenizer.padding_side, batch_first=True)

        labels = pad_sequence(labels, padding_value=IGNORE_INDEX, padding_side=self.tokenizer.padding_side,
                              batch_first=True)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
