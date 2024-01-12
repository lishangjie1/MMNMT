# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
# DeepSpeed Team
import time
from typing import List

import deepspeed
import jsonlines
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import set_seed, AutoTokenizer

from .dist_utils import is_main_process, wait_for_everyone

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def merge_jsonl(output_path, input_paths: List[str]):
    with jsonlines.open(output_path, mode='w') as writer:
        for input_path in input_paths:
            with jsonlines.open(input_path) as reader:
                for obj in reader:
                    writer.write(obj)


def print_rank_0(msg='', **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, **kwargs)


def pretty_print(obj):
    def _pretty_print(obj, indent=0):
        """\n    Recursively pretty prints a nested dictionary or list object.\n    """
        if isinstance(obj, dict):
            print_rank_0('{')
            for key, value in obj.items():
                print_rank_0(' ' * (indent + 4) + str(key) + ': ', end='')
                _pretty_print(value, indent + 4)
                if key != list(obj.keys())[-1]:
                    print_rank_0(',')
            print_rank_0()
            print_rank_0(' ' * indent + '}', end='')
        elif isinstance(obj, list):
            print_rank_0('[', end='')
            for item in obj:
                # print(' ' * (indent + 4), end='')
                _pretty_print(item, indent + 4)
                if item != obj[-1]:
                    print_rank_0(',', end=' ')
            print_rank_0(']', end='')
        else:
            print_rank_0(obj, end='')

    _pretty_print(obj, indent=0)
    print_rank_0()


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  fast_tokenizer=True)
    return tokenizer


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_hf_tokenizer_local(model_name_or_path, model_max_length=None, tie_eos_and_pad=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer._pad_token is None and tie_eos_and_pad:
        tokenizer._pad_token = tokenizer._unk_token

    return tokenizer


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]

    return optimizer_grouped_parameters

def create_moe_param_groups(model_parameters):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {
        'params': [p for p in model_parameters if p.requires_grad],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def compile_dependencies():
    if is_main_process():
        start_time = time.time()
        print_rank_0('> compiling dataset index builder ...')
        from utils.data.data_utils import compile_helper
        compile_helper()
        print_rank_0('>>> done with dataset index builder. Compilation time: {:.3f} '
                     'seconds'.format(time.time() - start_time), flush=True)

    wait_for_everyone()
