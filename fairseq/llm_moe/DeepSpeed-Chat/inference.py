import argparse
import copy
import os
import math
import sys

import jsonlines
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))

import torch
from transformers import (
    AutoModelForCausalLM,
)

import deepspeed

from utils.data.finetuning_dataset import build_input_and_label_ids, pad_sequence
from utils.utils import to_device, set_random_seed, load_hf_tokenizer_local, pretty_print, merge_jsonl
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from utils.data.data_utils import batchify, shard_data
from utils.dist_utils import wait_for_everyone, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')

    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument("--model_type",
                        type=str,
                        help="The model architecture to inference.",
                        default=None,
                        choices=["llama", "bloom", "gpt2"])

    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')

    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")

    # generate arguments

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling; use greedy decoding if this option is disabled.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="The maximum number of new tokens to generate. max_seq_len will be overridden if this is set.",
    )

    # deepspeed-inference config
    parser.add_argument(
        "--mp_size",
        type=int,
        default=1,
        help="The model parallel size."
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="The data type of the model."
    )

    # data sharding

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards to split the data into."
    )

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="The shard id of the current process."
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_injection_policy(model_type):
    if model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        return {LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
    elif model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        return {GPT2Block: {'attn.c_proj', 'mlp.c_proj'}}
    else:
        raise NotImplementedError


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    tokenizer = load_hf_tokenizer_local(args.model_name_or_path, model_max_length=args.max_seq_len,
                                        tie_eos_and_pad=True)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            None,
                            disable_dropout=True, ds_inference=True)

    with jsonlines.open(args.data_path, 'r') as reader:
        all_data = [item for item in reader]

    # 1. 是否进行了数据分片
    # 2. 使用ZeRO Stage-3时是否使用了多GPU

    if args.num_shards > 1:
        all_data = shard_data(all_data, num_shards=args.num_shards, shard_id=args.shard_id)

    batches = list(batchify(all_data, batch_size=args.per_device_batch_size))

    if args.model_type is not None:
        injection_policy = get_injection_policy(args.model_type)

        engine = deepspeed.init_inference(
            model, mp_size=args.mp_size,
            dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float16,
            injection_policy=injection_policy
        )
    else:

        engine = deepspeed.init_inference(model, mp_size=args.mp_size,
                                          dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float16)

    engine.module.eval()
    model = engine.module

    num_micro_batches_per_epoch = len(batches)

    progress_bar = tqdm(range(num_micro_batches_per_epoch), disable=local_rank != 0)

    generate_kwargs = dict(max_length=args.max_seq_len, do_sample=args.do_sample, temperature=args.temperature,
                           top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    if args.max_new_tokens > 0:
        generate_kwargs['max_new_tokens'] = args.max_new_tokens

    output_name = f"{args.data_output_path}.{args.shard_id}"

    results = []

    for step, raw_batch in enumerate(batches):

        input_ids, _ = build_input_and_label_ids(samples=raw_batch, tokenizer=tokenizer, eval_mode=True)

        batch = {
            "input_ids": pad_sequence(input_ids, padding_value=tokenizer.pad_token_id)
        }

        batch['attention_mask'] = batch['input_ids'].ne(tokenizer.pad_token_id)

        batch = to_device(batch, device)

        outputs = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'], **generate_kwargs)
        raw_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if is_main_process():
            print(batch['input_ids'])
            print(outputs)
            print(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False))
            print(raw_outputs)

        wait_for_everyone()
        exit(1)

        output_samples = []

        for raw_sample, raw_output in zip(raw_batch, raw_outputs):
            pure_output = raw_output.replace(raw_sample['input'], '').strip()
            output_sample = copy.deepcopy(raw_sample)
            output_sample['prediction'] = pure_output
            output_samples.append(output_sample)

        progress_bar.update(1)
        results.extend(output_samples)

    if is_main_process():
        writer = jsonlines.open(output_name, flush=True, mode='w')
        writer.write_all(results)

    wait_for_everyone()


if __name__ == "__main__":
    main()
