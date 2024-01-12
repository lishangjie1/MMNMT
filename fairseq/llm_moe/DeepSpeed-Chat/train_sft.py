#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils.data.finetuning_dataset import FinetuningDataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_optimizer_grouped_parameters, \
    save_zero_three_model, load_hf_tokenizer_local, pretty_print
from utils.dist_utils import is_main_process
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model
import deepspeed.comm as ds_comm
from deepspeed.utils import log_dist


def all_reduce_mean(tensor):
    with torch.no_grad():
        reduced_tensor = tensor.clone()
        ds_comm.all_reduce(reduced_tensor)
        return reduced_tensor / ds_comm.get_world_size()


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')

    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `6,2,2`'
                             'will use 60% of data for phase 1, 20% for phase 2'
                             'and 20% for phase 3.')
    # parser.add_argument(
    #     '--sft_only_data_path',
    #     nargs='*',
    #     default=[],
    #     help='Path to the dataset for only using in SFT phase.')
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
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")

    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')

    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    parser.add_argument('--steps_per_print', type=int, default=100,
                        help='Number of steps between printing training logs')

    parser.add_argument('--steps_per_save', type=int, default=1000, help='Number of steps between saving model weights')

    # wandb settings
    parser.add_argument(
        '--wandb_enable', action='store_true', help='Enable wandb logging'
    )

    parser.add_argument(
        '--wandb_key', type=str, default=None, help='wandb key',
    )

    parser.add_argument(
        '--wandb_team', type=str, default='default_team', help='wandb team',
    )

    parser.add_argument(
        '--wandb_group', type=str, default='default_group', help='wandb group',
    )

    parser.add_argument(
        '--wandb_project', type=str, default='default_project', help='wandb project',
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # # Validate settings
    # if args.gradient_checkpointing and args.lora_dim > 0:
    #     assert (
    #         not args.only_optimize_lora
    #     ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def all_reduce_mean(tensor):
    with torch.no_grad():
        reduced_tensor = tensor.clone()
        ds_comm.all_reduce(reduced_tensor)
        return reduced_tensor / ds_comm.get_world_size()


def main():
    args = parse_args()

    if args.wandb_enable:
        import wandb
    else:
        wandb = None

    if wandb is not None:
        assert args.wandb_key is not None, "Please provide wandb key"
        wandb.login(anonymous='allow', key=args.wandb_key)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()

    # if args.local_rank == -1:
    #     device = torch.device("cuda")
    # else:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    monitor_dir = os.path.join(args.output_dir, 'logs')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')

    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        args=args,
        offload=args.offload,
        stage=args.zero_stage,
        monitor_dir=monitor_dir,
        steps_per_print=args.steps_per_print,
    )

    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    pretty_print(ds_config)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer_local(args.model_name_or_path, model_max_length=args.max_seq_len,
                                        tie_eos_and_pad=True)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # all_dataset =
    # Prepare the data
    train_phase = 1

    train_dataset = FinetuningDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        compute_target_only=True
    )
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    # train_dataset, eval_dataset = create_prompt_dataset(
    #     args.local_rank,
    #     args.data_path,
    #     args.data_split,
    #     args.data_output_path,
    #     train_phase,
    #     args.seed,
    #     tokenizer,
    #     args.max_seq_len,
    #     sft_only_data_path=args.sft_only_data_path)

    # TODO:
    # Here we should return a dataset with fields:
    # input_ids:
    # attention_masks:
    # labels: (In stage one they use input_ids as well)

    # DataLoaders creation:
    # if args.local_rank == -1:
    #     train_sampler = RandomSampler(train_dataset)
    #     eval_sampler = SequentialSampler(eval_dataset)
    # else:
    #     train_sampler = DistributedSampler(train_dataset)
    #     eval_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_dataset.collect_fn,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    # eval_dataloader = DataLoader(eval_dataset,
    #                              collate_fn=default_data_collator,
    #                              sampler=eval_sampler,
    #                              batch_size=args.per_device_eval_batch_size)

    # def evaluation(model, eval_dataloader):
    #     model.eval()
    #     losses = 0
    #     for step, batch in enumerate(eval_dataloader):
    #         batch = to_device(batch, device)
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #
    #         loss = outputs.loss
    #         losses += loss.float()
    #     losses = losses / (step + 1)
    #     try:
    #         perplexity = torch.exp(losses)
    #     except OverflowError:
    #         perplexity = float("inf")
    #     try:
    #         perplexity = get_all_reduce_mean(perplexity).item()
    #     except:
    #         pass
    #     return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    # Note:
    # optimizer and lr_scheduler are not necessarily specified in the config file.
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # # Train!
    # print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    num_micro_batches_per_epoch = len(train_dataloader)
    progress_bar = tqdm(range(num_micro_batches_per_epoch * args.num_train_epochs), disable=local_rank != 0)

    for epoch in range(args.num_train_epochs):

        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {num_micro_batches_per_epoch}"
        )

        model.train()

        losses = []

        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)

            # reduced CE Loss per micro_batch_size per rank
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            losses.append(loss.item())

            model.backward(loss)
            model.step()

            if (step + 1) % args.steps_per_print == 0:
                progress_bar.update(args.steps_per_print)

            if model.is_gradient_accumulation_boundary():
                train_loss_this_step = sum(losses) / len(losses)

                # 1. Logging
                if model.monitor.enabled:
                    if model.global_rank == 0:
                        global_samples = model.global_samples
                        global_steps = model.global_steps + 1
                        summary_events = [(
                            f"Train/Samples/real_training_loss",
                            train_loss_this_step, global_samples
                        ), ]

                        model.monitor.write_events(summary_events)

                        log_dist(f"GlobalSteps: {global_steps} Loss: {train_loss_this_step:.3f}", ranks=[0])

                losses = []

                # 2. Checkpoint saving
                # Here we use global samples
                if (model.global_steps + 1) % args.steps_per_save == 0:

                    # 1. reduce mean loss
                    reduced_loss = all_reduce_mean(torch.tensor(train_loss_this_step).to(device)).item()

                    # 2. saving checkpoint
                    # 3. wait_for_everyone
                    client_sd = dict()

                    # Saving these stats in order to resume training
                    client_sd['global_steps'] = model.global_steps
                    client_sd['micro_steps'] = model.micro_steps
                    client_sd['global_samples'] = model.global_samples
                    client_sd['num_epoches'] = epoch

                    ckpt_id = f"epoch{epoch}_iter{model.global_steps + 1}_loss{reduced_loss:.3f}"
                    model.save_checkpoint(checkpoint_dir, ckpt_id, client_state=client_sd)

                # 3. Maybe: Evaluation

        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:

        print_rank_0('saving the final model ...')
        model = convert_lora_to_linear_layer(model)

        if is_main_process():
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
