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

from utils.data.pretraining_dataset import PretrainingDataset
from utils.data.indexed_dataset import make_dataset as make_indexed_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, get_optimizer_grouped_parameters, create_moe_param_groups, \
    load_hf_tokenizer_local, pretty_print, compile_dependencies
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, unfuse_lora_weight_from_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model
import numpy as np
import deepspeed.comm as ds_comm
from deepspeed.utils import log_dist


# from deepspeed.comm import all_reduce

def save_hf_format(model, tokenizer, step, args, sub_folder="hf_model"):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config_"+str(step)+".json"
    WEIGHTS_NAME = "pytorch_model_"+str(step)+".bin"
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


def build_pretrain_dataset(data_path, seq_length: int = 2048, seed: int = 1234):
    """
    data_path: (str) Path of indexed dataset.
    """

    # 1. get indexed dataset

    indexed_dataset = make_indexed_dataset(data_path, impl='mmap', skip_warmup=True)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    documents = np.arange(start=0, stop=total_num_of_documents, dtype=np.int32)

    exp_name = f"{seq_length}_{seed}"

    training_dataset = PretrainingDataset(
        data_prefix=data_path,
        exp_name=exp_name,
        documents=documents,
        indexed_dataset=indexed_dataset,
        seq_length=seq_length,
        seed=seed
    )

    return training_dataset


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
        "--tokenizer_path",
        type=str,
        help=
        "Path to tokenizer.",
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
        default=2048,
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
                        default="",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    # training settings
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
    
    parser.add_argument(
        '--local_rank', type=int, default=None,
    )
    parser.add_argument('--is_moe',
                        action='store_true')
    parser.add_argument('--gate_type', type=str, default=None,
                       choices=["fix_weight", "continual_gate", "discrete_gate"])
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
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

    # compile data dependencies on-the-fly
    compile_dependencies()

    monitor_dir = os.path.join(args.output_dir, 'logs')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')

    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(args=args,
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

    # tokenizer = load_hf_tokenizer_local(args.model_name_or_path, model_max_length=args.max_seq_len,
    #                                     tie_eos_and_pad=True)

    # model = create_hf_model(AutoModelForCausalLM,
    #                         args.model_name_or_path,
    #                         tokenizer,
    #                         ds_config,
    #                         disable_dropout=args.disable_dropout)
    # from moe_model.configuration_llama import LlamaConfig
    # config = LlamaConfig(
    #     vocab_size=32000,
    #     hidden_size=256,
    #     intermediate_size=512,
    #     num_hidden_layers=3,
    #     num_attention_heads=32,
    #     hidden_act="silu",
    #     max_position_embeddings=2048,
    #     initializer_range=0.02,
    #     rms_norm_eps=1e-6,
    #     use_cache=True,
    #     pad_token_id=0,
    #     bos_token_id=1,
    #     eos_token_id=2,
    #     tie_word_embeddings=False,
    #     moe_expert_count=2,
    #     sparse_step=3,
    #     ep_size=2,
    # )
    # from moe_model.modeling_llama import LlamaForCausalLM
    # model = LlamaForCausalLM(config)
    from moe_model.tokenization_qwen import QWenTokenizer
    from moe_model.configuration_qwen import QWenConfig
    from moe_model.modeling_qwen import QWenLMHeadModel
    if args.is_moe:
        
        tokenizer = QWenTokenizer.from_pretrained(args.tokenizer_path)
        # qwen_1b8
        moe_config = QWenConfig(vocab_size=151936,
                        hidden_size=2048,
                        num_hidden_layers=24,
                        num_attention_heads=16,
                        emb_dropout_prob=0.1,
                        attn_dropout_prob=0.0,
                        layer_norm_epsilon=1e-5,
                        initializer_range=0.02,
                        max_position_embeddings=2048,
                        scale_attn_weights=True,
                        use_cache=True,
                        bf16=True,
                        fp16=False,
                        fp32=False,
                        kv_channels=128,
                        rotary_pct=1.0,
                        rotary_emb_base=10000,
                        use_dynamic_ntk=False,
                        use_logn_attn=False,
                        use_flash_attn=False,
                        intermediate_size=11008,
                        no_bias=True,
                        tie_word_embeddings=False,
                        seq_length=2048,
                        moe_expert_count=2,
                        sparse_step=1,
                        ep_size=2,
                        share_weight=0.5,
                        gate_type=args.gate_type,
                        gate_loss_weight=0.0
        )

        #qwen-7b
        # from moe_model.configuration_qwen_7b import QWenConfig
        # from moe_model.modeling_qwen_7b import QWenLMHeadModel
        # moe_config = QWenConfig(vocab_size=151936,
        #                 hidden_size=4096,
        #                 num_hidden_layers=32,
        #                 num_attention_heads=32,
        #                 emb_dropout_prob=0.0,
        #                 attn_dropout_prob=0.0,
        #                 layer_norm_epsilon=1e-06,
        #                 initializer_range=0.02,
        #                 max_position_embeddings=6144,
        #                 scale_attn_weights=True,
        #                 use_cache=True,
        #                 bf16=True,
        #                 fp16=False,
        #                 fp32=False,
        #                 kv_channels=128,
        #                 rotary_pct=1.0,
        #                 rotary_emb_base=10000,
        #                 use_dynamic_ntk=True,
        #                 use_logn_attn=True,
        #                 use_flash_attn=False,
        #                 intermediate_size=22016,
        #                 no_bias=True,
        #                 tie_word_embeddings=False,
        #                 seq_length=2048,
        #                 moe_expert_count=4,
        #                 sparse_step=1,
        #                 ep_size=4,
        #                 share_weight=0.5,
        #                 gate_type=args.gate_type
        # )
        
        moe_config.fp32 = False
        moe_config.bf16 = True # training on V100
        # try:
        #     dense_config = QWenConfig.from_pretrained(args.model_name_or_path)
        #     dense_config.fp32 = False
        #     dense_config.bf16 = True
        #     dense_model = QWenLMHeadModel.from_pretrained(args.model_name_or_path,
        #                                 from_tf=bool(".ckpt" in args.model_name_or_path),
        #                                 config=dense_config)
        #     dense_model_state_dict = dense_model.state_dict()
        # except:
        #     print(f"fail to execuate from_pretrained, try to load model file from {args.model_name_or_path}/mp_rank_00_model_states.pt ...")
        #     dense_model_state_dict = torch.load(f"{args.model_name_or_path}/mp_rank_00_model_states.pt", map_location="cpu")["module"]
        model = QWenLMHeadModel(moe_config)
        model.load_dense_model(args.model_name_or_path)

        from fix_parameters import freeze_non_moe_parameters
        freeze_non_moe_parameters(model)
    else:
        tokenizer = QWenTokenizer.from_pretrained(args.model_name_or_path)
        config = QWenConfig.from_pretrained(args.model_name_or_path)
        config.bf16 = True
        config.use_flash_attn = False
        model = QWenLMHeadModel.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config)

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
    
    num_params = sum(param.numel() for param in model.parameters())
    moe_params = sum(param.numel() for name, param in model.named_parameters() if "moe" in name)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('total params', num_params)
    print('local MoE params', moe_params)
    print('trainable params', trainable_params)

    # all_dataset =
    # Prepare the data
    train_phase = 1

    # train_dataset = FinetuningDataset(
    #     data_path=args.data_path,
    #     tokenizer=tokenizer,
    #     compute_target_only=True
    # )

    train_dataset = build_pretrain_dataset(
        data_path=args.data_path,
        seq_length=args.max_seq_len,
        seed=args.seed
    )

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_dataset.collect_fn,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    # Split weights in two groups, one with weight decay and the other not.
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    #     model, args.weight_decay)
    optimizer_moe_grouped_parameters = create_moe_param_groups(model.parameters())

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    optimizer = AdamOptimizer(optimizer_moe_grouped_parameters,
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

    # If exist checkpoint, resume from training

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # # Train!
    # print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):

        num_micro_batches_per_epoch = len(train_dataloader)

        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {num_micro_batches_per_epoch}"
        )

        model.train()

        losses = []
        moe_gate_losses = []
        moe_metadata = {}
        progress_bar = tqdm(range(num_micro_batches_per_epoch), disable=local_rank != 0)

        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            # print(batch["input_ids"].shape)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.is_moe:
                #gather moe metadata
                for metadata_i in outputs.moe_metadata:
                    if metadata_i is not None:
                        for key in metadata_i:
                            if key not in moe_metadata:
                                moe_metadata[key] = []
                            moe_metadata[key].append(metadata_i[key])

                moe_gate_loss = outputs.moe_gate_loss
                final_loss = loss + moe_config.gate_loss_weight * moe_gate_loss
                model.backward(final_loss)
            else:
                model.backward(loss)
            model.step()
            losses.append(loss)
            if args.is_moe:
                moe_gate_losses.append(moe_gate_loss)
            if (step + 1) % args.steps_per_print == 0:
                progress_bar.update(args.steps_per_print)

            if model.is_gradient_accumulation_boundary():
                train_loss_this_step = sum(losses) / len(losses)
                moe_gate_losses_this_step = 0.0
                if args.is_moe:
                    moe_gate_losses_this_step = sum(moe_gate_losses) / len(moe_gate_losses)

                # 1. Logging
                #if model.monitor.enabled:
                if model.global_rank == 0:
                    global_samples = model.global_samples
                    global_steps = model.global_steps + 1
                    summary_events = [(
                        f"Train/Samples/real_training_loss",
                        train_loss_this_step, global_samples
                    ), ]

                    model.monitor.write_events(summary_events)
                    #  MoE_Gate_Loss: {moe_gate_losses_this_step:.3f} Gate_Loss_Weight: {moe_config.gate_loss_weight}
                    log_dist(f"GlobalSteps: {global_steps} Loss: {train_loss_this_step:.3f} MoE_Gate_Loss: {moe_gate_losses_this_step:.3f} Gate_Loss_Weight: {moe_config.gate_loss_weight}" , ranks=[0])
                    if args.is_moe:
                        # print metadata
                        print_str = f"MoE Metadata: "
                        for key in moe_metadata:
                            value = sum(moe_metadata[key]) / len(moe_metadata[key])
                            print_str += f" | {key}: {value:.3f}"
                        log_dist(print_str , ranks=[0])
                losses = []
                moe_gate_losses = []
                moe_metadata = {}

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

                    if args.lora_dim > 0:
                        lora_fused_model = convert_lora_to_linear_layer(model)
                        lora_fused_model.save_checkpoint(checkpoint_dir, ckpt_id, client_state=client_sd)
                        model = unfuse_lora_weight_from_linear_layer(lora_fused_model)
                    else:
                        model.save_checkpoint(checkpoint_dir, ckpt_id, client_state=client_sd)


        model.tput_timer.update_epoch_count()

    # if args.output_dir is not None:
    #     print_rank_0('saving the final model ...')
    #     model = convert_lora_to_linear_layer(model)
    #
    #     if args.global_rank == 0:
    #         save_hf_format(model, tokenizer, args)
    #
    #     if args.zero_stage == 3:
    #         # For zero stage 3, each gpu only has a part of the model, so we need a special save function
    #         save_zero_three_model(model,
    #                               args.global_rank,
    #                               args.output_dir,
    #                               zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
