#!/usr/bin/env python
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import argparse
import os
from collections import OrderedDict

import torch
import mindspeed.megatron_adaptor
from mindspeed.core.distributed.layerzero.state.scripts import layerzero_checkpointer
from mindspeed.core.distributed.layerzero.state.scripts.layerzero_checkpointer import LayerzeroCheckpoint
ARGS_KEY = 'args'

FINAL_LAYER_NORM_KEY = 'final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None,
                        type=str, help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder', default=None,
                        type=str, help='Output Megatron checkpoint folder')
    parser.add_argument('--prefix', default="predictor",
                        help='Model prefix used in Layerzero')
    parser.add_argument('--target_tp', default=1,
                        type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1,
                        type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true',
                        help='Convert for release purpose, reset some (progress) counters.')
    parser.add_argument('--ema_model', action='store_true',
                        help='Convert Ema models')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(os.path.join(
                base_folder, iter_folder, ckpt_path))

    return path_list


def _save_checkpoint(file_path, chkpt_sd):
    ckpt_dir, _ = os.path.split(file_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def _create_rank_checkpoint(zero_checkpoint, tp_index, pp_index, tp_degree, pp_degree, for_release=False):
    checkpoint_sd = OrderedDict()
    checkpoint_sd[layerzero_checkpointer.MODEL_SD_KEY] = zero_checkpoint.create_rank_checkpoint(
        tp_index, pp_index, tp_degree, pp_degree)
    iteration = zero_checkpoint.get_iteration()
    checkpoint_sd[ITERATION_KEY] = iteration
    checkpoint_sd[ARGS_KEY] = zero_checkpoint.get_args()
    # Adjust specific fields
    checkpoint_sd[ARGS_KEY].tensor_model_parallel_size = tp_degree
    checkpoint_sd[ARGS_KEY].pipeline_model_parallel_size = pp_degree
    if for_release:
        checkpoint_sd[ARGS_KEY].consumed_train_samples = 0
        checkpoint_sd[ARGS_KEY].consumed_valid_samples = 0
    checkpoint_sd[CHECKPOINT_VERSION_KEY] = CHECKPOINT_VERSION_VALUE
    return checkpoint_sd


def _create_latest_file(base_folder, iteration):
    file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(iteration))


def main():
    print(f'Convert Layerzero dist Checkpoint to a SINGLE Megatron Checkpoint')

    args = parse_arguments()
    print(f'Converting Layerzero checkpoint in {args.input_folder} to Megatron checkpoint in {args.output_folder}')
    if args.ema_model:
        from mindspeed.core.distributed.layerzero.state.scripts.layerzero_checkpointer import set_ema_model
        set_ema_model()
    if args.prefix is not None:
        from mindspeed.core.distributed.layerzero.state.scripts.layerzero_checkpointer import remove_model_prefix
        remove_model_prefix(args.prefix)
        
    lz_checkpoint = LayerzeroCheckpoint(args.input_folder)
    iteration = lz_checkpoint.get_iteration()
    _create_latest_file(args.output_folder, iteration)
    checkpoint_paths = _create_checkpoint_paths(
        args.output_folder, iteration, args.target_tp, args.target_pp)
    for i in range(0, args.target_tp):
        for j in range(0, args.target_pp):
            sd = _create_rank_checkpoint(
                lz_checkpoint, i, j, args.target_tp, args.target_pp, args.for_release)
            _save_checkpoint(checkpoint_paths[i][j], sd)


if __name__ == "__main__":
    main()
