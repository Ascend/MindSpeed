# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import torch

_GLOBAL_ATTN_MASK = None


def set_attention_mask(attn_mask):
    global _GLOBAL_ATTN_MASK
    _GLOBAL_ATTN_MASK = attn_mask


def generate_attention_mask(args, compress, device):
    global _GLOBAL_ATTN_MASK
    if not args.use_flash_attn:
        warnings.warn("Flash Attention is highly recommended")
        _GLOBAL_ATTN_MASK = (
            torch.tril(
                torch.ones(
                    [args.micro_batch_size, 1, args.seq_length, args.seq_length],
                    dtype=bool,
                    device=device
                ), 
                diagonal=-(args.pre_tockens + 1)
            ) + 
            torch.triu(
                torch.ones(
                    [args.micro_batch_size, 1, args.seq_length, args.seq_length], 
                    dtype=bool, 
                    device=device
                ), 
                diagonal=args.next_tockens + 1
            )
        )
        return

    if compress:
        seq_len = 2048
    else:
        seq_len = args.seq_length
    
    _GLOBAL_ATTN_MASK = torch.triu(
        torch.ones(
            (seq_len, seq_len), 
            device=device, 
            dtype=torch.bool
        ), 
        diagonal=1
    )


def get_attention_mask(args):
    global _GLOBAL_ATTN_MASK
    if _GLOBAL_ATTN_MASK is not None:
        return _GLOBAL_ATTN_MASK

    device = 'npu'
    compress = True

    if getattr(args, "multi_head_latent_attention", False):
        compress = False

    generate_attention_mask(args, compress, device)

    return _GLOBAL_ATTN_MASK