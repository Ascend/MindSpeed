# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Seq1F1B KV Cache Management for Sequence Parallel Training

This module implements KV (Key-Value) cache management for 1F1B sequence parallel training.
It provides span information tracking and efficient cache operations to handle sequence
splits across pipeline stages, enabling gradient checkpointing and recomputation while
maintaining training consistency.

Key Components:
- SpanInfo: Data structure for tracking sequence chunk metadata
- Seq1F1BCache: Autograd function for KV cache management across spans
- Cache reordering and slicing utilities for sequence splits
"""

from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class SpanInfo:
    """Sequence chunk information class for managing sequence chunk processing in pipeline parallelism.
    
    Attributes:
        span_idx: Current chunk index
        span_num: Total number of chunks
        span_start: Start position of current chunk in the sequence
        span_end: End position of current chunk in the sequence
        kv_cache: KV cache pool for storing intermediate results
        seq_dim: Sequence dimension in tensors
        micro_batch_idx: Micro-batch index for pipeline parallelism
        actual_seq_qlen: Actual query sequence lengths list for FA
        actual_seq_kvlen: Actual key-value sequence lengths list for FA
    """
    span_idx: int
    span_num: int
    span_start: int
    span_end: int
    kv_cache: Dict
    seq_dim: int
    micro_batch_idx: int
    actual_seq_qlen: List
    actual_seq_kvlen: List
    
    def __str__(self):
        """Return string representation of span information."""
        str_info = f'SpanInfo: span_idx={self.span_idx},'
        str_info += f'span_num={self.span_num},'
        str_info += f'span_start={self.span_start},'
        str_info += f'span_end={self.span_end},'
        str_info += f'seq_dim={self.seq_dim},'
        str_info += f'micro_batch_idx={self.micro_batch_idx},'
        str_info += f'actual_seq_qlen={self.actual_seq_qlen},'
        str_info += f'actual_seq_kvlen={self.actual_seq_kvlen},'
        return str_info

    @property
    def last_span_idx(self):
        """Check if current chunk is the last chunk in the sequence."""
        return self.span_idx == self.span_num - 1


def slice_tensor(tensor: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    """Slice tensor along specified dimension.
    
    Args:
        tensor: Input tensor to slice
        start: Start index for slicing (None for beginning)
        end: End index for slicing (None for end)
        dim: Dimension along which to slice
        
    Returns:
        Sliced tensor
    """
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


def reorder(kv: torch.Tensor, kv_lens: List, sp_num: int, seq_dim: int):
    """Reorder KV cache from interleaved pattern to contiguous pattern in Multi-Head Latent Attention.
    
    Example: Transform "abab" pattern to "aabb" pattern for efficient processing.
    
    Args:
        kv: Input key-value tensor with interleaved pattern
        kv_lens: List of segment lengths for each sequence part
        sp_num: Number of splits/partitions
        seq_dim: Sequence dimension in the tensor
        
    Returns:
        Reordered tensor with contiguous pattern
    """
    kv_shape = list(kv.shape)  # abab
    seq_len = kv_shape[seq_dim] // sp_num
    size = kv_shape[:seq_dim] + [sp_num, seq_len] + kv_shape[seq_dim + 1:]
    kv_splits = kv.reshape(size).split(kv_lens, dim=seq_dim + 1)  # abab->[ab,ab]->[[a],[a]] [[b],[b]]
    size_ = kv_shape[:seq_dim] + [-1] + kv_shape[seq_dim + 1:]
    kv = torch.cat([t.view(size_) for t in kv_splits]).contiguous()  # [[a],[a]] [[b],[b]]->[aa,bb]->aabb
    return kv


class Seq1F1BCache(torch.autograd.Function):
    """Seq1F1B Cache Management.
    
    Manages KV cache for sequence processing in pipeline parallel training.
    Supports both forward propagation and gradient computation during backward pass.
    """

    @staticmethod
    def forward(ctx, input_tensor, input_name, span_info):
        """Forward pass for sequence cache management.
        
        Args:
            ctx: Context object for storing information for backward pass
            input_tensor: Input tensor for current chunk
            input_name: Identifier for the cache (e.g., 'key', 'value')
            span_info: Span information object containing chunk metadata
            
        Returns:
            Processed tensor with appropriate caching
        """
        ctx.span_info = span_info
        ctx.kv_cache_pool = span_info.kv_cache
        seq_dim = span_info.seq_dim
        kv_cache_pool = span_info.kv_cache
        span_idx = span_info.span_idx
        last_idx = span_info.last_span_idx

        # Set cache keys
        cache_idx = input_name
        cache_lens_idx = input_name + '_lens'

        # Initialize cache pool if not exists
        if input_name not in kv_cache_pool:
            kv_cache_pool[cache_idx] = torch.tensor([], device=input_tensor.device, dtype=input_tensor.dtype)
            kv_cache_pool[cache_lens_idx] = []
        cache_num = len(kv_cache_pool[cache_lens_idx])
        input_tensor = input_tensor.detach()
        input_len = input_tensor.shape[seq_dim]

        # Determine if this is the first forward pass for this chunk
        is_first_forward = span_idx == cache_num

        if is_first_forward:
            # First forward pass
            kv_cache_pool[cache_lens_idx].append(input_len)
            input_all = torch.cat([kv_cache_pool[cache_idx], input_tensor], dim=seq_dim) if span_idx != 0 else input_tensor
            if not last_idx:
                # If not the last chunk, update the cache
                kv_cache_pool[cache_idx] = input_all
        else:
            # Second forward pass (during recomputation phase)
            if last_idx:
                # If last chunk, append input_tensor to cache
                kv_cache_pool[cache_idx] = torch.cat([kv_cache_pool[cache_idx], input_tensor], dim=seq_dim)
            else:
                # If not last chunk, truncate cache
                kv_cache_pool[cache_idx] = slice_tensor(
                    kv_cache_pool[cache_idx], 
                    None, 
                    -kv_cache_pool[cache_lens_idx][span_idx + 1], 
                    seq_dim)
            # Update length list to current chunk
            kv_cache_pool[cache_lens_idx] = kv_cache_pool[cache_lens_idx][:span_idx + 1]
            input_all = kv_cache_pool[cache_idx]

        input_all_len = input_all.shape[seq_dim]
        offset = input_all_len - input_len

        # Store information for backward pass
        ctx.seqlen = input_all_len
        ctx.offset = offset
        ctx.cache_idx = cache_idx

        return input_all.detach()


    @staticmethod
    def backward(ctx, grad):
        """Backward pass for gradient computation.
        
        Args:
            ctx: Context object with information stored during forward pass
            grad: Input gradient from subsequent layers
            
        Returns:
            Computed gradients for inputs
        """
        span_idx = ctx.span_info.span_idx
        seq_dim = ctx.span_info.seq_dim
        last_idx = ctx.span_info.last_span_idx

        cache_grad_idx = ctx.cache_idx + '_grad'

        # If not the last chunk, accumulate gradient from cache
        if not last_idx:
            grad += ctx.kv_cache_pool[cache_grad_idx]

        # Store current chunk's gradient to cache for previous chunks
        ctx.kv_cache_pool[cache_grad_idx] = slice_tensor(grad, None, ctx.offset, seq_dim).contiguous()

        # If this is the first chunk, clean up the cache
        if span_idx == 0:
            for key in list(ctx.kv_cache_pool.keys()):
                if '_grad' not in key:
                    ctx.kv_cache_pool.pop(key)
            ctx.kv_cache_pool.pop(cache_grad_idx)

        # Return gradient for current chunk only
        grad = slice_tensor(grad, ctx.offset, ctx.seqlen, seq_dim)
        return grad, None, None