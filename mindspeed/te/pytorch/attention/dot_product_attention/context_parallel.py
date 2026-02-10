# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from einops import rearrange

from .kvallgather_context_parallel import (
    AttnFuncWithCPAndKVAllGatherForSBHD,
    AttnFuncWithCPAndKVAllGatherForTHD,
)

from .ulysses_context_parallel import AttnFuncWithCPAndQKVOA2A
from .utils import prepare_sbhd_format, prepare_thd_format


class BaseCPStrategy(torch.nn.Module, ABC):
    """Base class for all Context Parallelism strategies"""
    
    def __init__(self,
                 softmax_scale: float,
                 attention_dropout: float = 0.0,
                 attention_type: str = "self",
                 deterministic: bool = False):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.deterministic = deterministic

    @abstractmethod
    def forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            max_seqlen_q,
            max_seqlen_kv,
            cp_group,
            cp_global_ranks,
            cp_stream,
            **kwargs
    ) -> torch.Tensor:
        pass


class UlyssesCPStrategy(BaseCPStrategy):
    """Ulysses-based Context Parallelism Strategy"""

    def __init__(self,
                 softmax_scale: float,
                 attention_dropout: float = 0.0,
                 attention_type: str = "self",
                 deterministic: bool = False,
                 scatter_idx: int = 2,
                 gather_idx: int = 0,
                 gather_size_calculator=None, # Injected dependency
    ):
        super().__init__(softmax_scale, attention_dropout, attention_type, deterministic)
        self.ulysses_comm_para = {
            'scatter_idx': scatter_idx,
            'gather_idx': gather_idx,
            'gather_size_calculator': gather_size_calculator
        }

    def forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            max_seqlen_q,
            max_seqlen_kv,
            cp_group,
            cp_global_ranks,
            cp_stream,
            **kwargs
    ):
        return AttnFuncWithCPAndQKVOA2A(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            self.attention_dropout,
            self.softmax_scale,
            self.deterministic,
            cp_group,
            cp_stream,
            self.ulysses_comm_para
        )


class KVAllGatherCPStrategy(BaseCPStrategy):
    """AllGather-based Context Parallelism Strategy"""

    def forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            max_seqlen_q,
            max_seqlen_kv,
            cp_group,
            cp_global_ranks,
            cp_stream,
            **kwargs
    ):  
        
        # Prepare tensors based on format
        if qkv_format == 'sbhd':
            query_layer, key_layer, value_layer, n_head = prepare_sbhd_format(
                query_layer, key_layer, value_layer
            )

            return AttnFuncWithCPAndKVAllGatherForSBHD.apply(
                query_layer,
                key_layer,
                value_layer,
                n_head,
                attention_mask,
                qkv_format,
                attn_mask_type,
                self.attention_dropout,
                self.softmax_scale,
                self.deterministic,
                cp_group,
                cp_stream
            )
            
        elif qkv_format == 'thd':
            n_head, cu_seqlens_q, cu_seqlens_kv = prepare_thd_format(
                query_layer, cu_seqlens_q, cu_seqlens_kv
            )

            return AttnFuncWithCPAndKVAllGatherForTHD.apply(
                query_layer,
                key_layer,
                value_layer,
                n_head,
                attention_mask,
                qkv_format,
                attn_mask_type,
                self.attention_dropout,
                self.softmax_scale,
                self.deterministic,
                cp_group,
                cu_seqlens_q,
                cu_seqlens_kv
            )
            
        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")


class CPStrategyFactory:
    """Factory for creating Context Parallelism strategies"""
    
    _STRATEGY_MAP = {
        "kvallgather_cp_algo": KVAllGatherCPStrategy,
        "ulysses_cp_algo": UlyssesCPStrategy,
    }
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        **strategy_params
    ) -> BaseCPStrategy:
        """Create a CP strategy instance
        
        Args:
            strategy_type: Type of strategy to create
            **strategy_params: Parameters for the strategy constructor
            
        Returns:
            Instance of the requested strategy
            
        Raises:
            ValueError: If strategy_type is not supported
        """
        if strategy_type not in CPStrategyFactory._STRATEGY_MAP:
            available = CPStrategyFactory.get_available_strategies()
            raise ValueError(
                f"Unsupported CP strategy: {strategy_type}. "
                f"Available strategies: {available}"
            )
        
        strategy_class = CPStrategyFactory._STRATEGY_MAP[strategy_type]
        return strategy_class(**strategy_params)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names"""
        return list(CPStrategyFactory._STRATEGY_MAP.keys())