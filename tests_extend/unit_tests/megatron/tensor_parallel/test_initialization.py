# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor

import megatron.core.parallel_state as Utils
from unit_tests.common import DistributedTest
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding, RowParallelLinear, ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


class TestInitialization(DistributedTest):  
    world_size = 8 
    args = parse_args(None, True)
    set_args(args)
    transformer_config = TransformerConfig(num_layers=1, hidden_size=12,
                                           num_attention_heads=4, use_cpu_initialization=True)
    
    @pytest.mark.timeout(100)
    def test_embedding_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        

        tp1 = VocabParallelEmbedding(num_embeddings=16, embedding_dim=4,
                                     init_method=self.transformer_config.init_method,
                                     config=self.transformer_config).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = VocabParallelEmbedding(num_embeddings=16, embedding_dim=4,
                                     init_method=self.transformer_config.init_method,
                                     config=self.transformer_config).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)
        Utils.destroy_model_parallel()

    @pytest.mark.timeout(100)
    def test_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = RowParallelLinear(input_size=16, output_size=16,
                                init_method=self.transformer_config.init_method,
                                bias=True, input_is_parallel=False,
                                config=self.transformer_config,
                                skip_bias_add=False).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = RowParallelLinear(input_size=16, output_size=16,
                                init_method=self.transformer_config.init_method,
                                bias=True,
                                input_is_parallel=False,
                                config=self.transformer_config,
                                skip_bias_add=False).weight
        
        if torch.distributed.get_rank() == 0:
            assert tp4.shape[1] * 4 == tp1.shape[1]
            assert torch.allclose(tp1[:, :4], tp4)
        Utils.destroy_model_parallel()

    @pytest.mark.timeout(100)
    def test_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = ColumnParallelLinear(input_size=16, output_size=16,
                                   init_method=self.transformer_config.init_method,
                                   bias=True, config=self.transformer_config,
                                   skip_bias_add=False).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = ColumnParallelLinear(input_size=16, output_size=16,
                                   init_method=self.transformer_config.init_method,
                                   bias=True, config=self.transformer_config,
                                   skip_bias_add=False).weight
        
        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)
        Utils.destroy_model_parallel()
