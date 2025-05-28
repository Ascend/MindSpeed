# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import sys
import pytest
import torch

from mindspeed import megatron_adaptor_v2
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core.transformer.moe.moe_utils import (save_to_aux_losses_tracker, clear_aux_losses_tracker)
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.initialize import _initialize_distributed, _set_random_seed
from megatron.legacy.model.transformer import ParallelTransformer, NoopTransformerLayer

from mindspeed.core.transformer.moe.moe_utils import get_mean

from tests_extend_v2.unit_tests.common import DistributedTest


os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = "1"


class TestUnalignedPP(DistributedTest):
    world_size = 8

    def initialize_env(self):
        self.ori_sys = sys.argv
        sys.argv = [
            sys.argv[0],
            '--num-layers', '24',
            '--hidden-size', '8',
            '--ffn-hidden-size', '8',
            '--num-attention-heads', '8',
            '--tokenizer-type', 'Llama2Tokenizer',
            '--tokenizer-model', '/home/dataset/model/llama-2-7b-hf/tokenizer.model',
            '--seq-length', '128',
            '--max-position-embeddings', '128',
            '--micro-batch-size', '1',
            '--global-batch-size', '8',
            '--lr-warmup-fraction', '0.01',
            '--bf16',
            '--data-path',
            '--transformer-impl local'
            '/home/dataset/llama2/alpaca_text_document',
            '--seed', '1234',
        ]
    
    def del_env(self):
        sys.argv = self.ori_sys

    def init_parallel_transformer(self):
        args = get_args()
        self.transformer_config = TransformerConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            ffn_hidden_size=args.hidden_size,
            use_cpu_initialization=args.use_cpu_initialization,
            fp16=False,
            sequence_parallel=args.sequence_parallel,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
        )
        self.parallel_transformer = ParallelTransformer(self.transformer_config,
                                                        model_type=ModelType.encoder_or_decoder)

    def set_args(self, tp_pp_vp_stage_num_layers, pipeline_num_transformer_layers):
        args = parse_args(ignore_unknown_args=True)
        (tp, pp, vp_stage, num_layers) = tp_pp_vp_stage_num_layers
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.num_layers_per_virtual_pipeline_stage = vp_stage
        args.model_type = ModelType.encoder_or_decoder
        args.pipeline_num_transformer_layers = pipeline_num_transformer_layers
        args.num_layers = num_layers
        # In validate_args(), first get args.batch_size, and then del args.batch_size, so you need to set some
        # parameters first to prevent errors from running validate_args() again.
        args.batch_size = None
        args.warmup = None
        args.model_parallel_size = None
        args.checkpoint_activations = False
        args.recompute_activations = False
        args.encoder_num_layers = None
        args.sequence_parallel = None
        args.encoder_seq_length = None
        args.start_weight_decay = None
        args.end_weight_decay = None
        args.transformer_impl = "local"
        validate_args(args)
        set_args(args)

    def initialize_distributed(self):
        args = get_args()
        destroy_model_parallel()
        # Pytorch distributed.
        _initialize_distributed(None, None)

        # Random seeds for reproducibility.
        _set_random_seed(args.seed, args.data_parallel_random_init)

    @pytest.mark.parametrize("tp_pp_vp_stage_num_layers", [(2, 4, 1, 8)])
    @pytest.mark.parametrize("pipeline_num_transformer_layers", ["[[0,1], [1,1], [1,1],[3,0]]"])
    def test_moe_metrics_with_unaligned_layers(self, tp_pp_vp_stage_num_layers, pipeline_num_transformer_layers):
        self.initialize_env()
        self.set_args(tp_pp_vp_stage_num_layers, pipeline_num_transformer_layers)
        self.initialize_distributed()
        self.init_parallel_transformer()
        args = get_args()
        num_layers = tp_pp_vp_stage_num_layers[3]
        assert num_layers == args.num_layers
        assert num_layers == self.transformer_config.num_layers
        loss = torch.tensor(1).npu()
        name = "load_balancing_loss"

        for layer_number in range(1, num_layers + 1):
            save_to_aux_losses_tracker(name, loss, layer_number, num_layers)

        total_loss_dict = dict()
        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        aux_losses = {k: v['values'].float() for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            loss_list_mean = get_mean(loss_list)
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list_mean
                else:
                    total_loss_dict[name] += loss_list_mean
        clear_aux_losses_tracker()
        assert total_loss_dict.get(name) == 1
        del parallel_state._MOE_LAYER_WISE_LOGGING_TRACKER[name]
        self.del_env()
    