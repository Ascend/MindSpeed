# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import sys

# Setting sys.argv is mainly to ensure that --noop-layers is not None, so that the code block (which will be executed
# after determining that noop_layers is not None) will be executed in megatron_adaptor.
sys.argv = [
    sys.argv[0],
    '--noop-layers', '22,23',
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
    '/home/dataset/llama2/alpaca_text_document',
    '--seed', '1234',
]
import torch
import torch_npu
import pytest
import mindspeed.megatron_adaptor

from mindspeed.model.transformer import NoopTransformerLayer
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.legacy.model.enums import LayerType
from megatron.core import mpu
from megatron.training import get_args
from megatron.legacy.model.transformer import ParallelTransformer
from megatron.training.arguments import parse_args, validate_args
from megatron.core.transformer.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.training.initialize import _initialize_distributed, _set_random_seed
from unit_tests.common import DistributedTest

os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = "1"


def _get_offset(self, config):
    argument = get_args()
    if config.virtual_pipeline_model_parallel_size is not None:
        assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
            'num_layers_per_stage must be divisible by ' \
            'virtual_pipeline_model_parallel_size'
        assert argument.model_type != ModelType.encoder_and_decoder
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]
        offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                 (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
    else:
        # Each stage gets a contiguous set of layers.
        if argument.model_type == ModelType.encoder_and_decoder and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            if self.layer_type == LayerType.encoder:
                offset = pipeline_rank * self.num_layers
            else:
                num_ranks_in_enc = argument.pipeline_model_parallel_split_rank
                offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
        else:
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
    return offset


class TestNoopLayer(DistributedTest):

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

    def set_args(self, tp_pp_vp_stage, num_layers, noop_layers):
        args = parse_args(ignore_unknown_args=True)
        (tp, pp, vp_stage) = tp_pp_vp_stage
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.num_layers_per_virtual_pipeline_stage = vp_stage
        args.model_type = ModelType.encoder_or_decoder
        args.noop_layers = noop_layers
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
        validate_args(args)
        set_args(args)

    def initialize_distributed(self):
        args = get_args()
        destroy_model_parallel()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        _set_random_seed(args.seed, args.data_parallel_random_init)

    @pytest.mark.parametrize("tp_pp_vp_stage", [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)])
    @pytest.mark.parametrize("num_layers", [24, 10])
    @pytest.mark.parametrize("noop_layers", ["0,23", "22,23", "0,9", "0,4,7,8,9", None])
    def test_noop_layers(self, tp_pp_vp_stage, num_layers, noop_layers):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_parallel_transformer()
        args = get_args()
        assert num_layers == args.num_layers

        if isinstance(noop_layers, str):
            assert args.noop_layers == {int(x) for x in noop_layers.split(',') if 0 <= int(x) < args.num_layers}

        assert isinstance(args.noop_layers, (set, type(None))), \
            f"args.noop_layers should be set or None，but got {type(args.noop_layers)}"

        for i, layer in enumerate(self.parallel_transformer.layers):
            offset = _get_offset(self.parallel_transformer, self.transformer_config)
            global_num_layers = i + offset + 1

            if isinstance(args.noop_layers, set) and global_num_layers - 1 in args.noop_layers:
                assert isinstance(layer, NoopTransformerLayer)
            else:
                assert not isinstance(layer, NoopTransformerLayer)
