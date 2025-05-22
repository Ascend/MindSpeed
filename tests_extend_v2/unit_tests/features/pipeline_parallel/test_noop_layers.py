# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
import os
import sys
from argparse import Namespace

import torch
import torch_npu
import pytest

import mindspeed.megatron_adaptor_v2

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core import mpu
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.moe.moe_utils import (
    save_to_aux_losses_tracker,
    clear_aux_losses_tracker,
)

from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_args
from megatron.training.training import num_floating_point_operations
from megatron.training.initialize import (
    _initialize_distributed,
    _set_random_seed,
)

from mindspeed.core.pipeline_parallel.noop_layers.adaptor import (
    origin_flop_calculator,
)
from mindspeed.core.pipeline_parallel.noop_layers.moe_metrics_tracker import (
    get_mean,
)
from mindspeed.core.transformer.transformer_block import _get_layer_offset
from mindspeed.core.pipeline_parallel.noop_layers.adaptor import (
    NoopTransformerLayer,
)

from tests_extend.unit_tests.common import DistributedTest


os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
# Setting sys.argv is mainly to ensure that --noop-layers is not None,
# so that the code block (which will be executed
# after determining that noop_layers is not None)
# will be executed in megatron_adaptor.
sys.argv = [
    sys.argv[0],
    "--noop-layers",
    "22,23",
    "--num-layers",
    "24",
    "--hidden-size",
    "8",
    "--ffn-hidden-size",
    "8",
    "--num-attention-heads",
    "8",
    "--tokenizer-type",
    "Llama2Tokenizer",
    "--tokenizer-model",
    "/home/dataset/model/llama-2-7b-hf/tokenizer.model",
    "--seq-length",
    "128",
    "--max-position-embeddings",
    "128",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "8",
    "--lr-warmup-fraction",
    "0.01",
    "--bf16",
    "--data-path",
    "/home/dataset/llama2/alpaca_text_document",
    "--seed",
    "1234",
]


def track_moe_metrics_mock(loss_scale, writer, total_loss_dict=None):
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    args = get_args()
    # Aux loss logging
    if writer is not None:
        aux_losses = {
            k: v["values"].float() * loss_scale
            for k, v in tracker.items()  # type: ignore
        }
        for name, loss_list in aux_losses.items():
            loss_list_mean = get_mean(args, loss_list)
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list_mean
                else:
                    total_loss_dict[name] += loss_list_mean
    clear_aux_losses_tracker()


def track_moe_metrics_megatron_original_mock(
    loss_scale,
    writer,
    total_loss_dict=None,
):
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    # Aux loss logging
    if writer is not None:
        aux_losses = {
            k: v["values"].float() * loss_scale
            for k, v in tracker.items()  # type: ignore
        }
        for name, loss_list in aux_losses.items():
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()
    clear_aux_losses_tracker()


class TestNoopLayer(DistributedTest):
    world_size = 8

    def init_transformer_block(self):
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
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
        )
        self.parallel_transformer = TransformerBlock(
            config=self.transformer_config,
            spec=transformer_layer_spec,
            post_layer_norm=False,
            pre_process=False,
            post_process=False,
        )

    def set_args(self, tp_pp_vp_stage, num_layers, noop_layers):
        args = parse_args(ignore_unknown_args=True)
        (tp, pp, vp_stage) = tp_pp_vp_stage
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.num_layers_per_virtual_pipeline_stage = vp_stage
        args.model_type = ModelType.encoder_or_decoder
        args.noop_layers = noop_layers
        args.num_layers = num_layers
        # In validate_args(), first get args.batch_size,
        # and then del args.batch_size, so you need to set some parameters
        # first to prevent errors from running validate_args() again.
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
        _initialize_distributed(None, None)

        # Random seeds for reproducibility.
        _set_random_seed(args.seed, args.data_parallel_random_init)

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [24, 10])
    @pytest.mark.parametrize("noop_layers", ["0,4,7,8,9"])
    def test_valid_noop_layers(self, tp_pp_vp_stage, num_layers, noop_layers):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_transformer_block()
        args = get_args()
        assert num_layers == args.num_layers
        assert num_layers == self.transformer_config.num_layers
        assert args.noop_layers == {0, 4, 7, 8, 9}

        for i, layer in enumerate(self.parallel_transformer.layers):
            offset = _get_layer_offset(args)
            global_num_layers = i + offset + 1

            if (
                isinstance(args.noop_layers, set)
                and global_num_layers - 1 in args.noop_layers
            ):
                assert isinstance(layer, NoopTransformerLayer)
            else:
                assert not isinstance(layer, NoopTransformerLayer)

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [10])
    @pytest.mark.parametrize("noop_layers", ["0,2,15,6,8"])
    def test_invalid_noop_layers_out_of_range(
        self, tp_pp_vp_stage, num_layers, noop_layers
    ):
        with pytest.raises(AssertionError) as context:
            self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        assert (
            "each element in args.noop_layers(0,2,15,6,8) "
            "should bigger or equal to 0 and smaller than "
            "args.num_layers(10)"
        ) in str(context.value)

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [4])
    @pytest.mark.parametrize("noop_layers", ["0,1,1,3"])
    def test_moe_metrics_with_noop_layers(
        self, tp_pp_vp_stage, num_layers, noop_layers
    ):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_transformer_block()
        args = get_args()
        loss = torch.tensor(1).npu()
        name = "load_balancing_loss"

        for layer_number in range(1, num_layers + 1):
            if (
                isinstance(args.noop_layers, set)
                and layer_number - 1 in args.noop_layers
            ) or args.noop_layers is None:
                save_to_aux_losses_tracker(
                    name,
                    loss,
                    layer_number,
                    num_layers,
                )

        total_loss_dict = dict()
        track_moe_metrics_mock(1, True, total_loss_dict)
        assert total_loss_dict.get(name) == 3
        del parallel_state._MOE_LAYER_WISE_LOGGING_TRACKER[name]

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [4])
    @pytest.mark.parametrize("noop_layers", [None])
    def test_moe_metrics_without_noop_layers(
        self, tp_pp_vp_stage, num_layers, noop_layers
    ):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_transformer_block()
        args = get_args()
        loss = torch.tensor(1).npu()
        name = "load_balancing_loss"

        for layer_number in range(1, num_layers + 1):
            if (
                isinstance(args.noop_layers, set)
                and layer_number - 1 in args.noop_layers
            ) or args.noop_layers is None:
                save_to_aux_losses_tracker(
                    name,
                    loss,
                    layer_number,
                    num_layers,
                )

        total_loss_dict = dict()
        track_moe_metrics_mock(1, True, total_loss_dict)
        assert total_loss_dict.get(name) == 1
        del parallel_state._MOE_LAYER_WISE_LOGGING_TRACKER[name]

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [4])
    @pytest.mark.parametrize("noop_layers", ["0,3"])
    def test_moe_metrics_megatron_original_with_noop_layers(
        self, tp_pp_vp_stage, num_layers, noop_layers
    ):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_transformer_block()
        args = get_args()
        loss = torch.tensor(1).npu()
        name = "load_balancing_loss"

        for layer_number in range(1, num_layers + 1):
            if (
                isinstance(args.noop_layers, set)
                and layer_number - 1 in args.noop_layers
            ) or args.noop_layers is None:
                save_to_aux_losses_tracker(
                    name,
                    loss,
                    layer_number,
                    num_layers,
                )

        total_loss_dict = dict()
        track_moe_metrics_megatron_original_mock(1, True, total_loss_dict)
        # using the megatron original track moe metrics function
        # and using the noop_layers,
        # will get the following wrong results:
        assert total_loss_dict.get(name) == 0.5
        del parallel_state._MOE_LAYER_WISE_LOGGING_TRACKER[name]

    @pytest.mark.parametrize(
        "tp_pp_vp_stage",
        [(2, 2, 1), (2, 2, None), (2, 1, None), (1, 2, None), (1, 1, None)],
    )
    @pytest.mark.parametrize("num_layers", [4])
    @pytest.mark.parametrize("noop_layers", [None])
    def test_moe_metrics_megatron_original_without_noop_layers(
        self, tp_pp_vp_stage, num_layers, noop_layers
    ):
        self.set_args(tp_pp_vp_stage, num_layers, noop_layers)
        self.initialize_distributed()
        self.init_transformer_block()
        args = get_args()
        loss = torch.tensor(1).npu()
        name = "load_balancing_loss"

        for layer_number in range(1, num_layers + 1):
            if (
                isinstance(args.noop_layers, set)
                and layer_number - 1 in args.noop_layers
            ) or args.noop_layers is None:
                save_to_aux_losses_tracker(
                    name,
                    loss,
                    layer_number,
                    num_layers,
                )

        total_loss_dict = dict()
        track_moe_metrics_mock(1, True, total_loss_dict)
        assert total_loss_dict.get(name) == 1
        del parallel_state._MOE_LAYER_WISE_LOGGING_TRACKER[name]

    @pytest.mark.parametrize(
        "num_layers, noop_layers, megatron_expected, mindspeed_expected",
        [
            (4, None, 184817811456.0, 184817811456.0),
            (4, {1, 2}, 184817811456.0, 114152177664.0),
        ],
    )
    def test_cacl_flop_adaptor(
        self, num_layers, noop_layers, megatron_expected, mindspeed_expected
    ):
        args = Namespace()
        args.num_layers = num_layers
        args.noop_layers = noop_layers
        args.kv_channels = 128
        args.num_attention_heads = 6
        args.hidden_size = 768
        args.seq_length = 768
        args.ffn_hidden_size = 3072
        args.padded_vocab_size = 12288
        args.group_query_attention = False
        args.swiglu = False
        args.num_experts = None
        args.is_hybrid_model = None
        args.mtp_num_layers = None
        args.moe_ffn_hidden_size = 0
        args.multi_latent_attention = None
        args.moe_shared_expert_intermediate_size = None
        batch_size = 1
        megatron_flop = origin_flop_calculator(args, batch_size)
        mindspeed_flop = num_floating_point_operations(args, batch_size)
        assert megatron_flop == megatron_expected
        assert mindspeed_flop == mindspeed_expected
