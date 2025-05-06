# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class MLAFeature(MindSpeedFeature):
    """
    Multi-Head Latent Attention feature.

    Arguments(other):
        - use_flash_attn
    """

    def __init__(self):
        super().__init__(
            'multi-head-latent-attention',
            optimization_level=2
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(
            title='multi head latent attention'
        )

        # main contral argument
        group.add_argument(
            '--multi-head-latent-attention',
            action='store_true',
            default=False,
            help='Use Multi-head Latent Attention(MLA)'
        )
        # mla arguments
        group.add_argument(
            '--qk-rope-head-dim',
            type=int,
            default=None,
            help='The qk head dim for rope'
        )
        group.add_argument(
            '--qk-nope-head-dim',
            type=int,
            default=None,
            help='The qk head dim for only self-attn'
        )

        # yarn arguments
        group.add_argument(
            '--rope-scaling-type',
            type=str,
            default=None,
            choices=['yarn', ],
            help='Set the rope scaling type, only support "yarn" type now'
        )
        group.add_argument(
            '--rope-scaling-beta-fast',
            type=int,
            default=32,
            help='Yarn rope: rope beta fast'
        )
        group.add_argument(
            '--rope-scaling-beta-slow',
            type=int,
            default=1,
            help='Yarn rope: rope beta slow'
        )
        # megatron has similar argument，only default is not same
        group.add_argument(
            '--rope-scaling-factor',
            type=float,
            default=1.0,
            help='Yarn rope: rope factor'
        )
        group.add_argument(
            '--rope-scaling-mscale',
            type=float,
            default=1.0,
            help='Yarn rope: rope mscale'
        )
        group.add_argument(
            '--rope-scaling-mscale-all-dim',
            type=float,
            default=0.0,
            help='Yarn rope: rope mscale all dim'
        )
        group.add_argument(
            '--rope-scaling-original-max-position-embeddings',
            type=int,
            default=None,
            help='Yarn rope: rope original max position embeddings'
        )

        group.add_argument(
            '--shape-order',
            type=str,
            default='SBH',
            choices=['SBH', 'BSH', 'BSND'],
            help='input shape order used by Flash attention'
        )

    def validate_args(self, args: Namespace):
        if args.multi_head_latent_attention:
            if args.kv_lora_rank is None:
                raise AssertionError(
                    'The parameter kv-lora-rank should be '
                    'set when use multi_head_latent_attention.'
                )
            elif args.v_head_dim is None:
                raise AssertionError(
                    'The parameter v-head-dim should be '
                    'set when use multi_head_latent_attention.'
                )
            elif args.qk_rope_head_dim is None:
                raise AssertionError(
                    'The parameter qk-rope-head-dim should be '
                    'set when use multi_head_latent_attention.'
                )
            elif args.qk_nope_head_dim is None:
                raise AssertionError(
                    'The parameter qk-nope-head-dim should be '
                    'set when use multi_head_latent_attention.'
                )
        if args.rope_scaling_type == "yarn":
            if args.rope_scaling_original_max_position_embeddings is None:
                raise AssertionError(
                    'The parameter '
                    'rope_scaling_original_max_position_embeddings '
                    'should be set when use yarn.'
                )

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.transformer.multi_head_latent_attention.adaptor import (
            SelfAttentionSubmodules,
            self_attention_init_wrapper,
            attention_forward,
            dot_product_attention_init_wrapper,
            get_gpt_layer_local_spec_wrapper,
            rotary_embedding_init_wrapper,
        )
        patch_manager.register_patch('megatron.core.transformer.attention.SelfAttentionSubmodules',
                                     SelfAttentionSubmodules)
        patch_manager.register_patch(
            'megatron.core.transformer.attention.SelfAttention.__init__',
            self_attention_init_wrapper
        )
        patch_manager.register_patch(
            "megatron.core.transformer.attention.Attention.forward",
            attention_forward
        )
        patch_manager.register_patch(
            'megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
            dot_product_attention_init_wrapper
        )
        patch_manager.register_patch(
            'megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
            get_gpt_layer_local_spec_wrapper
        )
        patch_manager.register_patch(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
            rotary_embedding_init_wrapper
        )
