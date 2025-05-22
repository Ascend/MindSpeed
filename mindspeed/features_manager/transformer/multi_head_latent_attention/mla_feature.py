# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class MLAFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__(
            'multi-latent-attention',
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

    def is_need_apply(self, args):
        return (self.optimization_level <= args.optimization_level and
                (getattr(args, self.feature_name, None) or getattr(args, "multi_head_latent_attention", None))) \
            or self.default_patches

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

            # map the mindspeed argument to megatron
            args.qk_head_dim = args.qk_nope_head_dim
            args.qk_pos_emb_head_dim = args.qk_rope_head_dim

            if not args.multi_latent_attention:
                args.multi_latent_attention = args.multi_head_latent_attention

    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace
    ):
        from mindspeed.core.transformer.multi_head_latent_attention import (
            multi_latent_attention_init_impl,
            dot_product_attention_init_wrapper,
        )
        patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MultiLatentAttention.__init__',
                                     multi_latent_attention_init_impl)
        patch_manager.register_patch(
            'megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
            dot_product_attention_init_wrapper
        )
