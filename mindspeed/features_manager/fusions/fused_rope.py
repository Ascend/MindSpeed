# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature


class FusedRoPEFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(
            'use-fused-rotary-pos-emb',
            optimization_level=0
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='fusion')

        group.add_argument(
            "--use-fused-rotary-pos-emb",
            action='store_true',
            help="Use fused rotary-pos-emb."
        )

    def validate_args(self, args: Namespace):
        if (
            args.use_fused_rotary_pos_emb and
            args.position_embedding_type != 'rope'
        ):
            raise AssertionError(
                '--use-fused-rotary-pos-emb must enable with'
                '--position-embedding-type=rope'
            )

    def register_patches(self, patch_manager, args: Namespace):
        from mindspeed.core.fusions.fused_rope import (apply_rotary_pos_emb_bshd, transformer_config_post_init_wrapper,
                                                       apply_rotary_pos_emb)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd',
                                     apply_rotary_pos_emb_bshd)
        patch_manager.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
                                     transformer_config_post_init_wrapper)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb',
                                     apply_rotary_pos_emb)