from mindspeed.features_manager.feature import MindSpeedFeature


class DeepSeekSparseAttention(MindSpeedFeature):
    def __init__(self):
        super().__init__('experimental_attention_variant', optimization_level=2)

    @staticmethod
    def _get_context_parallel_size(args):
        context_parallel_size = getattr(args, 'context_parallel_size', 1)
        if context_parallel_size is None:
            return 1
        return int(context_parallel_size)

    def is_need_apply(self, args):
        """Check if the DSA feature needs to be applied."""
        return getattr(args, 'experimental_attention_variant', None) == 'dsa'

    def register_args(self, parser):
        group = parser.add_argument_group(title='experimental-attention-variant')
        group.add_argument("--use-dsa-absorb", action='store_true', help="Enable matrix absorption in DSA.")
        group.add_argument(
            "--use-fused-lightning-indexer", action='store_true', help="Enable fused lightning indexer in DSA."
        )
        group.add_argument(
            "--use-fused-sparse-flash-attention", action='store_true', help="Enable sparse flashattention in DSA."
        )
        group.add_argument(
            "--use-fused-lightning-indexer-kl-loss",
            action='store_true',
            help="Enable sparse lightning indexer kl loss in DSA.",
        )
        group.add_argument(
            "--apply-rope-in-complex", action='store_true', help="Apply complex computation of rope in DSA."
        )

    def validate_args(self, args):
        is_dsa = getattr(args, 'experimental_attention_variant', None) == 'dsa'
        dsa_options_enabled = any(
            getattr(args, option_name, False)
            for option_name in (
                'use_dsa_absorb',
                'use_fused_lightning_indexer',
                'use_fused_sparse_flash_attention',
                'use_fused_lightning_indexer_kl_loss',
                'apply_rope_in_complex',
            )
        )
        if not is_dsa:
            if dsa_options_enabled:
                raise AssertionError("DSA-specific options require --experimental-attention-variant dsa.")
            return

        if not getattr(args, 'qk_layernorm', False):
            raise AssertionError('Megatron bug: qk_layernorm required for DSA MLA qk norm calculation.')

        # P1+P2+P3 are tightly coupled: P2's raw NPU op has no autograd (backward is
        # handled by P3's SparseLIGradKlLoss), and P3 reuses P1's softmax stats and
        # P2's precomputed results. They must be enabled together (or all off).
        use_p1 = getattr(args, 'use_fused_sparse_flash_attention', False)
        use_p2 = getattr(args, 'use_fused_lightning_indexer', False)
        use_p3 = getattr(args, 'use_fused_lightning_indexer_kl_loss', False)
        if (use_p1 or use_p2 or use_p3) and not (use_p1 and use_p2 and use_p3):
            raise AssertionError(
                "DSA P1+P2+P3 are tightly coupled and must be enabled together: "
                "--use-fused-sparse-flash-attention (P1), --use-fused-lightning-indexer (P2), "
                "--use-fused-lightning-indexer-kl-loss (P3). P2 uses a raw NPU op without "
                "autograd (backward handled by P3), and P3 reuses P1's softmax stats."
            )
        if use_p3 and getattr(args, 'num_attention_heads', None) not in (32, 64, 128):
            raise AssertionError(
                "DSA fused lightning indexer KL loss requires --num-attention-heads to be "
                "32, 64, or 128. The NPU npu_sparse_lightning_indexer_grad_kl_loss op "
                f"receives the TP-allgathered attention head count as Q_N, but got "
                f"{getattr(args, 'num_attention_heads', None)}. Disable the fused DSA "
                "P1/P2/P3 path or use a supported attention-head count."
            )

        if self._get_context_parallel_size(args) > 1:
            if args.context_parallel_algo != 'kvallgather_cp_algo':
                raise AssertionError("DSA with context_parallel requires kvallgather_cp_algo")

        if getattr(args, 'eod_mask_loss', False):
            raise ValueError("DSA does not support EOD mask loss.")

    def register_patches(self, patch_manager, args):
        """Register all DSA NPU optimization patches.

        Patch coupling:
        - P1+P2+P3 are tightly coupled and registered as a group (when all enabled)
        - P4 (complex RoPE) and P5 (matrix absorption) are independent
        - Without any NPU flags: pure Megatron native DSA
        """
        import logging

        logger = logging.getLogger(__name__)

        # P1+P2+P3: Tightly-coupled NPU fused operators
        use_p1 = getattr(args, 'use_fused_sparse_flash_attention', False)
        use_p2 = getattr(args, 'use_fused_lightning_indexer', False)
        use_p3 = getattr(args, 'use_fused_lightning_indexer_kl_loss', False)

        from mindspeed.core.transformer.experimental_attention_variant.dsa_module_spec import (
            get_experimental_attention_variant_module_spec,
        )

        patch_manager.register_patch(
            'megatron.core.models.gpt.experimental_attention_variant_module_specs.'
            'get_experimental_attention_variant_module_spec',
            get_experimental_attention_variant_module_spec,
        )
        from megatron.core.transformer.experimental_attention_variant import dsa as megatron_dsa

        if megatron_dsa.hadamard_transform is None:
            from mindspeed.core.transformer.experimental_attention_variant.dsa_hadamard import (
                hadamard_transform,
            )

            megatron_dsa.hadamard_transform = hadamard_transform
            logger.info("DSA: fast_hadamard_transform is unavailable; using MindSpeed torch fallback.")

        if use_p1 and use_p2 and use_p3:
            from megatron.core.transformer.experimental_attention_variant.dsa import (
                DSAIndexer,
                DSAttention,
            )
            from mindspeed.core.transformer.experimental_attention_variant import dsa_npu_fused  # pylint: disable=no-name-in-module

            # P1: Replace DSAttention.forward with fused NPU path
            DSAttention.forward = dsa_npu_fused.fused_dsa_attn_forward

            # P2: Replace DSAIndexer.forward_with_scores with NPU fused path
            # Save original method for fallback (use_fused_lightning_indexer=False path)
            dsa_npu_fused._original_DSAIndexer_forward_with_scores = DSAIndexer.forward_with_scores
            DSAIndexer.forward_with_scores = dsa_npu_fused.forward_with_scores

            logger.info(
                "DSA P1+P2+P3: Tightly-coupled NPU fused operators registered "
                "(DSAttention.forward + DSAIndexer.forward_with_scores replaced)"
            )

        # P4: Replace DSAIndexer._apply_rope with complex domain RoPE (independent)
        if getattr(args, 'apply_rope_in_complex', False):
            import torch
            from megatron.core.transformer.experimental_attention_variant.dsa import (
                DSAIndexer,
            )
            from mindspeed.core.transformer.experimental_attention_variant.dsa_rope import (  # pylint: disable=no-name-in-module
                apply_rope_in_complex,
            )

            def _apply_rope_complex(self, x, rotary_pos_emb, mscale):
                """Replacement for DSAIndexer._apply_rope using complex RoPE."""
                no_pe_dim = self.index_head_dim - self.qk_pos_emb_head_dim
                x_nope, x_pe = torch.split(x, [no_pe_dim, self.qk_pos_emb_head_dim], dim=-1)
                x_pe = apply_rope_in_complex(x_pe, rotary_pos_emb, mscale=mscale)
                return torch.cat([x_nope, x_pe], dim=-1)

            DSAIndexer._apply_rope = _apply_rope_complex

            logger.info("DSA P4: DSAIndexer._apply_rope replaced with complex domain RoPE")

        # P5: MLASelfAttention -> MLASelfAttentionAbsorb (training-time matrix absorption, independent)
        if getattr(args, 'use_dsa_absorb', False):
            from mindspeed.core.transformer.experimental_attention_variant.dsa_matrix_absorption import (  # pylint: disable=no-name-in-module
                MLASelfAttentionAbsorb,
                compute_dsa_indexer_loss,
                get_dsa_module_spec_for_backend,
                unfused_dsa_fn,
            )

            patch_manager.register_patch(
                'megatron.core.transformer.multi_latent_attention.MLASelfAttention',
                MLASelfAttentionAbsorb,
            )
            patch_manager.register_patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.unfused_dsa_fn',
                unfused_dsa_fn,
            )
            patch_manager.register_patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.compute_dsa_indexer_loss',
                compute_dsa_indexer_loss,
            )
            patch_manager.register_patch(
                'megatron.core.models.gpt.experimental_attention_variant_module_specs.get_dsa_module_spec_for_backend',
                get_dsa_module_spec_for_backend,
            )
            logger.info(
                "DSA P5: Training-time matrix absorption patches registered "
                "(MLASelfAttentionAbsorb + DSA helpers + DSA module spec)"
            )

        # CP support: kvallgather context parallel with DSA
        if (
            self._get_context_parallel_size(args) > 1
            and getattr(args, 'context_parallel_algo', None) == 'kvallgather_cp_algo'
        ):
            from mindspeed.core.transformer.experimental_attention_variant.dsa_kvallgather_context_parallel import (  # pylint: disable=no-name-in-module
                transformer_config_post_init_wrapper,
            )
            from megatron.core.transformer.transformer_config import TransformerConfig

            # Wrap TransformerConfig.__post_init__ to bypass the DSA+CP assertion
            TransformerConfig.__post_init__ = transformer_config_post_init_wrapper(TransformerConfig.__post_init__)

            logger.info(
                "DSA CP: kvallgather context parallel support enabled. "
                "TransformerConfig.__post_init__ wrapped to bypass DSA+CP assertion. "
                "CP-aware fused operators will be invoked from fused_dsa_attn_forward."
            )
