from mindspeed.features_manager.feature import MindSpeedFeature


class DeepSeekSparseAttention(MindSpeedFeature):
    def __init__(self):
        super().__init__('dsa', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-dsa-absorb", action='store_true', help="Enable matrix absorption in DSA.")
        group.add_argument("--use-fused-lightning-indexer", action='store_true', help="Enable fused lightning indexer in DSA.")
        group.add_argument("--use-fused-sparse-flash-attention", action='store_true', help="Enable sparse flashattention in DSA.")
        group.add_argument("--use-fused-lightning-indexer-kl-loss", action='store_true', help="Enable sparse lightning indexer kl loss in DSA.")

    def validate_args(self, args):
        if not getattr(args, 'qk_layernorm') and getattr(args, 'experimental_attention_variant') == 'dsa':
            raise AssertionError(
                'Megatron bug: qk_layernorm required for DSA MLA qk norm calculation.'
            )

        if args.use_fused_lightning_indexer or args.use_fused_sparse_flash_attention or args.use_fused_lightning_indexer_kl_loss:
            if not (args.use_dsa_absorb and args.use_fused_lightning_indexer 
                    and args.use_fused_sparse_flash_attention 
                    and args.use_fused_lightning_indexer_kl_loss):
                raise AssertionError(
                    "To use the DSA fusion operator, you must simultaneously enable "
                    "`--use-dsa-absorb`, `--use-fused-lightning-indexer`, "
                    "`--use-fused-sparse-flash-attention`, and "
                    "`--use-fused-lightning-indexer-kl-loss`."
                )

    def register_patches(self, pm, args):
        from mindspeed.core.transformer.experimental_attention_variant.dsa_matrix_naive import rotate_activation
        pm.register_patch('megatron.core.transformer.experimental_attention_variant.dsa.rotate_activation', rotate_activation)
        if args.use_dsa_absorb:
            from mindspeed.core.transformer.experimental_attention_variant.dsa_matrix_absorption import MLASelfAttentionAbsorb, unfused_dsa_fn, \
                compute_dsa_indexer_loss, get_dsa_module_spec_for_backend
            pm.register_patch('megatron.core.transformer.multi_latent_attention.MLASelfAttention', MLASelfAttentionAbsorb)
            pm.register_patch('megatron.core.transformer.experimental_attention_variant.dsa.unfused_dsa_fn', unfused_dsa_fn)
            pm.register_patch('megatron.core.transformer.experimental_attention_variant.dsa.compute_dsa_indexer_loss', compute_dsa_indexer_loss)
            pm.register_patch('megatron.core.models.gpt.experimental_attention_variant_module_specs.get_dsa_module_spec_for_backend', get_dsa_module_spec_for_backend)
        
        from mindspeed.core.transformer.experimental_attention_variant.dsa_fused import forward_with_scores, fused_dsa_attn_forward
        pm.register_patch('megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer.forward_with_scores', forward_with_scores)
        pm.register_patch('megatron.core.transformer.experimental_attention_variant.dsa.DSAttention.forward', fused_dsa_attn_forward)