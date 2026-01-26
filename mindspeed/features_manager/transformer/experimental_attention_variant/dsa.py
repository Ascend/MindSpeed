from mindspeed.features_manager.feature import MindSpeedFeature


class DeepSeekSparseAttention(MindSpeedFeature):
    def __init__(self):
        super().__init__('experimental_attention_variant', optimization_level=2)

    def is_need_apply(self, args):
        """Check if the DSA feature needs to be applied."""
        return getattr(args, 'experimental_attention_variant', None) == 'dsa'

    def register_args(self, parser):
        group = parser.add_argument_group(title='experimental-attention-variant')
        group.add_argument("--experimental-attention-variant", type=str, default=None,
                           choices=['dsa'],
                           help="Experimental attention variant (e.g., dsa for DeepSeek Sparse Attention).")
        group.add_argument("--use-dsa-absorb", action='store_true', help="Enable matrix absorption in DSA.")
        group.add_argument("--use-fused-lightning-indexer", action='store_true',
                           help="Enable fused lightning indexer in DSA.")
        group.add_argument("--use-fused-sparse-flash-attention", action='store_true',
                           help="Enable sparse flashattention in DSA.")
        group.add_argument("--use-fused-lightning-indexer-kl-loss", action='store_true',
                           help="Enable sparse lightning indexer kl loss in DSA.")

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

            if args.context_parallel_size > 1 and args.context_parallel_algo != 'kvallgather_cp_algo':
                raise AssertionError(
                    "Context parallel is only supported with kvallgather_cp_algo for DSA."
                )