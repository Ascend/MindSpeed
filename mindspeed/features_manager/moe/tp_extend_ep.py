from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoETpExtendEpFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('moe-tp-extend-ep', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--moe-tp-extend-ep", action='store_true',
                           help="use tp group to extend experts parallelism"
                                "instead of sharding weight tensor of experts in tp group")
        # this param can be removed
        group.add_argument("--moe-permutation-async-comm", action='store_true',
                           help="overlap moe permutation 3 all gather communications")

    def validate_args(self, args):
        if args.moe_tp_extend_ep:
            if args.num_experts % (args.tensor_model_parallel_size * args.expert_model_parallel_size) != 0:
                raise AssertionError('`--moe-tp-extend-ep` only support when num_experts % ( tp * ep ) == 0')
            if not (args.moe_permutation_async_comm and args.moe_grouped_gemm):
                raise AssertionError(
                    '`--moe-tp-extend-ep` needs `--moe-permutation-async-comm` and `--moe-grouped-gemm`.')
            if args.moe_expert_capacity_factor is not None:
                raise AssertionError('`--moe-tp-extend-ep` only support when moe_expert_capacity_factor is None.')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllSEQTptoEpMoELayer
        from mindspeed.core.transformer.moe.moe_feature.common import routing_tp_extend_ep
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall_seq':
            if args.moe_tp_extend_ep:
                patch_manager.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing',
                                             routing_tp_extend_ep)
                if not args.moe_alltoall_overlap_comm:
                    patch_manager.register_patch(
                        'megatron.core.transformer.moe.moe_layer.MoELayer',
                        MindSpeedAlltoAllSEQTptoEpMoELayer)
