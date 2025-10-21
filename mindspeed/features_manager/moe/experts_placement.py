from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class ExpertsPlacementFeature(MindSpeedFeature):
    '''
    MoE Expert load replacement 
    '''
    def __init__(self):
        super().__init__('enable-expert-placement')
        
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-expert-placement', action='store_true',
        help='balance the experts load within ep group.')
        group.add_argument('--expert-placement-freq', type=int, default=50,
            help='define the fix frequency of expert placement')
        group.add_argument('--enable-fine-grained-expert-placement', action='store_true',
            help='balance the experts load within ep group.')
        group.add_argument("--print-expert-load", action='store_true',
            help="print tokens of each expert in each layer of each stage in dp 0")
        group.add_argument('--fine-grained-expert-placement-thre', type=float, default=0.08,
            help='define the fix frequency of expert placement')
    
    def validate_args(self, args):
        # expert placement check
        if getattr(args, "enable_expert_placement", False):
            if not args.use_distributed_optimizer:
                print('--enable_expert_placement only supported with distributed optimizer')
            if not hasattr(args, "expert_model_parallel_size") and args.expert_model_parallel_size > 1:
                print('--enable_expert_placement only supported with expert_model_parallel_size larger than 1')
            if not hasattr(args, "expert_placement_freq") and args.expert_placement_freq > 1:
                print('--enable_expert_placement only supported with expert_placement_freq larger than 1')
            if args.moe_extended_tp:
                print('--enable_expert_placement not supported with moe_extended_tp')
        if args.enable_fine_grained_expert_placement:
            if not getattr(args, "enable_expert_placement", False):
                print('--enable_fine_grained_expert_placement only supported with enable_expert_placement')
            if not (hasattr(args, "fine_grained_expert_placement_thre") 
                    and args.fine_grained_expert_placement_thre > 0):
                print('--enable_fine_grained_expert_placement requires a expert placement trigger threshold')

    def register_patches(self, patch_manager, args):
        if getattr(args, "enable_expert_placement", False) or getattr(args, "print_expert_load", False): 
            from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.moe_layer import MindSpeedFbOverlapMoELayer
            from mindspeed.core.transformer.moe.expert_placement.planner import expert_placement_init, predict_expert_load
            MindSpeedFbOverlapMoELayer.expert_placement_init = expert_placement_init
            MindSpeedFbOverlapMoELayer.predict_expert_load = predict_expert_load
            

