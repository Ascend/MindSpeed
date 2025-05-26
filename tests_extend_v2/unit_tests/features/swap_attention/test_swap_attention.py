import torch
from mindspeed import megatron_adaptor_v2
import torch_npu

from mindspeed.core.memory.swap_attention.adaptor import AdaptiveRecomputeSwap
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from tests_extend_v2.unit_tests.common import DistributedTest
from tests_extend_v2.commons import initialize_model_parallel


class AdaptiveRecomputePolicy(AdaptiveRecomputeSwap):
    # swap_attention
    def __init__(self):
        super().__init__()
        initialize_model_parallel(1, 1)
        self.pp_rank = 0
        pm.register_patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank',
                          self.get_pipeline_model_parallel_rank_patch, True)
        pm.apply_patches()

    def get_pipeline_model_parallel_rank_patch(self):
        return self.pp_rank


class TestSwapAttention(DistributedTest):
    world_size = 1
    reuse_dist_env = False

    @staticmethod
    def check_result(arp, check_swap, check_prefetch, check_recompute, check_noop):
        prefetch_recompute_group, interval, num_prefetch, swap_noop_layers = arp.solve_prefetch_policy()
        swap_list, prefetch_list, recompute_list = prefetch_recompute_group
        assert swap_list == check_swap
        assert prefetch_list == check_prefetch
        assert recompute_list == check_recompute
        assert swap_noop_layers == check_noop

    @staticmethod
    def config_args(args):
        args.pipeline_model_parallel_size = 1
        args.num_layers = 8
        args.recompute_num_layers = 4
        args.virtual_pipeline_model_parallel_size = 1
        args.enable_recompute_layers_per_pp_rank = False

    def test_storage_copy_interface(self):
        tensor1 = torch.randn([2048, 1, 4096], dtype=torch.bfloat16, device='npu:0')
        tensor_cpu = torch.empty(tensor1.shape, dtype=tensor1.dtype, pin_memory=True, device='cpu')
        tensor_storage_size = tensor1.untyped_storage().size()

        stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(stream):
            stream.wait_stream(torch.npu.current_stream())
            tensor_cpu.untyped_storage().copy_(tensor1.untyped_storage(), non_blocking=True)

        torch.npu.current_stream().wait_stream(stream)
        assert torch.allclose(tensor1.cpu().float().sum(), tensor_cpu.float().sum())

        tensor1.untyped_storage().resize_(0)

        with torch_npu.npu.stream(stream):
            torch.npu.current_stream().wait_stream(stream)
            tensor1.untyped_storage().resize_(tensor_storage_size)
            tensor1.untyped_storage().copy_(tensor_cpu.untyped_storage(), non_blocking=True)

        torch.npu.current_stream().wait_stream(stream)
        assert torch.allclose(tensor1.cpu().float().sum(), tensor_cpu.float().sum())

    def test_swap_attention_cal_prefetch_list(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        self.check_result(arp,
                          [['0', '1', '2', '3', '4', '5', '6', '7']],
                          [['0', '1', '2', '3', '4', '5', '6', '7']],
                          [['0', '1', '2', '3']],
                          [])

    def test_swap_attention_cal_prefetch_list_enable_pp(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.pipeline_model_parallel_size = 2
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        arp.pp_rank = 0
        self.check_result(arp,
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [['0', '1', '2', '3']],
                          [])

    def test_swap_attention_cal_prefetch_list_enable_pp_enable_noop_layers(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.pipeline_model_parallel_size = 2
        args.noop_layers = {0, 7}
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        arp.pp_rank = 0
        self.check_result(arp,
                          [['', '1', '2', '3']],
                          [['', '1', '2', '3']],
                          [['', '1', '2', '3']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1', '2', '']],
                          [['0', '1', '2', '']],
                          [['0', '1', '2', '']],
                          [7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_noop_layers(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 1
        args.virtual_pipeline_model_parallel_size = 4
        args.noop_layers = {0, 7}
        args.enable_recompute_layers_per_pp_rank = True
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        arp.pp_rank = 0
        self.check_result(arp,
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [7])

        args.enable_recompute_layers_per_pp_rank = False
        args.recompute_num_layers = 1
        arp.pp_rank = 0
        self.check_result(arp,
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [[''], ['0'], ['0'], ['0']],
                          [0])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [['0'], ['0'], ['0'], ['']],
                          [7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_multiple_noop_layers(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.pipeline_model_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 2
        args.noop_layers = {0, 1, 6, 7}
        args.enable_recompute_layers_per_pp_rank = True
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        arp.pp_rank = 0
        self.check_result(arp,
                          [['', ''], ['0', '1']],
                          [['', ''], ['0', '1']],
                          [['', ''], ['0', '1']],
                          [0, 1])

        arp.pp_rank = 1
        self.check_result(arp,
                          [['0', '1'], ['', '']],
                          [['0', '1'], ['', '']],
                          [['0', '1'], ['', '']],
                          [6, 7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_multiple_noop_layers_with_inter_layer(self):
        args = parse_args(None, True)
        self.config_args(args)
        args.num_layers = 16
        args.pipeline_model_parallel_size = 4
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 2
        args.noop_layers = {0, 7}
        args.enable_recompute_layers_per_pp_rank = True
        args.world_size = 8
        args.micro_batch_size = 8
        args.reduce_recompute_for_last_chunk = None
        set_args(args)
        arp = AdaptiveRecomputePolicy()
        arp.pp_rank = 0
        self.check_result(arp,
                          [['', '1'], ['0', '1']],
                          [['', '1'], ['0', '1']],
                          [['', '1'], ['0', '1']],
                          [0])

        arp.pp_rank = 3
        self.check_result(arp,
                          [['0', ''], ['0', '1']],
                          [['0', ''], ['0', '1']],
                          [['0', ''], ['0', '1']],
                          [7])
