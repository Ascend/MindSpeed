from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from mindspeed.core.memory.swap_attention import adaptor as swap_attention_adaptor
from mindspeed.core.memory.swap_attention.adaptor import AdaptiveRecomputeSwap
from tests_extend.unit_tests.common import DistributedTest


class AdaptiveRecomputePolicy(AdaptiveRecomputeSwap):
    """Exercise the production policy without initializing Megatron globals."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pp_rank = 0
        self.is_last_stage = False

    def solve_prefetch_policy(self):
        with (
            patch.object(swap_attention_adaptor, 'get_args', return_value=self.args),
            patch.object(
                swap_attention_adaptor.parallel_state,
                'get_pipeline_model_parallel_rank',
                side_effect=lambda: self.pp_rank,
            ),
            patch.object(
                swap_attention_adaptor.parallel_state,
                'is_pipeline_last_stage',
                side_effect=lambda **_: self.is_last_stage,
            ),
        ):
            return super().solve_prefetch_policy()


class TestSwapAttentionStorage(DistributedTest):
    world_size = 1
    reuse_dist_env = False

    @pytest.mark.slow
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


class TestSwapAttentionPolicy:
    @staticmethod
    def check_result(arp, check_swap, check_prefetch, check_recompute, check_noop):
        prefetch_recompute_group, interval, num_prefetch, swap_noop_layers = arp.solve_prefetch_policy()
        swap_list, prefetch_list, recompute_list = prefetch_recompute_group
        assert swap_list == check_swap
        assert prefetch_list == check_prefetch
        assert recompute_list == check_recompute
        assert swap_noop_layers == check_noop

    @staticmethod
    def config_args():
        return SimpleNamespace(
            pipeline_model_parallel_size=1,
            num_layers=8,
            recompute_num_layers=4,
            virtual_pipeline_model_parallel_size=1,
            num_layers_per_virtual_pipeline_stage=None,
            enable_recompute_layers_per_pp_rank=False,
            recompute_method=None,
            reduce_recompute_for_last_chunk=None,
            noop_layers=None,
        )

    def test_swap_attention_cal_prefetch_list(self):
        args = self.config_args()
        arp = AdaptiveRecomputePolicy(args)
        self.check_result(
            arp,
            [['0', '1', '2', '3', '4', '5', '6', '7']],
            [['0', '1', '2', '3', '4', '5', '6', '7']],
            [['0', '1', '2', '3']],
            [],
        )

    def test_swap_attention_cal_prefetch_list_enable_pp(self):
        args = self.config_args()
        args.pipeline_model_parallel_size = 2
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp, [['0', '1', '2', '3']], [['0', '1', '2', '3']], [['0', '1', '2', '3']], [])

        arp.pp_rank = 1
        self.check_result(arp, [['0', '1', '2', '3']], [['0', '1', '2', '3']], [['0', '1', '2', '3']], [])

    def test_swap_attention_cal_prefetch_list_enable_pp_enable_noop_layers(self):
        args = self.config_args()
        args.pipeline_model_parallel_size = 2
        args.noop_layers = {0, 7}
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp, [['', '1', '2', '3']], [['', '1', '2', '3']], [['', '1', '2', '3']], [0])

        arp.pp_rank = 1
        self.check_result(arp, [['0', '1', '2', '']], [['0', '1', '2', '']], [['0', '1', '2', '']], [7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_noop_layers(self):
        args = self.config_args()
        args.pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 1
        args.virtual_pipeline_model_parallel_size = 4
        args.noop_layers = {0, 7}
        args.enable_recompute_layers_per_pp_rank = True
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(
            arp, [[''], ['0'], ['0'], ['0']], [[''], ['0'], ['0'], ['0']], [[''], ['0'], ['0'], ['0']], [0]
        )

        arp.pp_rank = 1
        self.check_result(
            arp, [['0'], ['0'], ['0'], ['']], [['0'], ['0'], ['0'], ['']], [['0'], ['0'], ['0'], ['']], [7]
        )

        args.enable_recompute_layers_per_pp_rank = False
        args.recompute_num_layers = 1
        arp.pp_rank = 0
        self.check_result(
            arp, [[''], ['0'], ['0'], ['0']], [[''], ['0'], ['0'], ['0']], [[''], ['0'], ['0'], ['0']], [0]
        )

        arp.pp_rank = 1
        self.check_result(
            arp, [['0'], ['0'], ['0'], ['']], [['0'], ['0'], ['0'], ['']], [['0'], ['0'], ['0'], ['']], [7]
        )

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_multiple_noop_layers(self):
        args = self.config_args()
        args.pipeline_model_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 2
        args.noop_layers = {0, 1, 6, 7}
        args.enable_recompute_layers_per_pp_rank = True
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp, [['', ''], ['0', '1']], [['', ''], ['0', '1']], [['', ''], ['0', '1']], [0, 1])

        arp.pp_rank = 1
        self.check_result(arp, [['0', '1'], ['', '']], [['0', '1'], ['', '']], [['0', '1'], ['', '']], [6, 7])

    def test_swap_attention_cal_prefetch_list_enable_vpp_enable_multiple_noop_layers_with_inter_layer(self):
        args = self.config_args()
        args.num_layers = 16
        args.pipeline_model_parallel_size = 4
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = 2
        args.noop_layers = {0, 7}
        args.enable_recompute_layers_per_pp_rank = True
        arp = AdaptiveRecomputePolicy(args)
        arp.pp_rank = 0
        self.check_result(arp, [['', '1'], ['0', '1']], [['', '1'], ['0', '1']], [['', '1'], ['0', '1']], [0])

        arp.pp_rank = 3
        self.check_result(arp, [['0', ''], ['0', '1']], [['0', ''], ['0', '1']], [['0', ''], ['0', '1']], [7])
