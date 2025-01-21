import os
import unittest
from unittest.mock import patch
import torch.distributed
from mindspeed import megatron_adaptor

from mindspeed.core.memory.adaptive_memory.adaptive_memory_solver import AdaptMemGraphSolver
from mindspeed.core.memory.adaptive_memory.adaptive_memory_tool import ModuleAction, FuncLocation
from mindspeed.core.memory.adaptive_memory.adaptive_memory_cache import AdaptiveLayerMemPolicy
from mindspeed.core.memory.adaptive_memory.adaptive_memory_policy import AdaptMemPolicyManager
from megatron.core import parallel_state as ps
from megatron.training import print_rank_0
from megatron.core.num_microbatches_calculator import get_num_microbatches, init_num_microbatches_calculator
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args


def init_args():
    args = parse_args(None, True)
    set_args(args)
    args.num_layers = 24
    args.global_batch_size = 16
    args.micro_batch_size = 2
    args.data_parallel_size = 1
    try:
        get_num_microbatches()
    except Exception as e:
        init_num_microbatches_calculator(0, None, args.global_batch_size, args.micro_batch_size,
                                         args.data_parallel_size)
    return args


parallels = []


def mock_pp_size():
    return parallels[0]


def mock_vpp_size():
    return parallels[1]


def mock_pp_rank():
    return parallels[2]


def mock_function(pp, vpp, rank):
    parallels.clear()
    parallels.extend([pp, vpp, rank])
    old_pp_func = ps.get_pipeline_model_parallel_world_size
    old_vpp_func = ps.get_virtual_pipeline_model_parallel_world_size
    old_pp_rank = ps.get_pipeline_model_parallel_rank
    ps.get_pipeline_model_parallel_world_size = mock_pp_size
    ps.get_virtual_pipeline_model_parallel_world_size = mock_vpp_size
    ps.get_pipeline_model_parallel_rank = mock_pp_rank
    return old_pp_func, old_vpp_func, old_pp_rank


def reset_function(old_pp_func, old_vpp_func, old_pp_rank):
    ps.get_pipeline_model_parallel_world_size = old_pp_func
    ps.get_virtual_pipeline_model_parallel_world_size = old_vpp_func
    ps.get_pipeline_model_parallel_rank = old_pp_rank


class TestAdaptMemGraphSolver:

    def test_func_location_with_pp_vpp(self):
        init_args()
        f1, f2, f3 = mock_function(4, 3, 0)

        solver = AdaptMemGraphSolver()
        solver.func_locations.clear()
        solver.func_locations.append(FuncLocation(0, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(1, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(2, "PipeExpert", ModuleAction.SWAP))
        solver.func_locations.append(FuncLocation(3, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(4, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(5, "PipeExpert", ModuleAction.NONE))

        # test forward cnt in first chunk
        for i in range(8):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.NONE

        # test forward cnt in second chunk
        for i in range(8, 16, 2):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.SWAP
        for i in range(9, 16, 2):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.NONE

        # test forward cnt in last chunk
        for i in range(16, 32):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.NONE

        reset_function(f1, f2, f3)

    def test_func_location_with_pp(self):
        init_args()
        f1, f2, f3 = mock_function(4, 1, 0)

        solver = AdaptMemGraphSolver()
        solver.func_locations.clear()
        solver.func_locations.append(FuncLocation(0, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(1, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(2, "PipeExpert", ModuleAction.SWAP))
        solver.func_locations.append(FuncLocation(3, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(4, "PipeExpert", ModuleAction.NONE))
        solver.func_locations.append(FuncLocation(5, "PipeExpert", ModuleAction.NONE))
        # test unmatched
        for i in range(0, 100, 6):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.NONE

        # test matched
        for i in range(2, 100, 6):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.SWAP

        reset_function(f1, f2, f3)

    def test_func_location_without_pp_vpp(self):
        args = init_args()
        f1, f2, f3 = mock_function(1, None, 0)

        solver = AdaptMemGraphSolver()
        solver.func_locations.clear()
        num_layers = args.num_layers
        for i in range(num_layers):
            if i != 2:
                solver.func_locations.append(FuncLocation(i, "PipeExpert", ModuleAction.NONE))
            else:
                solver.func_locations.append(FuncLocation(2, "PipeExpert", ModuleAction.SWAP))
        # test unmatched
        for i in range(0, 100, 24):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.NONE

        # test matched
        for i in range(2, 100, 24):
            assert solver.get_func_action("PipeExpert", i) == ModuleAction.SWAP

        reset_function(f1, f2, f3)


def mock_init_args(args):
    args.num_layers = 12
    args.micro_batch_size = 2
    args.global_batch_size = 16
    args.data_parallel_size = 1
    try:
        get_num_microbatches()
    except Exception as e:
        init_num_microbatches_calculator(0, None, args.global_batch_size, args.micro_batch_size,
                                         args.data_parallel_size)


def mock_get_vpp_1():
    return 1


def mock_get_vpp_3():
    return 3


def mock_get_pp_1():
    return 1


def mock_get_pp_2():
    return 2


def mock_get_pp_rank():
    return 0


def mock_layer_per_chunk_6():
    return 6


def mock_layer_per_chunk_2():
    return 2


def mock_layer_num_per_chunk_12():
    return 12


class TestPolicySelection(unittest.TestCase):
    test_memory_policy_combinations = [
        {
            "recompute": [],
            "swap": [
                "0"
            ],
            "memory": 800,
            "time": 80,
            "adapt_type": 1
        },
        {
            "recompute": [
                "0"
            ],
            "swap": [],
            "memory": 500,
            "time": 100,
            "adapt_type": 0
        },
        {
            "recompute": [],
            "swap": [],
            "memory": 1000,
            "time": 50,
            "adapt_type": 3
        },
        {
            "recompute": [
                "0.input_norm"
            ],
            "swap": [],
            "memory": 550,
            "time": 95,
            "adapt_type": 3
        },
        {
            "recompute": [],
            "swap": [
                "0.input_norm"
            ],
            "memory": 600,
            "time": 90,
            "adapt_type": 3
        },
        {
            "recompute": [
                "0.self_attention"
            ],
            "swap": [
                "0.input_norm"
            ],
            "memory": 650,
            "time": 85,
            "adapt_type": 3
        },
        {
            "recompute": [
                "0.self_attention"
            ],
            "swap": [],
            "memory": 700,
            "time": 80,
            "adapt_type": 3
        },
        {
            "recompute": [
                "0.input_norm"
            ],
            "swap": [
                "0.self_attention"
            ],
            "memory": 750,
            "time": 75,
            "adapt_type": 3
        },
        {
            "recompute": [],
            "swap": [
                "0.self_attention"
            ],
            "memory": 850,
            "time": 70,
            "adapt_type": 3
        }
    ]

    def common_mem_select_policy_function(self, device_memory, args):
        mock_init_args(args)
        manager = AdaptMemPolicyManager()
        manager.full_recompute_comb = AdaptiveLayerMemPolicy(recompute=["0"], swap=[],
                                                             memory=500,
                                                             time=100,
                                                             adapt_type=ModuleAction.RECOMPUTE)
        manager.full_swap_comb = AdaptiveLayerMemPolicy(recompute=[], swap=["0"],
                                                        memory=800,
                                                        time=80, adapt_type=ModuleAction.SWAP)
        manager.without_adaptive_comb = AdaptiveLayerMemPolicy(recompute=[], swap=[],
                                                               memory=1000, time=500,
                                                               adapt_type=ModuleAction.NONE)
        manager.policy_combinations.clear()
        for v in self.test_memory_policy_combinations:
            temp_single_combianation = AdaptiveLayerMemPolicy(recompute=v["recompute"],
                                                              swap=v["swap"],
                                                              memory=v["memory"],
                                                              time=v["time"],
                                                              adapt_type=v["adapt_type"])
            manager.policy_combinations.append(temp_single_combianation)
        manager.adapt_modules_num = 2
        solver = AdaptMemGraphSolver()
        solver.adapt_mem_policy.clear()
        solver.knapsack_best(device_memory)
        adapt_mem_policy_list = solver.get_adapt_mem_policy_list()
        del solver
        print_rank_0(f"adapt_mem_policy_list:{adapt_mem_policy_list}")
        return adapt_mem_policy_list

    # 用例1：pp = 1(未开启pp), vpp = 1(未开启vpp)，device_memory非常大
    # adapt_mem_policy_list预期结果：
    # [[12, <LayerAction.NONE: 3>, <ModuleAction.NONE: 2>, <ModuleAction.NONE: 2>]]
    @patch.object(AdaptMemGraphSolver, 'get_layer_num_per_chunk', side_effect=mock_layer_num_per_chunk_12)
    @patch('megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size',
           side_effect=mock_get_vpp_1)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', side_effect=mock_get_pp_1)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank', side_effect=mock_get_pp_rank)
    def test_select_mem_policy_without_pp_large_device_memory(self, mock_method1, mock_method2, mock_method3,
                                                              mock_method4):
        device_memory = 999999
        args = parse_args(None, True)
        set_args(args)
        args.virtual_pipeline_model_parallel_size = 1
        adapt_mem_policy_list = self.common_mem_select_policy_function(device_memory, args)
        self.assertEqual(adapt_mem_policy_list, [[12, 3, 2, 2]])

    # 用例2：pp = 2, vpp = 1(未开启vpp)，device_memory非常大
    # adapt_mem_policy_list预期结果：
    # [[6, <LayerAction.NONE: 3>, <ModuleAction.NONE: 2>, <ModuleAction.NONE: 2>]]
    @patch.object(AdaptMemGraphSolver, 'get_layer_num_per_chunk', side_effect=mock_layer_per_chunk_6)
    @patch('megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size',
           side_effect=mock_get_vpp_1)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', side_effect=mock_get_pp_2)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank', side_effect=mock_get_pp_rank)
    def test_select_mem_policy_without_vpp_large_device_memory(self, mock_method1, mock_method2, mock_method3,
                                                               mock_method4):
        device_memory = 999999
        args = parse_args(None, True)
        set_args(args)
        args.virtual_pipeline_model_parallel_size = 1
        adapt_mem_policy_list = self.common_mem_select_policy_function(device_memory, args)
        self.assertEqual(adapt_mem_policy_list, [[6, 3, 2, 2]])

    # 用例3：pp = 2, vpp = 1(未开启vpp)，device_memory非常小
    # adapt_mem_policy_list预期结果：
    # [[6, <LayerAction.FULL_RECOMPUTE: 0>, <ModuleAction.RECOMPUTE: 0>, <ModuleAction.RECOMPUTE: 0>]]
    @patch.object(AdaptMemGraphSolver, 'get_layer_num_per_chunk', side_effect=mock_layer_per_chunk_6)
    @patch('megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size',
           side_effect=mock_get_vpp_1)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', side_effect=mock_get_pp_2)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank', side_effect=mock_get_pp_rank)
    def test_select_mem_policy_without_vpp_little_device_memory(self, mock_method1, mock_method2, mock_method3,
                                                                mock_method4):
        device_memory = 1000
        args = parse_args(None, True)
        set_args(args)
        args.virtual_pipeline_model_parallel_size = 1
        adapt_mem_policy_list = self.common_mem_select_policy_function(device_memory, args)
        self.assertEqual(adapt_mem_policy_list, [[6, 0, 0, 0]])

    # 用例4：pp = 2, vpp = 3，device_memory非常大
    # adapt_mem_policy_list预期结果：
    # [[6, <LayerAction.NONE: 3>, <ModuleAction.NONE: 2>, <ModuleAction.NONE: 2>]]
    @patch.object(AdaptMemGraphSolver, 'get_layer_num_per_chunk', side_effect=mock_layer_per_chunk_2)
    @patch('megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size',
           side_effect=mock_get_vpp_3)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', side_effect=mock_get_pp_2)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank', side_effect=mock_get_pp_rank)
    def test_select_mem_policy_with_vpp_large_device_memory(self, mock_method1, mock_method2, mock_method3,
                                                            mock_method4):
        device_memory = 999999
        args = parse_args(None, True)
        set_args(args)
        args.virtual_pipeline_model_parallel_size = 3
        args.num_layers_per_virtual_pipeline_stage = 12 // (args.virtual_pipeline_model_parallel_size * 2)
        adapt_mem_policy_list = self.common_mem_select_policy_function(device_memory, args)
        self.assertEqual(adapt_mem_policy_list, [[6, 3, 2, 2]])

    # 用例5：pp = 2, vpp = 3，device_memory非常小
    # adapt_mem_policy_list预期结果：
    # [[6, <LayerAction.FULL_RECOMPUTE: 0>, <ModuleAction.RECOMPUTE: 0>, <ModuleAction.RECOMPUTE: 0>]]
    @patch.object(AdaptMemGraphSolver, 'get_layer_num_per_chunk', side_effect=mock_layer_per_chunk_2)
    @patch('megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size',
           side_effect=mock_get_vpp_3)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_world_size', side_effect=mock_get_pp_2)
    @patch('megatron.core.parallel_state.get_pipeline_model_parallel_rank', side_effect=mock_get_pp_rank)
    def test_select_mem_policy_with_vpp_little_device_memory(self, mock_method1, mock_method2, mock_method3,
                                                             mock_method4):
        device_memory = 1000
        args = parse_args(None, True)
        set_args(args)
        args.virtual_pipeline_model_parallel_size = 3
        args.num_layers_per_virtual_pipeline_stage = 12 // (args.virtual_pipeline_model_parallel_size * 2)
        adapt_mem_policy_list = self.common_mem_select_policy_function(device_memory, args)
        self.assertEqual(adapt_mem_policy_list, [[6, 0, 0, 0]])
