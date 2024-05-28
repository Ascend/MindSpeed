from mindspeed import megatron_adaptor
from unittest import mock
import mindspeed.core.memory.adaptive_recomputing.adaptive_recompute
from megatron.core import parallel_state
from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import get_adaptive_recomputing
from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import is_activate_adaptive_recompute
from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import get_adaptive_recomputing_policy

from unit_tests.common import DistributedTest


class MockArgs:
    mock_args = None

    def __init__(self):
        self.checkpoint_activations = True
        self.adaptive_recompute_device_size = -1
        self.adaptive_recompute_profiling_step = 10
        self.adaptive_recompute_device_swap = False
        self.train_iters = 10
        self.recompute_granularity = None
        self.recompute_method = None


def set_mock_args(arg_checkpoint: bool = True,
                  arg_device_swap: bool = False,
                  arg_device_size: int = -1,
                  arg_profiling_step: int = 10):
    if MockArgs.mock_args is None:
        MockArgs.mock_args = MockArgs()

    MockArgs.mock_args.checkpoint_activations = arg_checkpoint
    MockArgs.mock_args.adaptive_recompute_device_size = arg_device_size
    MockArgs.mock_args.adaptive_recompute_profiling_step = arg_profiling_step
    MockArgs.mock_args.adaptive_recompute_device_swap = arg_device_swap

    return MockArgs.mock_args


class TestRecomputing(DistributedTest):
    world_size = 1

    def test_check_recompute_enable(self):
        recomputing = get_adaptive_recomputing()

        set_mock_args(True, False, -1, 10)
        with mock.patch.object(mindspeed.core.memory.adaptive_recomputing.adaptive_recompute, 'get_args',
                               return_value=MockArgs.mock_args):
            result, profile_step = is_activate_adaptive_recompute()
            assert result is False
        set_mock_args(False, True, 10, 0)
        with mock.patch.object(mindspeed.core.memory.adaptive_recomputing.adaptive_recompute, 'get_args',
                               return_value=MockArgs.mock_args):
            result, profile_step = is_activate_adaptive_recompute()
            assert result is True
        set_mock_args()

    def test_step_hook(self):
        with mock.patch.object(mindspeed.core.memory.adaptive_recomputing.adaptive_recompute, 'get_args',
                               return_value=MockArgs()):
            with mock.patch.object(parallel_state, 'get_tensor_model_parallel_world_size',
                                   return_value=0):
                with mock.patch.object(parallel_state, 'get_data_parallel_world_size',
                                       return_value=0):
                    models = None
                    recomputing = get_adaptive_recomputing()

                    # 初始化context
                    memory_info = recomputing.get_memory_status()
                    hccl_memory = (memory_info["all_memory"] - memory_info["free"] - memory_info["reserved_memory"])
                    get_adaptive_recomputing_policy().hccl_memory = max(hccl_memory,
                                                                        get_adaptive_recomputing_policy().hccl_memory)
                    recomputing.context['used_mem'] = memory_info["used_memory"]
                    recomputing.context['max_device_memory'] = memory_info["all_memory"]

                    recomputing.set_profiling_step(10)

                    recomputing.profiling_step = 0
                    recomputing.step_hook(models)
                    assert get_adaptive_recomputing_policy().context_copy is None

                    recomputing.profiling_step = 12
                    recomputing.step_hook(models)
                    assert get_adaptive_recomputing_policy().context_copy is None

                    recomputing.profiling_step = 11
                    with mock.patch.object(
                            mindspeed.core.memory.adaptive_recomputing.adaptive_recompute.AdaptiveRecomputePolicy,
                            'solve_recompute_policy', return_value=recomputing.context):
                        with mock.patch.object(mindspeed.core.memory.adaptive_recomputing.adaptive_recompute,
                                               'apply_adaptive_recompute', return_value=None):
                            with mock.patch.object(
                                    mindspeed.core.memory.adaptive_recomputing.adaptive_recompute_solver.GraphSolver,
                                    'build_solver_info', return_value=None):
                                recomputing.step_hook(models)
                                assert get_adaptive_recomputing_policy().context_copy is not None
