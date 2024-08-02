import os
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor
import megatron.core.parallel_state as ps
from unit_tests.common import DistributedTest
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args


class TestParallelStateEPCP(DistributedTest):
    world_size = 8

    args = parse_args(None, True)
    set_args(args)

    def test_initialize_and_destroy_model_parallel(self):
        with pytest.raises(RuntimeError):
            assert (ps.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, context_parallel_size=3))

        try:
            ps.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2, expert_model_parallel_size=4)
        except RuntimeError as exc:
            assert False, "dp * cp % ep == 0, which should not raise a RuntimeError"
        else:
            assert (ps.model_parallel_is_initialized())
            assert (ps.get_model_parallel_group() is not None)
            assert (ps.get_tensor_model_parallel_group() is not None)
            assert (ps.get_pipeline_model_parallel_group() is not None)
            assert (ps.get_data_parallel_group() is not None)
            assert (ps.get_expert_model_parallel_group() is not None)
            assert (ps.get_tensor_and_expert_parallel_group() is not None)
            assert (ps.get_data_modulo_expert_parallel_group() is not None)
        finally:
            ps.destroy_model_parallel()
            assert (ps._MODEL_PARALLEL_GROUP is None)
            assert (ps._MPU_EXPERT_MODEL_PARALLEL_RANK is None)
            assert (ps._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is None)


    @pytest.mark.parametrize("parallelism_config", [(2, 2, 2, 2), (2, 4, 1, 4), (1, 2, 4, 1)])
    def test_data_modulo_expert_parallel_initializations(self, parallelism_config):
        tp, cp, dp, ep = parallelism_config
        assert (dp * cp % ep == 0)
        rank = int(os.environ['LOCAL_RANK'])
        ps.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp, expert_model_parallel_size=ep)
        assert (torch.distributed.get_world_size(group=ps.get_data_modulo_expert_parallel_group()) == cp * dp // ep)
        assert (ps.get_data_modulo_expert_parallel_rank() == rank // (tp * ep))
        ps.destroy_model_parallel()


    @pytest.mark.parametrize("parallelism_config", [(2, 2, 2, 2), (2, 4, 1, 4), (1, 2, 4, 1)])
    def test_expert_model_parallel_world_size(self, parallelism_config):
        tp, cp, dp, ep = parallelism_config
        assert (dp * cp % ep == 0)
        ps.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp, expert_model_parallel_size=ep)
        assert (ps.get_expert_model_parallel_world_size() == ep)
        ps.destroy_model_parallel()


    @pytest.mark.parametrize("parallelism_config", [(2, 2, 2, 2), (2, 4, 1, 4), (1, 2, 4, 1)])
    def test_expert_model_parallel_rank(self, parallelism_config):
        tp, cp, dp, ep = parallelism_config
        assert (dp * cp % ep == 0)
        rank = int(os.environ['LOCAL_RANK'])
        ps.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp, expert_model_parallel_size=ep)
        assert (ps.get_expert_model_parallel_rank() == rank % (tp * ep) // tp)
        ps.destroy_model_parallel()