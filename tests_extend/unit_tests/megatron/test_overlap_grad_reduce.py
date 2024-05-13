import os
import pytest
import torch
import ascendspeed.megatron_adaptor
from megatron.core.distributed.grad_buffer import Bucket
from megatron.core import parallel_state
from unit_tests.common import DistributedTest
from commons import initialize_model_parallel


@pytest.mark.parametrize('dtype', [torch.float, torch.float16])
@pytest.mark.parametrize('use_distributed_optimizer', [True, False])
class TestOverlapGradReduce(DistributedTest):
    world_size = 8

    def test_overlap_grad_reduce(self, dtype, use_distributed_optimizer):
        param_size = [8, 8]
        os.environ['HCCL_DETERMINISTIC'] = 'True'

        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        params = []
        params_overlap = []
        count = 0
        for i in range(parallel_state.get_data_parallel_world_size()):
            tmp = torch.randn(param_size, dtype=dtype).cuda()
            count += tmp.numel()
            params.append(torch.nn.Parameter(tmp))
            params_overlap.append(torch.nn.Parameter(tmp.clone()))

        data = torch.randn(count, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False)
        data_overlap = data.clone()

        ref = Bucket(
            params=params,
            data=data,
            offset=torch.cuda.current_device(),
            data_parallel_group=parallel_state.get_data_parallel_group(),
            data_parallel_world_size=parallel_state.get_data_parallel_world_size(),
            overlap_grad_reduce=False,
            use_distributed_optimizer=use_distributed_optimizer,
        )

        overlap = Bucket(
            params=params_overlap,
            data=data_overlap,
            offset=torch.cuda.current_device(),
            data_parallel_group=parallel_state.get_data_parallel_group(),
            data_parallel_world_size=parallel_state.get_data_parallel_world_size(),
            overlap_grad_reduce=True,
            use_distributed_optimizer=use_distributed_optimizer,
        )

        ref.start_grad_sync()

        overlap.start_grad_sync()
        overlap.finish_grad_sync()

        if dtype == torch.bfloat16:
            ref.data = ref.data.float()
            overlap.data = overlap.data.float()

        assert torch.allclose(ref.data, overlap.data, rtol=0.0001, atol=0.0001), '{}\n{}'.format(ref.data, overlap.data)
