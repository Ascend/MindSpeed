import os

import pytest
import torch
import mindspeed.megatron_adaptor_v2
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBucketGroup
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args


@pytest.mark.parametrize('dtype', [torch.float, torch.float16])
@pytest.mark.parametrize('use_distributed_optimizer', [True, False])
class TestOverlapGradReduce(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)

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

        data = torch.zeros(count, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False)
        grad_data = torch.zeros(count, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False)
        data_overlap = data.clone()
        grad_data_overlap = grad_data.clone()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=use_distributed_optimizer,
            check_for_nan_in_grad=False,
        )

        ref_bucket = _ParamAndGradBucket(
            params=params,
            param_data=data,
            grad_data=grad_data,
            offset=torch.cuda.current_device(),
            numel_unpadded=count,
            gradient_scaling_factor=1.0,
            bucket_id=0,
        )

        ref = _ParamAndGradBucketGroup(
            [ref_bucket],
            ddp_config,
            parallel_state.get_data_parallel_group(),
            parallel_state.get_data_parallel_world_size(),
        )
        # test overlap_grad_reduce
        ddp_config.overlap_grad_reduce = True

        overlap_bucket = _ParamAndGradBucket(
            params=params_overlap,
            param_data=data_overlap,
            grad_data=grad_data_overlap,
            offset=torch.cuda.current_device(),
            numel_unpadded=count,
            gradient_scaling_factor=1.0,
            bucket_id=0,
        )

        overlap = _ParamAndGradBucketGroup(
            [overlap_bucket],
            ddp_config,
            parallel_state.get_data_parallel_group(),
            parallel_state.get_data_parallel_world_size(),
        )
        ref.start_grad_sync()

        overlap.start_grad_sync()
        overlap.finish_grad_sync()

        if dtype == torch.bfloat16:
            ref_bucket.param_data = ref_bucket.param_data.float()
            overlap_bucket.param_data = overlap_bucket.param_data.float()

        assert torch.allclose(ref_bucket.param_data, overlap_bucket.param_data, rtol=0.0001, atol=0.0001), '{}\n{}'.format(
            ref_bucket.param_data, overlap_bucket.param_data)
