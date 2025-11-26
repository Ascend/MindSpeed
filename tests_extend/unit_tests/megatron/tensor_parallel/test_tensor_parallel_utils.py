import os
import time
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor
from tests_extend.unit_tests.common import DistributedTest
import megatron.core.parallel_state as Utils
import megatron.core.tensor_parallel.utils as util
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args


class TestParallelStateTP(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)

    @pytest.mark.parametrize("tp", [8, 4, 2])
    def test_split_tensor_along_last_dim(self, tp):
        input_tensor = torch.rand((3, 16))
        torch.equal(input_tensor[0:tp, 0:tp], util.split_tensor_along_last_dim(input_tensor, tp)[0])
        torch.equal(input_tensor[tp:, tp:], util.split_tensor_along_last_dim(input_tensor, tp)[1])
        Utils.destroy_model_parallel()
    
    def test_split_tensor_into_1d_equal_chunks(self):
        tp = 2
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        input_tensor = torch.rand((3, 16))
        output_tensor = util.split_tensor_into_1d_equal_chunks(input_tensor)
        if rank % tp == 0:
            start = 0
            end = int(input_tensor.numel() / tp)
        else:
            start = int(input_tensor.numel() / tp)
            end = input_tensor.numel()
            
        assert torch.equal(output_tensor, input_tensor.flatten()[start:end])
        Utils.destroy_model_parallel()

    def test_gather_split_1d_tensor(self):
        tp = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        rank = int(os.environ['LOCAL_RANK'])
        input_tensor = torch.ones((2, 4)).cuda() * rank
        actual_output_tensor = util.gather_split_1d_tensor(input_tensor)
        if rank % 2 == 0:
            expected_output_tensor = torch.concat((input_tensor.flatten(), input_tensor.flatten() + 1))
        else: 
            expected_output_tensor = torch.concat((input_tensor.flatten() - 1, input_tensor.flatten()))
        assert(torch.equal(actual_output_tensor, expected_output_tensor))
        Utils.destroy_model_parallel()

    def test_vocab(self):
        world_size = 8
        rank = int(os.environ['LOCAL_RANK'])
        global_vocab_size = 1600
        per_partition_vocab_size = 1600 / world_size
        assert((rank * per_partition_vocab_size, (rank + 1) * per_partition_vocab_size) 
            == (util.VocabUtility.vocab_range_from_per_partition_vocab_size(global_vocab_size // world_size, rank, world_size)))
        assert((rank * per_partition_vocab_size, (rank + 1) * per_partition_vocab_size) 
            == (util.VocabUtility.vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size)))
        Utils.destroy_model_parallel()

