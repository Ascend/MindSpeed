import os
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel import mappings

import megatron.core.parallel_state as Utils
from unit_tests.common import DistributedTest


class TestTPMapping(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)
    Utils.world_size = 8
    
    def test_CopyToModelParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.ones((1)).cuda() * rank
        output_data = mappings._CopyToModelParallelRegion.backward(None, input_data)
        result = torch.ones(1).cuda()
        result = result * 22 if rank >= 4 else result * 6
        assert(torch.equal(output_data, result))
        assert(torch.equal(input_data, mappings.copy_to_tensor_model_parallel_region(input_data)))
        assert(torch.equal(input_data, mappings._CopyToModelParallelRegion.symbolic(None, input_data)))
        Utils.destroy_model_parallel()
    
    def test_ReduceFromModelParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.ones((1)).cuda() * rank
        output_data = mappings._ReduceFromModelParallelRegion.symbolic(None, input_data)
        result = torch.ones(1).cuda()
        result = result * 22 if rank >= 4 else result * 6
        assert(torch.equal(output_data, result))
        input_data = torch.ones((1)).cuda() * rank
        assert(torch.equal(mappings.reduce_from_tensor_model_parallel_region(input_data), result))
        assert(torch.equal(input_data, mappings._ReduceFromModelParallelRegion.backward(None, input_data)))
        Utils.destroy_model_parallel()
    
    def test_ScatterToModelParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.rand((8, 4)).cuda()
        output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)
        req_dim = int(rank % (Utils.world_size / 2))
        assert(torch.equal(output_data, input_data[:, req_dim].reshape((8, 1))))
        output_data = mappings._ScatterToModelParallelRegion.symbolic(None, input_data)
        assert(torch.equal(output_data, input_data[:, req_dim].reshape((8, 1))))
    
        input_data = torch.ones(8).cuda() * rank
        actual_output_data = mappings._ScatterToModelParallelRegion.backward(None, input_data)
        expected_output = torch.cat((
            torch.ones(8) * 0, 
            torch.ones(8) * 1, 
            torch.ones(8) * 2, 
            torch.ones(8) * 3)).cuda()
        if (rank >= 4):
            expected_output = expected_output + 4
        assert(torch.equal(actual_output_data, expected_output))
        Utils.destroy_model_parallel()
    
    def test_GatherFromModelParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.rand((8, 4)).cuda()
        req_dim = int(rank % (Utils.world_size / 2))
        output_data = mappings._GatherFromModelParallelRegion.backward(None, input_data)
        assert(torch.equal(output_data, input_data[:, req_dim].reshape((8, 1))))
        input_data = torch.ones(8).cuda() * rank
        actual_output_data = mappings.gather_from_tensor_model_parallel_region(input_data)
        expected_output = torch.cat((
            torch.ones(8) * 0, 
            torch.ones(8) * 1, 
            torch.ones(8) * 2, 
            torch.ones(8) * 3)).cuda()
        if (rank >= 4):
            expected_output = expected_output + 4
        assert(torch.equal(actual_output_data, expected_output))
        assert(torch.equal(mappings._GatherFromModelParallelRegion.symbolic(None, input_data), expected_output))
        Utils.destroy_model_parallel()
     
    def test_ScatterToSequenceParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.rand((8, 4)).cuda()
        req_dim = int(rank % (Utils.world_size / 2)) * 2
        output_data = mappings._ScatterToSequenceParallelRegion.symbolic(None, input_data)
        assert(torch.equal(output_data, input_data[req_dim:req_dim + 2, :]))
        output_data = mappings.scatter_to_sequence_parallel_region(input_data)
        assert(torch.equal(output_data, input_data[req_dim:req_dim + 2, :]))
        input_data = torch.ones(4).cuda() * rank
        output_data = mappings._ScatterToModelParallelRegion.backward(None, input_data)
        expected_output = torch.concat((
            torch.ones(4) * 0, 
            torch.ones(4) * 1, 
            torch.ones(4) * 2, 
            torch.ones(4) * 3)).cuda()
        if (rank >= 4):
            expected_output = expected_output + 4
        assert(torch.equal(output_data, expected_output))
        Utils.destroy_model_parallel()
    
    def test_GatherFromSequenceParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.ones(4).cuda() * rank
        output_data = mappings.gather_from_sequence_parallel_region(input_data)
        expected_output = torch.concat((
            torch.ones(4) * 0, 
            torch.ones(4) * 1, 
            torch.ones(4) * 2, 
            torch.ones(4) * 3)).cuda()
        if (rank >= 4):
            expected_output = expected_output + 4
        assert(torch.equal(output_data, expected_output))
        assert(torch.equal(mappings._GatherFromSequenceParallelRegion.symbolic(None, input_data), expected_output))
        input_data = torch.vstack((
            torch.ones(4) * 0, 
            torch.ones(4) * 1, 
            torch.ones(4) * 2, 
            torch.ones(4) * 3)).cuda()

        class Ctx:
            tensor_parallel_output_grad = True
        output_data = mappings._GatherFromSequenceParallelRegion.backward(Ctx(), input_data)
        expected_output = torch.ones((1, 4)).cuda() * 4 * int(rank % 4)
        assert(torch.equal(output_data[0], expected_output))
        Utils.destroy_model_parallel()
    
    def test_ReduceScatterToSequenceParallelRegion(self):
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(4, 2)
        input_data = torch.vstack((
            torch.ones(4) * 0, 
            torch.ones(4) * 1, 
            torch.ones(4) * 2, 
            torch.ones(4) * 3)).cuda()
        output_data = mappings.reduce_scatter_to_sequence_parallel_region(input_data)
        expected_output = torch.ones(4).cuda() * 4 * int(rank % 4)
        assert(torch.equal(output_data[0], expected_output))
        assert(torch.equal(mappings._ReduceScatterToSequenceParallelRegion.symbolic(None, input_data), expected_output.reshape((1, 4))))
        input_data = torch.ones(4).cuda() * rank
        output_data = mappings._ReduceScatterToSequenceParallelRegion.backward(None, input_data)
        expected_output = torch.concat((
            torch.ones(4) * 0, 
            torch.ones(4) * 1, 
            torch.ones(4) * 2, 
            torch.ones(4) * 3)).cuda()
        if (rank >= 4):
            expected_output = expected_output + 4
        assert(torch.equal(output_data, expected_output))
        Utils.destroy_model_parallel()
    
