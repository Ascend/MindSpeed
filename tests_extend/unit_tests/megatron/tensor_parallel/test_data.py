import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel.data import broadcast_data

import megatron.core.parallel_state as Utils
from unit_tests.common import DistributedTest


class TestTPData(DistributedTest):
    world_size = 8

    args = parse_args(None, True)
    set_args(args)
    
    def test_broadcast_data(self):
        Utils.initialize_model_parallel(2, 4)
        input_data = {
            0: torch.ones((8, 8)).cuda() * 0.0, 
            1: torch.ones((8, 8)).cuda() * 1.0, 
            2: torch.ones((8, 8)).cuda() * 2.0, 
            3: torch.ones((8, 8)).cuda() * 3.0, 
            4: torch.ones((8, 8)).cuda() * 4.0, 
            5: torch.ones((8, 8)).cuda() * 5.0, 
            6: torch.ones((8, 8)).cuda() * 6.0, 
            7: torch.ones((8, 8)).cuda() * 7.0
            }
        dtype = torch.float32
        actual_output = broadcast_data([0, 1], input_data, dtype)
        assert(torch.equal(actual_output[0], input_data[0]))
        assert(torch.equal(actual_output[1], input_data[1]))
        Utils.destroy_model_parallel()
