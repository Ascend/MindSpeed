import torch

from torch.autograd import Variable

from mindspeed import megatron_adaptor
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput


class TestCheckPointFunctionRecomputing:

    def test_checkpoint_function(self):
        def run_function(a, b):
            return a * a + b * b

        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        checkpoint_without_output = CheckpointWithoutOutput(get_cuda_rng_tracker)

        a = Variable(torch.randn(5, 5), requires_grad=True)
        b = Variable(torch.randn(5, 5), requires_grad=True)

        outputs = checkpoint_without_output.checkpoint(run_function, False, a, b)

        expected_outputs = run_function(a, b)
        assert torch.allclose(outputs, expected_outputs)

        checkpoint_without_output.recompute(None)

        outputs.sum().backward()

        a.grad.zero_()
        b.grad.zero_()

        a.retain_grad()
        b.retain_grad()
        recomputed_outputs = run_function(a, b)
        recomputed_outputs.sum().backward()

        assert torch.allclose(a.grad, torch.autograd.grad(expected_outputs.sum(), a)[0])
        assert torch.allclose(b.grad, torch.autograd.grad(expected_outputs.sum(), b)[0])
