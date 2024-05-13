import pytest
import torch
import torch.nn.functional as F
import torch_npu
from ascendspeed import megatron_adaptor

from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest

from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel import mappings


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


class TestTensorParallelCrossEntropy(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("tp_pp", [(1, 1), (2, 4)])
    def test_cross_entropy(self, tp_pp):
        (tp, pp) = tp_pp
        initialize_model_parallel(tp, pp)
        self.process_cross_entropy(tp)

    def process_cross_entropy(self, tensor_model_parallel_size=2):
        batch_size = 10
        seq_length = 15
        vocab_size_per_partition = 8
        logits_scale = 1000.0
        vocab_size = vocab_size_per_partition * tensor_model_parallel_size
        seed = 3123

        loss_torch, grad_torch = self.torch_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed)
        loss_tensor_parallel, grad_tensor_parallel = self.tensor_parallel_cross_entropy(batch_size, seq_length,
                                                                                        vocab_size, logits_scale, seed)

        error = loss_torch.sub_(loss_tensor_parallel).abs().max()
        assert error < 1.0e-6

        error = grad_torch.sub_(grad_tensor_parallel).abs().max()
        assert error < 1.0e-6

    def torch_cross_entropy(self, batch_size, seq_length, vocab_size, logits_scale, seed):
        identity, logits, target = self.get_logits_and_target(batch_size, seq_length, vocab_size, logits_scale, seed)
        loss = F.cross_entropy(logits.view(-1, logits.size()[-1]), target.view(-1), reduction='none').view_as(
            target).mean()
        loss.backward()
        return loss, identity.weight.grad

    def tensor_parallel_cross_entropy(self, batch_size, seq_length, vocab_size, logits_scale, seed):
        identity, logits, target = self.get_logits_and_target(batch_size, seq_length, vocab_size, logits_scale, seed)
        logits_parallel = mappings.scatter_to_tensor_model_parallel_region(logits)
        loss = vocab_parallel_cross_entropy(logits_parallel, target).mean()
        loss.backward()
        return loss, identity.weight.grad

    def get_logits_and_target(self, batch_size, seq_length, vocab_size, logits_scale, seed):
        set_random_seed(seed)
        identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).cuda()
        logits = identity()
        target = torch.cuda.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size)
        return identity, logits, target
