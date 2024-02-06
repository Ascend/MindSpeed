import os
import torch
from torch import nn
from megatron.model.rms_norm import RMSNorm
from megatron import get_args
import torch_npu


def rms_norm_init(self,
                  dim: int,
                  eps: float = 1e-6,
                  sequence_parallel: bool = False):
    """RMS Normaliation module

    Arguments:
        dim (int): The width of input, i.e. hidden size
        eps (float): epsilon to use for the norm, default to 1e-6
        sequence_parallel (bool): Set to true if sequence parallelism is being used,
            this marks the weights as needing to be allreduced.
    """
    super(RMSNorm, self).__init__()
    args = get_args()
    self.use_fused_rmsnorm = args.use_fused_rmsnorm
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.hook_module.hook_module import HOOKModule

        def norm_hook(grad):
            if HOOKModule.collect_flag:
                if grad.dtype == torch.float16:
                    grad = grad.float()
                HOOKModule.layernorm_list.append(torch.norm(grad))

        self.weight.register_hook(norm_hook)

    setattr(self.weight, 'sequence_parallel', sequence_parallel)


def rms_norm_forward(self, x):
    if self.use_fused_rmsnorm:
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
    output = self._norm(x.float()).type_as(x)
    return output * self.weight