from mindspeed.te.module.ops.cockernel_ops import CocKernelOps
from mindspeed.te.module.ops.default_ops import DefaultOps
from mindspeed.te.module.ops.mc2_ops import Mc2Ops
from mindspeed.te.module.ops.coc_ops import CocOps


OPS_MAP = {
    "mc2": Mc2Ops,
    "coc": CocOps,
    "coc_kernel": CocKernelOps,
    "default": DefaultOps
}


def get_ops():
    from megatron.training import get_args
    args = get_args()
    if not hasattr(args, 'comm_overlap_type'):
        args.comm_overlap_type = "default"
    return OPS_MAP[args.comm_overlap_type]


class DummyHandle:

    def wait(self, *args, **kwargs):
        pass
