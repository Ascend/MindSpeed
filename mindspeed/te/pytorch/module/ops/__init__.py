from mindspeed.te.pytorch.module.ops.default_ops import DefaultOps
from mindspeed.te.pytorch.module.ops.mc2_ops import Mc2Ops


def get_ops():
    from mindspeed.args_utils import get_full_args as get_args
    args = get_args()
    if hasattr(args, 'use_ascend_mc2') and args.use_ascend_mc2:
        return Mc2Ops
    else:
        return DefaultOps


class DummyHandle:

    def wait(self, *args, **kwargs):
        pass
