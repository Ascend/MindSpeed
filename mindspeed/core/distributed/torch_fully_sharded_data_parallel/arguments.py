from functools import wraps


def _add_distributed_args_wrapper(_add_distributed_args):
    @wraps(_add_distributed_args)
    def wrapper(parser):
        parser = _add_distributed_args(parser)

         # Modify the distributed-backend argument to support extended configurations
        for action in parser._actions:
            if '--distributed-backend' in action.option_strings:
                action.choices = ['nccl', 'gloo', 'npu:hccl,cpu:gloo']
                action.help = 'Backend for distributed training. Single: "nccl" or "gloo", Multiple: "nccl,gloo"'
                break
        return parser
    return wrapper