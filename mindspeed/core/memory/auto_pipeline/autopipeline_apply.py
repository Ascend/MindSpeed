from megatron.core import parallel_state, tensor_parallel
from mindspeed.args_utils import get_full_args as get_args


def apply_autopipeline(models):
    if isinstance(models, list):
        for model in models:
            apply_recompute_modules(model)
    else:
        apply_recompute_modules(models)


def apply_recompute_modules(model):
    args = get_args()
    for pp_rankid, recomp_value in enumerate(args.recompute_module_list):
        if pp_rankid == parallel_state.get_pipeline_model_parallel_rank():
            if recomp_value > 0:
                set_recompute_modules(model, recomp_value, args.recompute_type)


def set_recompute_modules(model, recomp_value, module_type):
    recomp_pool = []
    recomp_name = "module.module.language_model.encoder.layers."
    for i in range(0, recomp_value):
        tmp_recomp_name = recomp_name
        tmp_recomp_name += str(i)
        # mlp recompute type
        if module_type == 0:
            tmp_recomp_name += ".mlp"
            recomp_pool.append(tmp_recomp_name)
        # attention recompute type
        if module_type == 1:
            tmp_recomp_name += ".self_attention"
            recomp_pool.append(tmp_recomp_name)
        # layer recompute type
        if module_type == 2:
            recomp_pool.append(tmp_recomp_name)

    for name, module in model.named_modules():
        if name in recomp_pool:
            module.forward = hook_checkpoint_forward(module.forward)


def hook_checkpoint_forward(forward_func):
    def custom_forward(*args, **kargs):
        def inside_forward(*args):
            return forward_func(*args, **kargs)

        return tensor_parallel.checkpoint(inside_forward, None, *args)

    return custom_forward
