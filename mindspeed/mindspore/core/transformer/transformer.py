from functools import wraps
import torch
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.moe.moe_utils import only_recompute_activation
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu


def core_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.layer_number = getattr(self, "layer_number", None)
        is_recompute_activation = should_recompute_activation(self.layer_number)
        if get_args().moe_alltoall_overlap_comm and not isinstance(args[-1], torch.Tensor):
            moe_ctx = args[-1]
            args = args[:-1]

        def activation_function(*function_args):
            intermediate, bias = function_args
            if bias is not None:
                intermediate = intermediate + bias
            if self.config.gated_linear_unit:
                assert (self.config.activation_func == F.silu), 'Activation function must be silu when using fused_swiglu'
                if not hasattr(self, 'origin_activation_func'):
                    self.origin_activation_func = self.activation_func
                self.activation_func = fused_swiglu
                intermediate = self.activation_func(intermediate)
            else:
                intermediate = self.activation_func(intermediate)

            return intermediate

        moe_zero_memory = get_args().moe_zero_memory
        if not (is_recompute_activation or moe_zero_memory != "disable"):
            if hasattr(self, 'origin_activation_func'):
                self.activation_func = self.origin_activation_func
            output, output_bias = fn(self, *args, **kwargs)
        elif moe_zero_memory == "level1" and not only_recompute_activation(self.layer_number):
            if self.shared_expert:
                self.activation_function = activation_function
                hidden_states = args[0]
                fc1_out_parallel, bias_parallel = self.linear_fc1(hidden_states)
                act_out_parallel = activation_function(fc1_out_parallel, bias_parallel)
                output, output_bias = self.linear_fc2(act_out_parallel)
                fc1_out_parallel.untyped_storage().resize_(0)
                act_out_parallel.untyped_storage().resize_(0)
                moe_ctx.shared_fc1_out = fc1_out_parallel
                moe_ctx.shared_act_out = act_out_parallel
            else:
                output, output_bias = fn(self, *args, **kwargs)
        else:
            hidden_states = args[0]
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
            self.activation_checkpoint_manager = CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                                  False,
                                                                                  intermediate_parallel,
                                                                                  bias_parallel)
            output, output_bias = self.linear_fc2(intermediate_parallel)

            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            # if output.requires_grad:
            output.register_hook(self.activation_checkpoint_manager.recompute)
        return output, output_bias
    return wrapper