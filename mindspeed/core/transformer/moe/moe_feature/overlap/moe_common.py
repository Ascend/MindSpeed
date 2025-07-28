# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps
import torch
import torch_npu
import torch.nn.functional as F
from mindspeed.core.transformer.moe.moe_feature import (
     parallel_state, MLP, build_module, TransformerConfig, MLPSubmodules, TransformerConfig)
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu

AG_TP_HIDDEN_STATUS = None
AG_SHARED_EXPERTS_INPUTS = []
GEMM_BACKWARD_NEED_TENSORS = None
RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = None
SWAP_STREAM = None
SWAP_STREAM2 = None
SWAP_TENSOR = None
MATMUL_OUTPUT_GRAD = None
UNPERMUTED_TOKENS = None


def mlp_init(
    self,
    config: TransformerConfig,
    submodules: MLPSubmodules,
    is_expert: bool = False,
    input_size: int = None,
    with_shared_expert=False
):
    """
    Shared expert MLP init with Moe_overlap.
        In 0.10.0, the definition of shared_experts has conflict. 
        Rename the MindSpeed version to 'with_shared_expert'.
    """
    super(MLP, self).__init__(config=config)

    self.config: TransformerConfig = config

    self.input_size = input_size or self.config.hidden_size

    ffn_hidden_size = self.config.ffn_hidden_size
    if self.config.gated_linear_unit:
        ffn_hidden_size *= 2
    if with_shared_expert:
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            with_shared_expert=with_shared_expert
        )
    else:
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1'
        )

    self.activation_func = self.config.activation_func

    if with_shared_expert:
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            with_shared_expert=with_shared_expert
        )
    else:
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2'
        )

    self.with_shared_expert = with_shared_expert


def core_mlp_forward_wrapper(fn):
    """
    A wrapper about setting args for zero_memory&recompute in MLP.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if isinstance(args, tuple):
            args = list(args)

        if self.config.profile and not self.config.num_experts:
            from mindspeed.auto_settings.module.black.patch.hccl_operator import MOEOrMLPStartOp, MOEOrMLPEndOp
            args[0] = MOEOrMLPStartOp.apply(args[0])
            activation_func_1 = torch.nn.Softplus()
            args[0] = activation_func_1(args[0])

        self.layer_number = getattr(self, "layer_number", None)
        is_recompute_activation = should_recompute_activation(self.layer_number)
        if self.config.moe_alltoall_overlap_comm and not isinstance(args[-1], torch.Tensor):
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

        moe_zero_memory = self.config.moe_zero_memory
        if not (is_recompute_activation or moe_zero_memory != "disable"):
            if hasattr(self, 'origin_activation_func'):
                self.activation_func = self.origin_activation_func
            output, output_bias = fn(self, *args, **kwargs)
        elif moe_zero_memory == "level1" and not only_recompute_activation(self.config, layer_number=self.layer_number):
            # Only for zm1 in alltoall_seq dispatcher.
            if self.with_shared_expert:
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
            # [s, b, h]
            output, output_bias = self.linear_fc2(intermediate_parallel)

            # discard the output of the activation function,
            # which will be restored by recomputation during backward.
            self.activation_checkpoint_manager.discard_output()

            # when backward to output of dense_4h_to_h,
            # recompute and restore the output of activation function.
            if output.requires_grad:
                output.register_hook(self.activation_checkpoint_manager.recompute)

        if self.config.profile and not self.config.num_experts:
            activation_func_2 = torch.nn.Softshrink()
            output = activation_func_2(output)
            output = MOEOrMLPEndOp.apply(output)

        return output, output_bias
    return wrapper


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer import AllGatherOverlapMoeLayer, AlltoAllSeqOverlapMoeLayer, AlltoAllOverlapMoeLayer
        if self.config.moe_alltoall_overlap_comm or self.config.moe_allgather_overlap_comm:
            if self.mlp.__class__ is (AllGatherOverlapMoeLayer or AlltoAllSeqOverlapMoeLayer or AlltoAllOverlapMoeLayer):
                self.mlp.experts.layer_number = self.layer_number
                if self.config.moe_shared_expert_intermediate_size:
                    self.mlp.shared_experts.layer_number = self.layer_number
            else:
                self.mlp.layer_number = self.layer_number
    return wrapper


def get_swap_stream():
    global SWAP_STREAM2
    if SWAP_STREAM2 is None:
        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
    stream = SWAP_STREAM2
    return stream


def set_swap_status(tensor):
    global SWAP_TENSOR
    SWAP_TENSOR = tensor


def get_swap_status():
    global SWAP_STREAM
    if SWAP_STREAM is None:
        SWAP_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
    global SWAP_TENSOR
    stream = SWAP_STREAM
    tensor = SWAP_TENSOR
    SWAP_TENSOR = None
    return stream, tensor


def set_prob_backward_need_tensors(matmul_output_grad, unpermuted_tokens):
    global MATMUL_OUTPUT_GRAD
    MATMUL_OUTPUT_GRAD = matmul_output_grad
    global UNPERMUTED_TOKENS
    UNPERMUTED_TOKENS = unpermuted_tokens


def get_prob_backward_need_tensors():
    global SWAP_STREAM2
    if SWAP_STREAM2 is None:
        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
    global MATMUL_OUTPUT_GRAD
    global UNPERMUTED_TOKENS
    stream = SWAP_STREAM2
    matmul_output_grad = MATMUL_OUTPUT_GRAD
    unpermuted_tokens = UNPERMUTED_TOKENS
    MATMUL_OUTPUT_GRAD = None
    UNPERMUTED_TOKENS = None
    return stream, matmul_output_grad, unpermuted_tokens


def set_ag_tp_hidden_status(_inputs):
    global AG_TP_HIDDEN_STATUS
    AG_TP_HIDDEN_STATUS = _inputs


def get_ag_tp_hidden_status():
    global AG_TP_HIDDEN_STATUS
    result = AG_TP_HIDDEN_STATUS
    AG_TP_HIDDEN_STATUS = None
    return result


def set_gemm_backward_need_tensors(_inputs):
    global GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = _inputs


def get_gemm_backward_need_tensors():
    global GEMM_BACKWARD_NEED_TENSORS
    result = GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = None
    return result


def set_rs_global_hidden_states_grad_with_handle(_inputs):
    global RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = _inputs


def get_rs_global_hidden_states_grad_with_handle():
    global RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    result = RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE
    RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = None
    return result


ALL2ALL_EXPERTS_OUTPUT = None


def set_all2all_experts_output(_input):
    global ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = _input


def get_all2all_experts_output():
    global ALL2ALL_EXPERTS_OUTPUT
    result = ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = None
    return result


def only_recompute_activation(config, layer_number):

    vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    vpp_size = config.virtual_pipeline_model_parallel_size
    pp_size = config.transformer_pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = config.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = config.num_layers // pp_size
    else:
        layer_per_chunk = config.num_layers

    vpp_rank = vpp_rank or 0
    vpp_size = vpp_size or 1

    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    moe_zero_memory_num_layers = config.moe_zero_memory_num_layers

    if moe_zero_memory_num_layers:
        if recompute_priority < moe_zero_memory_num_layers:
            return False
        else:
            return True
    else:
        return False 


def forward_func(func, inputs):
    def detach_tensor(input_):
        if input_.requires_grad and input_.grad_fn is None:
            return input_
        new_input = input_.detach()
        new_input.requires_grad = True
        return new_input

    detach_inputs = []
    if isinstance(inputs, tuple):
        for input_ in inputs:
            if isinstance(input_, tuple):
                detach_input = []
                for i in input_:
                    if isinstance(i, torch.Tensor) and torch.is_floating_point(i):
                        detach_input.append(detach_tensor(i))
                    else:
                        detach_input.append(i)
                detach_inputs.append(tuple(detach_input))
            else:
                if isinstance(input_, torch.Tensor) and torch.is_floating_point(input_):
                    detach_input = detach_tensor(input_)
                else:
                    detach_input = input_
                detach_inputs.append(detach_input)
    elif isinstance(inputs, torch.Tensor):
        detach_inputs.append(detach_tensor(inputs))

    with torch.enable_grad():
        output = func(*detach_inputs)

    return output, *detach_inputs


def backward_func(func_tensor, gradinputs):
    if gradinputs is None or func_tensor.grad_fn is None:
        return
    if isinstance(gradinputs, torch.Tensor):
        func_tensor.backward(gradinputs)
    elif isinstance(gradinputs, tuple):
        func_tensor.backward(*gradinputs)

