# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from copy import deepcopy
from contextlib import AbstractContextManager, nullcontext
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.shared_experts import SharedExpertMLP, set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from mindspeed.core.transformer.moe.comm_utils import async_all_gather, async_reduce_scatter
from mindspeed.core.transformer.moe.moe_feature import MLPSubmodules
from .utils import run_graph_backward, detach_tensor


class NoTPCommContext(AbstractContextManager):
    def __init__(self, ag_out=None, cached_rs_input=None):
        super().__init__()
        self.ag_out = ag_out
        self.cached_rs_input = cached_rs_input
        self.orig_allgather_base = torch.distributed._all_gather_base
        self.orig_rs_base = torch.distributed._reduce_scatter_base

    def make_allgather_base_patch(self):
        def patch_fn(*args, **kwargs):
            out = args[0]
            out.data = self.ag_out.data
            return

        return patch_fn

    def make_reducescatter_base_patch(self):
        def patch_fn(*args, **kwargs):
            input_ = args[1]
            self.cached_rs_input.data = input_.data

            class DummyWaitHandle:
                def __init__(self):
                    pass

                @staticmethod
                def wait():
                    return
                
            handle = None
            if kwargs['async_op']:
                handle = DummyWaitHandle()

            return handle

        return patch_fn

    def __enter__(self):
        if self.ag_out is not None:
            torch.distributed._all_gather_base = self.make_allgather_base_patch()
        if self.cached_rs_input is not None:
            torch.distributed._reduce_scatter_base = self.make_reducescatter_base_patch()

        return

    def __exit__(self, exc_type, exc_value, traceback):
        if self.ag_out is not None:
            torch.distributed._all_gather_base = self.orig_allgather_base
        if self.cached_rs_input is not None:
            torch.distributed._reduce_scatter_base = self.orig_rs_base

        return


class SharedExpertMLPFbOverlap(SharedExpertMLP):
    """
    MLP layer for Shared Experts.
    """

    # This stream is used when '--moe-shared-expert-overlap' is set.
    # The shared experts are scheduled into this stream to be overlapped with the dispatcher.
    stream = None

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules, gate: bool):
        config = deepcopy(config)
        config.moe_shared_expert_overlap = True
        super().__init__(config=config, submodules=submodules, gate=False)
        self.config.coc_row_nocomm = self.config.coc_fused_kernel
        assert not self.use_shared_expert_gate
        self.cached_fc1_ag_input = None
        self.cached_fc1_input_grad = None
        self.fc1_input_comm_handle = None
        self.fc2_output_comm_handle = None
        self.cached_backward_grad = None
        self.pre_backward_handle = None
        self.cached_input_grad = None
        self.post_backward_handle = None
        self.tp_group = get_tensor_model_parallel_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        # overwrite the explicit expert comm back to False
        self.linear_fc1.explicit_expert_comm = False

        if self.tp_size > 1:
            assert self.config.sequence_parallel

    def pre_forward_comm(self, input_, wait_event=None):
        """
        All Gather for SP before forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is None

        if self.tp_size > 1:
            _, self.cached_fc1_ag_input, self.fc1_input_comm_handle = async_all_gather(
                input_, self.tp_group, event=wait_event, stream=torch.npu.current_stream() if wait_event else None,
                is_use_get_global_memory_buffer=False,
            )
        else:
            self.cached_fc1_ag_input, self.fc1_input_comm_handle = input_, None

        self.cached_fc1_input = input_

    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc1_input is not None
        assert self.cached_fc1_ag_input is not None

        if self.fc1_input_comm_handle:
            self.fc1_input_comm_handle.wait()
            self.fc1_input_comm_handle = None

        tp_comm_context = nullcontext()
        if self.tp_size > 1:
            tp_comm_context = NoTPCommContext(ag_out=self.cached_fc1_ag_input)
        with tp_comm_context:
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = self.linear_fc1(self.cached_fc1_input)

        self.cached_fc1_input = None
        self.cached_fc1_ag_input = None

        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(
                        intermediate_parallel, bias_parallel
                    )
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        self.cached_fc2_input = intermediate_parallel

    def linear_fc2_forward(self, overlapped_comm_output=None):
        """
        Do Linear FC2 forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_input is not None

        self.cached_fc2_output, _ = self.linear_fc2(self.cached_fc2_input)
        self.cached_fc2_input = None

    def post_forward_comm(self, wait_event=None):
        """
        Reduce scatter for SP after forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_output is not None


        if self.tp_size > 1:
            _, self.cached_output, self.fc2_output_comm_handle = async_reduce_scatter(
                self.cached_fc2_output, self.tp_group, event=wait_event,
                stream=torch.npu.current_stream() if wait_event else None
            )
        else:
            self.cached_output, self.fc2_output_comm_handle = self.cached_fc2_output, None

    def get_output(self):
        """
        Gets the module forward output.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is not None
        if self.fc2_output_comm_handle:
            self.fc2_output_comm_handle.wait()
            self.fc2_output_comm_handle = None

        output_before_rs = self.cached_fc2_output
        self.cached_fc2_output = None
        if self.tp_size > 1:
            # output_before_rs is only used for run backward graph for shared experts.
            output_before_rs.untyped_storage().resize_(0)


        output = self.cached_output
        self.cached_output = None

        return output, output_before_rs

    def pre_backward_comm(self, grad, wait_event=None):
        assert self.config.moe_shared_expert_overlap
        assert self.cached_backward_grad is None
        assert self.pre_backward_handle is None


        if self.tp_size > 1:
            _, self.cached_backward_grad, self.pre_backward_handle = async_all_gather(
                grad, self.tp_group, event=wait_event, stream=torch.npu.current_stream() if wait_event else None,
                is_use_get_global_memory_buffer=False,
            )
        else:
            self.cached_backward_grad, self.pre_backward_handle = grad.clone(), None

    def linear_fc2_act_fc1_backward(self, shared_experts_graph, keep_grad=True):
        assert self.cached_backward_grad is not None
        assert self.cached_fc1_input_grad is None
        if self.pre_backward_handle is not None:
            self.pre_backward_handle.wait()
            self.pre_backward_handle = None

        # create a dummy tensor to catch the backward reduce scatter input grad.
        self.cached_fc1_input_grad = torch.empty(1, device=shared_experts_graph[0].device, dtype=shared_experts_graph[0].dtype)
        tp_comm_context = nullcontext()
        if self.tp_size > 1:
            tp_comm_context = NoTPCommContext(cached_rs_input=self.cached_fc1_input_grad)

        with tp_comm_context:
            run_graph_backward(shared_experts_graph, self.cached_backward_grad, keep_grad=keep_grad)
        self.cached_backward_grad = None

    def post_backward_comm(self, wait_event=None):
        assert self.cached_input_grad is None
        assert self.pre_backward_handle is None
        assert self.cached_backward_grad is None
        assert self.cached_fc1_input_grad is not None

        if self.tp_size > 1:
            _, self.cached_input_grad, self.post_backward_handle = async_reduce_scatter(
                self.cached_fc1_input_grad, self.tp_group, event=wait_event,
                stream=torch.npu.current_stream() if wait_event else None
            )
        else:
            self.cached_input_grad, self.post_backward_handle = None, None

    def get_backward_grad(self):
        if self.post_backward_handle:
            self.post_backward_handle.wait()
            self.post_backward_handle = None
        self.cached_fc1_input_grad = None

        out_grad = self.cached_input_grad
        self.cached_input_grad = None
        if self.config.coc_row_nocomm and out_grad is not None:
            out_grad = out_grad.unsqueeze(1)

        return out_grad






