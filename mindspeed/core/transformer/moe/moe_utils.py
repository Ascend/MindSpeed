# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

AG_TP_HIDDEN_STATUS = None
AG_SHARED_EXPERTS_INPUTS = []
GEMM_BACKWARD_NEED_TENSORS = None
RS_GLOBAL_HIDDEN_STATES_GRAD_WITH_HANDLE = None
SWAP_STREAM = None
SWAP_STREAM2 = None
SWAP_TENSOR = None
MATMUL_OUTPUT_GRAD = None
UNPERMUTED_TOKENS = None


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


def forward_func(func, inputs):
    def detach_tensor(input_):
        if input_.requires_grad and input_.grad_fn is None:
            return input_
        else:
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


def permute(tokens, indices, topk: int = 1):
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    # sorted_indices = torch.argsort(flatten_indices, stable=True)  # argsort int64 will be run on host cpu
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
        topk: int = 1
):
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    sorted_indices = torch.argsort(sorted_indices.float()).int()
    unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens
