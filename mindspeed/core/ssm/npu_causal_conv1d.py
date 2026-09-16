# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""FLA-NPU AscendC causal-convolution adapter for GatedDeltaNet."""

import torch


def _load_ascendc_ops():
    """Load FLA-NPU's public AscendC bindings only when this backend is used."""
    from fla_npu.ops.ascendc import npu_causal_conv1d, npu_causal_conv1d_bwd

    return npu_causal_conv1d, npu_causal_conv1d_bwd


def _activation_mode(activation):
    if activation is None:
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise ValueError(f"Unsupported activation: {activation}")


def _query_start_loc(cu_seqlens):
    return [int(value) for value in cu_seqlens.detach().cpu().tolist()]


class _AscendCCausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, activation_mode, cu_seqlens):
        ascendc_forward, ascendc_backward = _load_ascendc_ops()
        op_weight = weight.transpose(0, 1).contiguous()
        query_start_loc = None
        is_packed = cu_seqlens is not None

        if is_packed:
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("Packed causal_conv1d expects [1, T, D] input")
            op_x = x.squeeze(0).contiguous()
            query_start_loc = _query_start_loc(cu_seqlens)
            num_sequences = len(query_start_loc) - 1
        else:
            op_x = x.contiguous()
            num_sequences = x.shape[0]

        conv_states = torch.zeros(
            num_sequences,
            op_weight.shape[0] - 1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        forward_kwargs = {
            "x": op_x,
            "weight": op_weight,
            "bias": bias,
            "conv_states": conv_states,
            "activation_mode": activation_mode,
            "pad_slot_id": -1,
            "run_mode": 0,
            "head_num": 0,
        }
        if is_packed:
            forward_kwargs.update(
                query_start_loc=query_start_loc,
                cache_indices=list(range(num_sequences)),
                initial_state_mode=[0] * num_sequences,
            )
        y = ascendc_forward(**forward_kwargs)

        tensors = [x, op_weight]
        if bias is not None:
            tensors.append(bias)
        ctx.save_for_backward(*tensors)
        ctx.has_bias = bias is not None
        ctx.activation_mode = activation_mode
        ctx.query_start_loc = query_start_loc
        ctx.is_packed = is_packed
        ctx.ascendc_forward = ascendc_forward
        ctx.ascendc_backward = ascendc_backward
        return y.unsqueeze(0) if is_packed else y

    @staticmethod
    def backward(ctx, grad_output):
        saved = list(ctx.saved_tensors)
        x = saved.pop(0)
        op_weight = saved.pop(0)
        bias = saved.pop(0) if ctx.has_bias else None

        op_x = x.squeeze(0).contiguous() if ctx.is_packed else x.contiguous()
        op_grad = grad_output.squeeze(0).contiguous() if ctx.is_packed else grad_output.contiguous()
        preactivation = None
        if ctx.activation_mode != 0:
            num_sequences = len(ctx.query_start_loc) - 1 if ctx.is_packed else x.shape[0]
            conv_states = torch.zeros(
                num_sequences,
                op_weight.shape[0] - 1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            forward_kwargs = {
                "x": op_x,
                "weight": op_weight,
                "bias": bias,
                "conv_states": conv_states,
                "activation_mode": 0,
                "pad_slot_id": -1,
                "run_mode": 0,
                "head_num": 0,
            }
            if ctx.is_packed:
                forward_kwargs.update(
                    query_start_loc=ctx.query_start_loc,
                    cache_indices=list(range(num_sequences)),
                    initial_state_mode=[0] * num_sequences,
                )
            preactivation = ctx.ascendc_forward(**forward_kwargs)

        dx, dw, db, _ = ctx.ascendc_backward(
            x=op_x,
            y=preactivation,
            weight=op_weight,
            dy=op_grad,
            initial_state=None,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=ctx.activation_mode,
            input_layout="TND" if ctx.is_packed else "BSH",
        )
        if ctx.is_packed:
            dx = dx.unsqueeze(0)
        return dx, dw.transpose(0, 1).contiguous(), db if ctx.has_bias else None, None, None


def causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """Run AscendC causal convolution for fixed BSH or packed TND GDN input."""
    if initial_state is not None or output_final_state:
        raise NotImplementedError("GDN causal_conv1d does not support convolution state")

    output = _AscendCCausalConv1d.apply(
        x,
        weight,
        bias,
        _activation_mode(activation),
        cu_seqlens,
    )
    return output, None
