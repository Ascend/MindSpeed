# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import fla_npu
    from fla_npu.ops.ascendc import npu_causal_conv1d, npu_causal_conv1d_bwd
except ModuleNotFoundError:
    fla_npu = None


def _activation_mode(activation: Optional[str]) -> int:
    if activation is None:
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise NotImplementedError("activation must be None, silu, or swish")


def _to_npu_conv_state(state: Optional[torch.Tensor], batch: int, width: int, dim: int, x: torch.Tensor):
    if state is None:
        return torch.zeros(batch, width - 1, dim, dtype=x.dtype, device=x.device)
    if state.shape == (batch, dim, width - 1):
        return state.transpose(1, 2).contiguous()
    if state.shape == (batch, width - 1, dim):
        return state.contiguous()
    raise ValueError(
        f"initial_state must have shape {(batch, dim, width - 1)} or "
        f"{(batch, width - 1, dim)}, but got {tuple(state.shape)}"
    )


def _from_npu_conv_state(state: torch.Tensor):
    return state.transpose(1, 2).contiguous()


def _torch_causal_conv1d_segment(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
):
    seq_len = x.shape[1]
    out = F.conv1d(
        input=x.transpose(1, 2).contiguous(),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=weight.shape[-1] - 1,
        groups=weight.shape[0],
    )
    out = out[..., :seq_len]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise NotImplementedError("activation must be None, silu, or swish")
    return out.transpose(1, 2).contiguous()


def _torch_causal_conv1d_packed_merged(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    query_start_loc: list[int],
):
    width = weight.shape[-1]
    pad_len = width - 1
    num_seqs = len(query_start_loc) - 1

    if num_seqs <= 1:
        return _torch_causal_conv1d_segment(x, weight, bias, activation)

    padded_chunks = []
    out_positions = []
    pos = 0
    zero_pad = torch.zeros(1, pad_len, x.shape[-1], dtype=x.dtype, device=x.device)
    for i in range(num_seqs):
        s, e = query_start_loc[i], query_start_loc[i + 1]
        padded_chunks.append(x[:, s:e, :])
        out_positions.append((pos, pos + e - s))
        pos += e - s
        if i < num_seqs - 1:
            padded_chunks.append(zero_pad)
            pos += pad_len

    x_padded = torch.cat(padded_chunks, dim=1)
    total_padded = x_padded.shape[1]
    out_padded = _torch_causal_conv1d_segment(x_padded, weight, bias, activation)

    extracted = []
    for start, end in out_positions:
        extracted.append(out_padded[:, start:end, :])
    return torch.cat(extracted, dim=1)


def _torch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    query_start_loc: Optional[list[int]],
    has_initial_state: bool = False,
):
    if query_start_loc is None:
        return _torch_causal_conv1d_segment(x, weight, bias, activation)

    if not has_initial_state:
        return _torch_causal_conv1d_packed_merged(x, weight, bias, activation, query_start_loc)

    outputs = []
    for start, end in zip(query_start_loc, query_start_loc[1:]):
        outputs.append(_torch_causal_conv1d_segment(x[:, start:end, :], weight, bias, activation))
    return torch.cat(outputs, dim=1)


class _NpuCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        query_start_loc: Optional[list[int]],
    ):
        batch, _, dim = x.shape
        weight_npu = weight.transpose(0, 1).contiguous()
        width = weight_npu.shape[0]
        activation_mode = _activation_mode(activation)

        if query_start_loc is None:
            conv_states = _to_npu_conv_state(None, batch, width, dim, x)
            out = npu_causal_conv1d(
                x=x,
                weight=weight_npu,
                bias=bias,
                conv_states=conv_states,
                activation_mode=activation_mode,
                pad_slot_id=-1,
                run_mode=0,
            )
        else:
            x_packed = x.squeeze(0).contiguous()
            num_seqs = len(query_start_loc) - 1
            conv_states = _to_npu_conv_state(None, num_seqs, width, dim, x)
            out = npu_causal_conv1d(
                x=x_packed,
                weight=weight_npu,
                bias=bias,
                conv_states=conv_states,
                query_start_loc=query_start_loc,
                cache_indices=list(range(num_seqs)),
                initial_state_mode=[0] * num_seqs,
                activation_mode=activation_mode,
                pad_slot_id=-1,
                run_mode=0,
            ).unsqueeze(0)

        ctx.save_for_backward(x, weight)
        ctx.bias = bias
        ctx.activation = activation
        ctx.query_start_loc = query_start_loc
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        weight_npu = weight.transpose(0, 1).contiguous()
        width = weight_npu.shape[0]
        activation_mode = _activation_mode(ctx.activation)
        query_start_loc = ctx.query_start_loc

        if query_start_loc is None:
            op_x = x.contiguous()
            op_grad_out = grad_out.contiguous()
            num_seqs = x.shape[0]
            input_layout = "BSH"
        else:
            op_x = x.squeeze(0).contiguous()
            op_grad_out = grad_out.squeeze(0).contiguous()
            num_seqs = len(query_start_loc) - 1
            input_layout = "TND"

        preactivation = None
        if activation_mode != 0:
            conv_states = _to_npu_conv_state(None, num_seqs, width, x.shape[-1], x)
            forward_kwargs = {
                "x": op_x,
                "weight": weight_npu,
                "bias": ctx.bias,
                "conv_states": conv_states,
                "activation_mode": 0,
                "pad_slot_id": -1,
                "run_mode": 0,
            }
            if query_start_loc is not None:
                forward_kwargs.update(
                    query_start_loc=query_start_loc,
                    cache_indices=list(range(num_seqs)),
                    initial_state_mode=[0] * num_seqs,
                )
            preactivation = npu_causal_conv1d(**forward_kwargs)

        grad_x, grad_weight, grad_bias, _ = npu_causal_conv1d_bwd(
            x=op_x,
            y=preactivation,
            weight=weight_npu,
            dy=op_grad_out,
            initial_state=None,
            dht=None,
            query_start_loc=query_start_loc,
            activation=activation_mode,
            input_layout=input_layout,
        )
        if query_start_loc is not None:
            grad_x = grad_x.unsqueeze(0)
        grad_weight = grad_weight.transpose(0, 1).contiguous()
        if ctx.bias is None:
            grad_bias = None

        return grad_x, grad_weight, grad_bias, None, None


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """FLA-compatible wrapper for the NPU causal conv1d custom op.

    FLA uses x as [batch, seq, dim] (or packed [1, total_seq, dim]) and
    weight as [dim, width]. The NPU op expects weight as [width, dim],
    requires conv_states, and uses query_start_loc for packed sequences.
    """
    if x.dim() != 3:
        raise ValueError(f"causal_conv1d expects a 3D input, but got x.dim()={x.dim()}")
    if weight.dim() != 2:
        raise ValueError(
            f"causal_conv1d expects a 2D weight, but got weight.dim()={weight.dim()}"
        )

    batch, seqlen, dim = x.shape
    weight_npu = weight.transpose(0, 1).contiguous()
    width = weight_npu.shape[0]
    activation_mode = _activation_mode(activation)

    if cu_seqlens is None:
        if initial_state is None and not output_final_state and torch.is_grad_enabled():
            out = _NpuCausalConv1dFunction.apply(x, weight, bias, activation, None)
            return out, None

        conv_states = _to_npu_conv_state(initial_state, batch, width, dim, x)
        out = npu_causal_conv1d(
            x=x,
            weight=weight_npu,
            bias=bias,
            conv_states=conv_states,
            activation_mode=activation_mode,
            pad_slot_id=-1,
            run_mode=0,
        )
        final_state = _from_npu_conv_state(conv_states) if output_final_state else None
        return out, final_state

    if batch != 1:
        raise ValueError("Packed causal_conv1d with cu_seqlens expects batch dimension to be 1")

    x_packed = x.squeeze(0).contiguous()
    query_start_loc = cu_seqlens.reshape(-1).tolist()
    num_seqs = len(query_start_loc) - 1
    if initial_state is None and not output_final_state and torch.is_grad_enabled():
        out = _NpuCausalConv1dFunction.apply(x, weight, bias, activation, query_start_loc)
        return out, None

    conv_states = _to_npu_conv_state(initial_state, num_seqs, width, dim, x)
    out = npu_causal_conv1d(
        x=x_packed,
        weight=weight_npu,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=list(range(num_seqs)),
        initial_state_mode=[1 if initial_state is not None else 0] * num_seqs,
        activation_mode=activation_mode,
        pad_slot_id=-1,
        run_mode=0,
    )
    final_state = _from_npu_conv_state(conv_states) if output_final_state else None
    return out.unsqueeze(0), final_state
