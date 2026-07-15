import torch
import torch_npu
from einops import rearrange
from functools import lru_cache


def check_optional_tensor(tensor, device, name):
    if not isinstance(tensor, (torch.Tensor, type(None))):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}.")
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{device}(arg0) and {tensor.device}({name})!"
        )


@lru_cache(maxsize=None)
def get_lut_256(device):
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float16, device=device
    )
    indices = torch.arange(256, device=device, dtype=torch.int32)
    low = indices & 0x0F
    high = (indices >> 4) & 0x0F
    lut = torch.stack([e2m1[low.long()], e2m1[high.long()]], dim=-1)
    return lut


def transform_mxfp4_to_mxfp8(quant_fp4_packed, scale_fp4_packed, mx4_block_size=32, mx8_block_h=128, mx8_block_w=128):
    B, H, W_packed = quant_fp4_packed.shape
    original_W = W_packed * 2
    device = quant_fp4_packed.device

    lut = get_lut_256(device)
    quant_fp4 = lut[quant_fp4_packed.long()].view(B, H, -1)

    total_groups = B * H * (original_W // mx4_block_size)
    scale_flat = scale_fp4_packed.view(-1, 2)
    scale_fp4 = scale_flat.view(-1)[:total_groups].view(B, H, -1)

    num_h_blocks = H // mx8_block_h
    num_w_blocks = original_W // mx8_block_w
    groups_per_block = mx8_block_w // mx4_block_size

    scale_blocks = scale_fp4.view(B, num_h_blocks, mx8_block_h, num_w_blocks, groups_per_block)
    s4_blocks = scale_blocks.unsqueeze(-1).expand(-1, -1, -1, -1, -1, mx4_block_size)
    s4_blocks = s4_blocks.reshape(B, num_h_blocks, mx8_block_h, num_w_blocks, mx8_block_w)
    s4_blocks = s4_blocks.permute(0, 1, 3, 2, 4)

    quant_blocks = quant_fp4.view(B, num_h_blocks, mx8_block_h, num_w_blocks, mx8_block_w)
    quant_blocks = quant_blocks.permute(0, 1, 3, 2, 4)

    s4_max = s4_blocks.amax(dim=(-2, -1))
    s8 = (s4_max - 6).clamp(0, 255).to(torch.uint8)

    exp_diff = s4_blocks - s8[:, :, :, None, None]
    scaled = quant_blocks * torch.exp2(exp_diff.to(torch.float32))
    scaled = scaled.clamp(-448.0, 448.0)
    quant_fp8_blocks = scaled.to(torch.float8_e4m3fn)

    quant_fp8 = quant_fp8_blocks.permute(0, 1, 3, 2, 4).reshape(B, H, original_W)
    return quant_fp8, s8


def expand_scale_to_per_row(scale_2d, block_size=32):
    B, Hb, Wb = scale_2d.shape
    H = Hb * block_size
    scale_expanded = scale_2d[:, :, None, :].expand(-1, -1, block_size, -1)
    scale_per_row = scale_expanded.reshape(B, H, Wb)
    scale_final = scale_per_row.view(B, H, Wb // 2, 2)
    return scale_final


def convert_mx_scale_to_axis2(weight_scale_mx, weight_shape, block_size_old=32):
    B, H_blocks_old, W_blocks_old = weight_scale_mx.shape
    _, H, W = weight_shape

    ratio = 2
    H_blocks_new = H_blocks_old // ratio

    reshaped = weight_scale_mx.float().view(B, H_blocks_new, ratio, W_blocks_old)

    scale_top = reshaped[:, :, 0, :]
    scale_bottom = reshaped[:, :, 1, :]

    def expand_width(scale_2d):
        expanded = scale_2d[:, :, :, None].expand(-1, -1, -1, block_size_old)
        return expanded.reshape(B, H_blocks_new, W)

    top_expanded = expand_width(scale_top)
    bottom_expanded = expand_width(scale_bottom)

    scale_final = torch.stack([top_expanded, bottom_expanded], dim=-1)
    return scale_final


def transform_grouped(input_tensor, input_scale_mxfp8, group_list, block_size):
    idx = torch.arange(group_list.size(0), device=group_list.device, dtype=group_list.dtype)
    zero_rows = group_list // (block_size // 2) + idx

    input_scale_repeat = input_scale_mxfp8.view(-1, input_scale_mxfp8.shape[1]).repeat_interleave(4, dim=0)
    input_scale_32_packed = input_scale_repeat.view(input_scale_mxfp8.shape[0] * 4, input_scale_mxfp8.shape[1], 2)

    scale_shape = input_tensor.shape[0] // block_size * 2 + group_list.shape[0]
    all_indices = torch.arange(scale_shape, device=input_scale_32_packed.device)
    non_zero_mask = ~torch.isin(all_indices, zero_rows)
    non_zero_rows = all_indices[non_zero_mask]

    input_bwd_scale_mxfp8 = torch.zeros(
        input_scale_32_packed.shape[0] + group_list.shape[0],
        input_scale_mxfp8.shape[1],
        2,
        dtype=input_scale_32_packed.dtype,
        device=input_scale_32_packed.device,
    )
    input_bwd_scale_mxfp8[non_zero_rows] = input_scale_32_packed

    return input_bwd_scale_mxfp8


class W4A8GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, group_list_type, group_list_data_type, block_size = group_args

        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        x_fwd_mxfp8, x_fwd_scale_mxfp8 = torch_npu.npu_dynamic_mx_quant(
            x, axis=-1, dst_type=torch_npu.float8_e4m3fn, block_size=block_size
        )

        if block_size == 32:
            x_bwd_mxfp8, x_bwd_scale_mxfp8 = torch_npu.npu_grouped_dynamic_mx_quant(
                x, group_list.to(torch.int32), round_mode="rint", dst_type=torch_npu.float8_e4m3fn, blocksize=32
            )
            weight_mxfp4, weight_mxscale_mxfp4 = torch_npu.npu_dynamic_mx_quant(
                weight, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0
            )
            weight_fwd_fp8, weight_scale_fp8 = transform_mxfp4_to_mxfp8(
                weight_mxfp4, weight_mxscale_mxfp4, mx4_block_size=32, mx8_block_h=32, mx8_block_w=32
            )

            weight_bwd_scale_mxfp8 = expand_scale_to_per_row(weight_scale_fp8, block_size=32).to(torch.uint8)
            weight_fwd_scale_mxfp8 = convert_mx_scale_to_axis2(weight_scale_fp8, weight.shape).to(torch.uint8)

            outputs = torch_npu.npu_grouped_matmul(
                [x_fwd_mxfp8],
                [weight_fwd_fp8],
                scale=[weight_fwd_scale_mxfp8],
                per_token_scale=[x_fwd_scale_mxfp8],
                group_list=group_list,
                group_type=0,
                output_dtype=torch.bfloat16,
                group_list_type=0,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                split_item=3,
            )

        else:
            x_bwd_mxfp8, x_scale_mxfp8 = torch_npu.npu_dynamic_mx_quant(
                x, axis=-2, dst_type=torch_npu.float8_e4m3fn, block_size=block_size
            )
            x_bwd_scale_mxfp8 = transform_grouped(x, x_scale_mxfp8, group_list, block_size)

            weight_mxfp4, weight_mxscale_mxfp4 = torch_npu.npu_dynamic_mx_quant(
                weight, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0
            )
            weight_fwd_fp8, weight_scale_fp8 = transform_mxfp4_to_mxfp8(
                weight_mxfp4, weight_mxscale_mxfp4, mx4_block_size=32, mx8_block_h=block_size, mx8_block_w=block_size
            )
            weight_scale_fp32 = torch.pow(2.0, weight_scale_fp8.to(torch.float32) - 127).to(torch.float32)
            x_scale_fp32 = (
                torch.pow(2.0, x_fwd_scale_mxfp8.to(torch.float32) - 127)
                .view(x_fwd_scale_mxfp8.shape[0], -1)
                .to(torch.float32)
            )
            weight_bwd_scale_mxfp8 = weight_scale_fp32

            outputs = torch_npu.npu_grouped_matmul(
                [x_fwd_mxfp8],
                [weight_fwd_fp8],
                scale=[weight_scale_fp32],
                per_token_scale=[x_scale_fp32],
                group_list=group_list,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
                group_list_type=0,
            )

        ctx.save_for_backward(
            x,
            weight,
            group_list,
            original_weight,
            x_bwd_mxfp8,
            x_bwd_scale_mxfp8,
            weight_fwd_fp8,
            weight_bwd_scale_mxfp8,
        )
        ctx.group_list = group_list
        ctx.block_size = block_size

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        (
            x,
            weight,
            group_list,
            original_weight,
            x_bwd_mxfp8,
            x_bwd_scale_mxfp8,
            weight_fwd_fp8,
            weight_bwd_scale_mxfp8,
        ) = ctx.saved_tensors
        block_size = ctx.block_size
        group_list = ctx.group_list
        grad_x_mxfp8, grad_x_scale = torch_npu.npu_dynamic_mx_quant(
            grad_outputs, axis=-1, dst_type=torch_npu.float8_e4m3fn, block_size=block_size
        )
        if block_size == 32:
            dx = torch_npu.npu_grouped_matmul(
                [grad_x_mxfp8],
                [rearrange(weight_fwd_fp8, 'n h f -> n f h')],
                scale=[rearrange(weight_bwd_scale_mxfp8, 'n h f g -> n f h g')],
                per_token_scale=[grad_x_scale],
                group_list=group_list,
                group_type=0,
                output_dtype=torch.bfloat16,
                group_list_type=0,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                split_item=3,
            )

            grad_weight_mxfp8, grad_weight_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                grad_outputs,
                group_list.to(torch.int32),
                round_mode="rint",
                dst_type=torch_npu.float8_e4m3fn,
                blocksize=block_size,
            )
            dw = torch_npu.npu_grouped_matmul(
                [x_bwd_mxfp8.t()],
                [grad_weight_mxfp8],
                scale=[grad_weight_scale],
                per_token_scale=[rearrange(x_bwd_scale_mxfp8, 'n h f -> h n f')],
                group_list=group_list,
                split_item=3,
                group_type=2,
                output_dtype=torch.bfloat16,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                group_list_type=0,
            )
        else:
            grad_x_scale = (
                torch.pow(2.0, grad_x_scale.to(torch.float32) - 127).view(grad_x_scale.shape[0], -1).to(torch.float32)
            )
            dx = torch_npu.npu_grouped_matmul(
                [grad_x_mxfp8],
                [weight_fwd_fp8.transpose(1, 2)],
                scale=[weight_bwd_scale_mxfp8.transpose(1, 2)],
                per_token_scale=[grad_x_scale],
                group_list=group_list,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
                group_list_type=0,
            )

            grad_weight_mxfp8, grad_scale_mxfp8 = torch_npu.npu_dynamic_mx_quant(
                grad_outputs, axis=-2, dst_type=torch_npu.float8_e4m3fn, block_size=block_size
            )
            grad_weight_scale = transform_grouped(grad_outputs, grad_scale_mxfp8, group_list, block_size)

            dw = torch_npu.npu_grouped_matmul(
                [x_bwd_mxfp8.t()],
                [grad_weight_mxfp8],
                scale=[grad_weight_scale],
                per_token_scale=[rearrange(x_bwd_scale_mxfp8, 'n h f -> h n f')],
                group_list=group_list,
                split_item=3,
                group_type=2,
                output_dtype=torch.bfloat16,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                group_list_type=0,
            )

        return None, dx[0], dw[0], None, None


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, original_weight=None, block_size=32):
    group_args = (group_list, group_type, 0, 0, block_size)
    return W4A8GMMFunction.apply(original_weight, x, weight, bias, group_args)
