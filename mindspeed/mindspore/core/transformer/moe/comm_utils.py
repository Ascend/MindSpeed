import einops
import torch
import torch_npu
import torch.distributed as dist
import mindspore


COMM_STREAM = None


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None, stream=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[int(sum(output_split_sizes))] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.cuda.current_device(),
        )

    if event or stream:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            if event:
                event.wait()
            if stream:
                COMM_STREAM.wait_stream(stream)
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True
        )
    return input_, a2a_out, handle


def transfer_tensor_last_dim_to_first(input_x):
    num_dims = input_x.dim()
    return einops.rearrange(input_x, "... lst -> lst ...").contiguous(), num_dims


def transfer_tensor_first_dim_to_last(input_x, num_dims):
    return einops.rearrange(input_x, "first ... -> ... first").contiguous()

