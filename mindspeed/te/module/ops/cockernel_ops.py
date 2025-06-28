import torch

from mindspeed.te.module.ops.comm_overlap_ops import CommOverlapOps, COMM_OVERLAP_CONFIG


class CocKernelOps(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        from mindspeed.op_builder import LcalOpBuilder
        mindspeed_ops = LcalOpBuilder().load()
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = allocate_for_output(x, weight, tp_world_size, is_gather=True)
        all_gather_grad_output = allocate_for_output(x, tp_world_size=tp_world_size, is_gather=True)

        device = input_.device.index
        comm_domain = str(device // tp_world_size)
        rank = device % tp_world_size
        mindspeed_ops.all_gather_matmul_v2(x, weight, bias, output, all_gather_grad_output, rank, tp_world_size,
                                                         comm_domain)
        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])
        return output, all_gather_grad_output, None

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        from mindspeed.op_builder import LcalOpBuilder
        mindspeed_ops = LcalOpBuilder().load()
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = allocate_for_output(x, weight.t(), tp_world_size, is_gather=False)

        device = input_.device.index
        comm_domain = str(device // tp_world_size)
        rank = device % tp_world_size
        mindspeed_ops.matmul_reduce_scatter(x, weight, bias, output, rank, tp_world_size, comm_domain)
        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])
        return output, None, None

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        from mindspeed.op_builder import LcalOpBuilder
        mindspeed_ops = LcalOpBuilder().load()
        output_orig_shape = get_output_shape(input_, weight.t(), 1, is_gather=True)
        input_ = reshape_to_2D(input_)
        output_ = allocate_for_output(input_, weight.t(), 1, is_gather=True)

        device = input_.device.index
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        comm_domain = str(device // tp_world_size)
        rank = device % tp_world_size
        mindspeed_ops.matmul_all_reduce(input_, weight, bias, output_, rank, tp_world_size, comm_domain)
        output_ = output_.reshape(output_orig_shape)

        return output_, None, None


def check_equal(a, b, error_info):
    if a != b:
        if torch.npu.current_device() == 0:
            print(error_info)


def get_output_shape(input1, input2=None, tp_world_size=1, is_gather=True):
    check_equal(input1.dim() >= 2 and (input2 is None or input2.dim() == 2), True,
                error_info="invalid matmul input shape for CoC")
    output_shape = list(input1.shape)[:-1] + list([input2.shape[-1]]) if input2 is not None else list(input1.shape)
    if not is_gather:
        check_equal(output_shape[0] % tp_world_size == 0 and output_shape[0] >= tp_world_size, True,
                    error_info="invalid matmul m shape for CoC")
    output_shape[0] = output_shape[0] * tp_world_size if is_gather else output_shape[0] // tp_world_size
    return output_shape


def reshape_to_2D(input_tensor):
    # Convert the tensor shapes to 2D for execution compatibility
    input_tensor = input_tensor.reshape(input_tensor.shape[0] * input_tensor.shape[1],
                                        input_tensor.shape[2])
    return input_tensor


def allocate_for_output(input1, input2=None, tp_world_size=1, is_gather=True):
    if input2 is not None:
        dim_size = list(input1.shape)[:-1] + list([input2.shape[1]])
    else:
        dim_size = list(input1.shape)
    dim_size[0] = dim_size[0] * tp_world_size if is_gather else dim_size[0] // tp_world_size
    output = torch.empty(dim_size, dtype=input1.dtype, device=torch.npu.current_device())
    return output