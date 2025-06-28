from enum import Enum

import torch

from mindspeed.te.module.ops.comm_overlap_ops import CommOverlapOps
from mindspeed.te.module.ops.comm_overlap_ops import COMM_OVERLAP_CONFIG


class CommunicationType(Enum):
    ALL_GATHER = 0
    ALL_REDUCE = 1
    REDUCE_SCATTER = 2


class CocOps(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        parallel_num = COMM_OVERLAP_CONFIG.parallel_num
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        output_orig_shape = get_output_shape(input_, weight, tp_world_size, is_gather=True)
        gathered_input_shape = get_output_shape(input_, None, tp_world_size, is_gather=True)
        input_ = reshape_to_2D(input_)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, weight, out=output_tensor)
            return output_tensor

        coc_parallel = COCParallel(input_, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                 weight_shape_list=list(weight.shape), parallel_num=parallel_num)
        output = coc_parallel.run()
        output = shuffle_as_coc_reduce_scatter(output, tp_world_size, parallel_num)

        total_input = shuffle_as_coc_reduce_scatter(coc_parallel.comm_output, tp_world_size,
                                                       parallel_num)
        total_input = total_input.reshape(gathered_input_shape)
        output = output.reshape(output_orig_shape)
        if bias is not None:
            output = output + bias
        return output, total_input, None

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        weight = weight.t()
        parallel_num = COMM_OVERLAP_CONFIG.parallel_num
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        output_orig_shape = get_output_shape(input_, weight, tp_world_size, is_gather=False)
        input_ = reshape_to_2D(input_)

        def compute_fcn(input_tensor):
            sub_output = torch.matmul(input_tensor, weight)
            return sub_output

        input_ = shuffle_as_coc_all_gather(input_, tp_world_size, parallel_num)
        coc_reduce_scatter = COCParallel(input_, CommunicationType.REDUCE_SCATTER, compute_fcn, compute_first=True,
                                         weight_shape_list=list(weight.shape), parallel_num=parallel_num)
        output_ = coc_reduce_scatter.run()
        output_ = output_.reshape(output_orig_shape)
        if bias is not None:
            output_ = output_ + bias
        return output_, None, None

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        weight = weight.t()
        parallel_num = COMM_OVERLAP_CONFIG.parallel_num
        output_orig_shape = get_output_shape(input_, weight, 1, is_gather=True)
        input_ = reshape_to_2D(input_)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, weight, out=output_tensor)
            return output_tensor

        coc_all_gather = COCParallel(input_, CommunicationType.ALL_REDUCE, compute_fcn, compute_first=True,
                                   weight_shape_list=list(weight.shape), parallel_num=parallel_num)
        output_ = coc_all_gather.run()
        output_ = output_.reshape(output_orig_shape)
        if bias is not None:
            output_ = output_ + bias
        return output_, None, None


class COCParallel:
    def __init__(self, input_data, comm_type, compute_fcn, compute_first=True, synchronize=True, weight_shape_list=None,
                 parallel_num=2):
        self.input_data = input_data
        self.split_num = parallel_num
        self.synchronize = synchronize
        self.comm_type = comm_type
        self.compute_fcn = compute_fcn
        self.compute_first = compute_first
        self.works = []
        self.group = COMM_OVERLAP_CONFIG.get_tp_group()
        self.world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        self.input_slice = input_data.shape[0] // self.split_num
        self.init_output_space(input_data, weight_shape_list, compute_first)

    def init_output_space(self, input_data, weight_shape_list, compute_first):
        if weight_shape_list is None:
            self.compute_output_shape_slice = list(input_data.shape)
        else:
            check_equal(input_data.shape[-1], weight_shape_list[0], error_info="In COCParallel, input_data should be of \
                        shape [m,k] and weight_shape_list should be [k,n]")
            self.compute_output_shape_slice = infer_matmul_out_shape(list(input_data.shape), weight_shape_list)
        self.output = self.allocate_output_memory()
        self.output_slice = self.output.shape[0] // self.split_num
        if compute_first:
            self.comm_output = self.output
        else:
            self.comm_output = self.allocate_communicate_memory_for_communicate_first()
        self.comm_slice = self.comm_output.shape[0] // self.split_num

    def get_dim_size_after_comm(self, dim_size):
        if self.comm_type == CommunicationType.ALL_GATHER:
            dim_size[0] = dim_size[0] * self.world_size
        elif self.comm_type == CommunicationType.REDUCE_SCATTER:
            dim_size[0] = dim_size[0] // self.world_size
        elif self.comm_type == CommunicationType.ALL_REDUCE:
            pass
        else:
            raise ValueError("Invalid comm_type.")
        return dim_size

    def allocate_output_memory(self):
        # No matter compute first or communicate first, the output shape remains the same
        output_dim_size = self.get_dim_size_after_comm(self.compute_output_shape_slice)
        output_ = torch.empty(output_dim_size, dtype=self.input_data.dtype,
                              device=torch.npu.current_device(), requires_grad=False)
        return output_

    def allocate_communicate_memory_for_communicate_first(self):
        dim_size = list(self.input_data.shape)
        dim_size = self.get_dim_size_after_comm(dim_size)
        comm_output = torch.empty(dim_size, dtype=self.input_data.dtype,
                                  device=torch.npu.current_device(), requires_grad=False)
        return comm_output

    def run_synchronize(self):
        for work in self.works:
            work.wait()
        return self.comm_output

    def run(self):
        if self.compute_first:
            return self.run_compute_first()
        else:
            return self.run_communicate_first()

    def comm_fcn(self, i, input_):
        if self.comm_type == CommunicationType.ALL_GATHER:
            output_ = self.comm_output[i * self.comm_slice: (i + 1) * self.comm_slice]
            work = torch.distributed._all_gather_base(output_, input_.contiguous(), group=self.group, async_op=True)
        elif self.comm_type == CommunicationType.REDUCE_SCATTER:
            output_ = self.comm_output[i * self.comm_slice: (i + 1) * self.comm_slice]
            work = torch.distributed._reduce_scatter_base(output_, input_.contiguous(), group=self.group, async_op=True)
        elif self.comm_type == CommunicationType.ALL_REDUCE:
            # all_reduce interface currently only supports overwriting the same address of input
            output_ = input_
            work = torch.distributed.all_reduce(output_, group=self.group, async_op=True)
        else:
            raise ValueError("Invalid comm_type.")
        return work, output_

    def get_input_slice(self, i):
        return self.input_data[i * self.input_slice: (i + 1) * self.input_slice]

    def run_compute_first(self):
        compute_outputs = []
        for i in range(self.split_num):
            input_slice = self.get_input_slice(i)
            if self.comm_type == CommunicationType.ALL_REDUCE:
                compute_output = self.output[i * self.comm_slice: (i + 1) * self.comm_slice]
                self.compute_fcn(input_tensor=input_slice, output_tensor=compute_output)
            else:
                compute_output = self.compute_fcn(input_slice)
            compute_outputs.append(compute_output)
            work, _ = self.comm_fcn(i, compute_output)
            self.works.append(work)

        if self.synchronize:
            return self.run_synchronize()
        else:
            return self.output, self.works

    def get_output_slice(self, i):
        return self.output[i * self.output_slice: (i + 1) * self.output_slice]

    def run_communicate_first(self):
        check_equal(self.synchronize, True, error_info="In COCParallel, must synchronize before return if communicate \
                    first")
        pre_work = None
        pre_output = None
        outputs = []

        for i in range(self.split_num):
            input_slice = self.get_input_slice(i)
            if self.comm_type == CommunicationType.ALL_REDUCE:
                input_ = torch.empty_like(input_slice).copy_(input_slice)
            else:
                input_ = input_slice
            work, output_i = self.comm_fcn(i, input_)
            outputs.append(output_i)

            self.works.append(work)

            if pre_output is not None:
                pre_work.wait()
                self.compute_fcn(input_tensor=pre_output, output_tensor=self.get_output_slice(i - 1))

            pre_work = work
            pre_output = output_i

        pre_work.wait()
        self.compute_fcn(input_tensor=pre_output, output_tensor=self.get_output_slice(self.split_num - 1))
        return self.output


def shuffle_as_coc_reduce_scatter(input_, world_size, parallel_num):
    per = input_.shape[0] // parallel_num // world_size
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [parallel_num, world_size, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def shuffle_as_coc_all_gather(input_, world_size, parallel_num):
    per = input_.shape[0] // parallel_num // world_size
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [world_size, parallel_num, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def infer_matmul_out_shape(shape_a, shape_b):
    shape_a[-1] = shape_b[-1]
    return shape_a


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