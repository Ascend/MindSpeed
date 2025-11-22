# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from megatron.training import get_args
from mindspeed.core.memory.swap_attention.prefetch import SwapTensor, get_layer_id, post_backward_hook_func


def sawp_attention_init(self, tensor, layer_name):
    self.tensor = tensor
    self.size = tensor.size()
    self.storage_size = tensor.storage().size()
    self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

    self.d2h_event = None
    self.h2d_event = torch.npu.Event()

    self.stat = "device"
    self.layer_name = layer_name

    self.prefetch_data_ptr = tensor.data_ptr()
    self.storage_data_ptr = tensor.storage().data_ptr()
    self.layer_id = None
    self.first_tensor = False
    self.last_tensor = False
    # MSA adapt: modify is_slice_tensor
    self.is_slice_tensor = tensor.storage().size() // tensor.element_size() != tensor.numel()
    self.stream = None
    self.layer_index = 0


@staticmethod
def no_swap_tensor(ori_tensor):
    if ori_tensor.numel() * ori_tensor.element_size() * 2 < 1024 * 1024:
        return True
    if ori_tensor.grad_fn is None:
        return True
    if ori_tensor.storage().size() == 0:
        return True
    # MSA adapt: Divided by element_size()
    if ori_tensor.storage().size() // ori_tensor.element_size() != ori_tensor.numel():
        return True
    if ori_tensor._base is not None and ori_tensor._base.dim() >= 5:
        return True

    return False


def pack_hook(self, ori_tensor):
    """hook function for forward to device to host
    Args:
        ori_tensor (Tensor): Device tensor expected to be swaped to host
    Returns:
        class: Information about swap and swaped host Tensor.
    """
    args = get_args()
    if args.eval_interval:
        if args.curr_iteration % args.eval_interval != 0:
            self.eval_end_flag = False
        if args.curr_iteration and args.curr_iteration % args.eval_interval == 0 and not self.eval_end_flag:
            self.prefetch_data_ptr_list = []
            self.prefetch_list = []
            self.slice_tensor_storage_ptr_list = []
            self.eval_end_flag = True

    # MSA adapt:
    if "image_encoder" in self.layer_name:
        return ori_tensor

    if self.no_swap_tensor(ori_tensor):
        return ori_tensor
    swap_tensor = SwapTensor(ori_tensor, self.layer_name)
    if not self.swap_tensors:
        swap_tensor.first_tensor = True
    # Records the slice tensor status.
    # MSA adapt: Divided by element_size()
    if ori_tensor.storage().size() // ori_tensor.element_size() != ori_tensor.numel():
        swap_tensor.is_slice_tensor = True
        if ori_tensor.storage().data_ptr() not in self.slice_tensor_storage_ptr:
            if self.swap_tensors and self.swap_tensors[0].layer_id != 0:
                self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()] = \
                    [f'{len(self.prefetch_list) - 1}_{len(self.swap_tensors)}']
            else:
                self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()] = \
                    [f'{len(self.prefetch_list)}_{len(self.swap_tensors)}']
        else:
            if self.swap_tensors and self.swap_tensors[0].layer_id != 0:
                self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()].append(
                    f'{len(self.prefetch_list) - 1}_{len(self.swap_tensors)}')
            else:
                self.slice_tensor_storage_ptr[ori_tensor.storage().data_ptr()].append(
                    f'{len(self.prefetch_list)}_{len(self.swap_tensors)}')

    # Records the same data_ptr tensor status.
    if ori_tensor.storage().data_ptr() in self.data_ptr:
        self.swap_tensors[self.data_ptr[ori_tensor.storage().data_ptr()]].stat = 'h2d'
        swap_tensor.stat = 'd2h'
        swap_tensor.tensor_cpu = self.swap_tensors[self.data_ptr[ori_tensor.storage().data_ptr()]].tensor_cpu
        self.data_ptr[ori_tensor.storage().data_ptr()] = len(self.swap_tensors)
    else:
        self.data_ptr[ori_tensor.storage().data_ptr()] = len(self.swap_tensors)

    swap_tensor.launch_d2h(self.prefetch_stream)
    swap_tensor.stream = self.prefetch_stream
    swap_tensor.layer_id = int(get_layer_id(swap_tensor.layer_name))
    self.swap_tensors.append(swap_tensor)
    self.forward_flag = True
    return swap_tensor


# register prefetch before backward, prefetch h2d
def prefetch_register_post_backward_hook(module, name, prefetch_args):
    # MSA adapt: Use register_full_backward_pre_hook function instead of register_backward_hook.
    module.register_full_backward_pre_hook(post_backward_hook_func(name, prefetch_args))
