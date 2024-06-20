# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd. All rights reserved.

import torch

from .async_comm_utils import (async_all_to_all, async_fw_ar_rs, get_fw_ag_output, async_fw_all_gather,
                               async_bw_all_gather)

ASYNC_BW_ALL_GATHER_COUNT = 0


def get_async_bw_all_gather_count():
    return ASYNC_BW_ALL_GATHER_COUNT


class PipeExpertUtil:
    multi_data = None
    num_local_experts = None
    slice_seq_size = None
    ep_size = None

    first_a2a_event = []
    second_a2a_event = []
    fw_ag_event = []
    bw_ag_event = []
    ar_rs_event = []

    @classmethod
    def set_parameters(cls, args, slice_seq_size):
        cls.multi_data = args[4]
        cls.num_local_experts = args[2]
        cls.slice_seq_size = slice_seq_size
        cls.ep_size = args[1]

    @classmethod
    def get_first_a2a_event(cls):
        return cls.first_a2a_event

    @classmethod
    def get_second_a2a_event(cls):
        return cls.second_a2a_event

    @classmethod
    def get_fw_ag_event(cls):
        return cls.fw_ag_event

    @classmethod
    def get_bw_ag_event(cls):
        return cls.bw_ag_event

    @classmethod
    def get_ar_rs_event(cls):
        return cls.ar_rs_event

    @classmethod
    def deal_data(cls, origin_data, output_data):
        for i in range(cls.num_local_experts):
            for j in range(cls.multi_data):
                if j != cls.multi_data - 1:
                    output_data.append(origin_data[i * cls.ep_size: (i + 1) * cls.ep_size,
                                       j * cls.slice_seq_size: (j + 1) * cls.slice_seq_size].clone().contiguous())
                else:
                    output_data.append(origin_data[i * cls.ep_size: (i + 1) * cls.ep_size,
                                       j * cls.slice_seq_size:].clone().contiguous())

    @classmethod
    def first_a2a_when_not_multi_stream(cls, input_data_list):
        for i in range(cls.num_local_experts):
            for j in range(cls.multi_data):
                input_data_list[j + i * cls.multi_data], handle = async_all_to_all(
                    input_data_list[j + i * cls.multi_data])
                cls.first_a2a_event.append(handle)

    @classmethod
    def fw_bw_ag_after_first_a2a_when_not_multi_stream(cls, input_data_list, num_local_experts_index, multi_data_index,
                                                       is_fw_ag):
        if (num_local_experts_index * cls.multi_data + multi_data_index) == 0:
            cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index].wait()
            if is_fw_ag:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index], handle = async_fw_all_gather(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.fw_ag_event.append(handle)
            else:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index], handle = async_bw_all_gather(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.bw_ag_event.append(handle)
        if (num_local_experts_index * cls.multi_data + multi_data_index) < (cls.num_local_experts * cls.multi_data - 1):
            cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index + 1].wait()
            if is_fw_ag:
                if num_local_experts_index * cls.multi_data + multi_data_index == 0:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_fw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1], None, True)
                else:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_fw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1])
                cls.fw_ag_event.append(handle)
            else:
                if num_local_experts_index * cls.multi_data + multi_data_index == 0:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_bw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1], None, True)
                else:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_bw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1])
                cls.bw_ag_event.append(handle)

    @classmethod
    def fw_bw_ag_after_first_a2a_when_multi_stream(cls, input_data_list, num_local_experts_index, multi_data_index,
                                                   is_fw_ag):
        if num_local_experts_index * cls.multi_data + multi_data_index == 0:
            input_data_list[num_local_experts_index * cls.multi_data + multi_data_index], handle = async_all_to_all(
                input_data_list[num_local_experts_index * cls.multi_data + multi_data_index])
            cls.first_a2a_event.append(handle)
            if is_fw_ag:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index], handle = async_fw_all_gather(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index],
                    cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.fw_ag_event.append(handle)
            else:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index], handle = async_bw_all_gather(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index],
                    cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.bw_ag_event.append(handle)
        if num_local_experts_index * cls.multi_data + multi_data_index < (cls.num_local_experts * cls.multi_data - 1):
            if is_fw_ag:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_all_to_all(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                    cls.fw_ag_event[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.first_a2a_event.append(handle)
                if num_local_experts_index * cls.multi_data + multi_data_index == 0:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_fw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        True)
                else:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_fw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index + 1])
                cls.fw_ag_event.append(handle)
            else:
                input_data_list[
                    num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_all_to_all(
                    input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                    cls.bw_ag_event[num_local_experts_index * cls.multi_data + multi_data_index])
                cls.first_a2a_event.append(handle)
                if num_local_experts_index * cls.multi_data + multi_data_index == 0:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_bw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        True)
                else:
                    input_data_list[
                        num_local_experts_index * cls.multi_data + multi_data_index + 1], handle = async_bw_all_gather(
                        input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1],
                        cls.first_a2a_event[num_local_experts_index * cls.multi_data + multi_data_index + 1])
                cls.bw_ag_event.append(handle)

    @classmethod
    def a2a_after_ar_rs(cls, num_local_experts_index, multi_data_index, output_list_for_each_multi_data,
                        outputs_list_for_each_local_expert):
        if cls.multi_data == 1:
            if num_local_experts_index > 0:
                cls.ar_rs_event[num_local_experts_index - 1].wait()
                outputs_list_for_each_local_expert[num_local_experts_index - 1][0], handle = async_all_to_all(
                    outputs_list_for_each_local_expert[num_local_experts_index - 1][0])
                cls.second_a2a_event.append(handle)
        else:
            if multi_data_index > 0:
                cls.ar_rs_event[num_local_experts_index * cls.multi_data + multi_data_index - 1].wait()
                output_list_for_each_multi_data[multi_data_index - 1], handle = async_all_to_all(
                    output_list_for_each_multi_data[multi_data_index - 1])
                cls.second_a2a_event.append(handle)
            else:
                if num_local_experts_index > 0:
                    cls.ar_rs_event[num_local_experts_index * cls.multi_data + multi_data_index - 1].wait()
                    outputs_list_for_each_local_expert[num_local_experts_index - 1][
                        cls.multi_data - 1], handle = async_all_to_all(
                        outputs_list_for_each_local_expert[num_local_experts_index - 1][cls.multi_data - 1])
                    cls.second_a2a_event.append(handle)

    @classmethod
    def a2a_for_final_data(cls, outputs_list_for_each_local_expert):
        if cls.multi_data == 1:
            cls.ar_rs_event[cls.num_local_experts - 1].wait()
            outputs_list_for_each_local_expert[cls.num_local_experts - 1][0], handle = async_all_to_all(
                outputs_list_for_each_local_expert[cls.num_local_experts - 1][0])
            cls.second_a2a_event.append(handle)
        else:
            cls.ar_rs_event[cls.num_local_experts * cls.multi_data - 1].wait()
            outputs_list_for_each_local_expert[cls.num_local_experts - 1][
                cls.multi_data - 1], handle = async_all_to_all(
                outputs_list_for_each_local_expert[cls.num_local_experts - 1][cls.multi_data - 1])
            cls.second_a2a_event.append(handle)


class PipeExpert(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Experts, *args):
        inputs = args[0]
        num_local_experts = args[2]
        sequence_parallel = args[3]
        multi_data = args[4]
        multi_stream = args[5]

        input_shape = list(inputs.size())
        slice_seq_size = input_shape[1] // multi_data

        ctx.num_local_experts = num_local_experts
        ctx.sequence_parallel = sequence_parallel
        ctx.multi_data = multi_data
        ctx.multi_stream = multi_stream

        inputs_list = []
        outputs_list_for_each_local_expert = []
        input_list_before_expert = []
        output_list_after_expert = []
        PipeExpertUtil.set_parameters(args, slice_seq_size)
        PipeExpertUtil.deal_data(inputs, inputs_list)
        inputs.untyped_storage().resize_(0)

        if not multi_stream:
            PipeExpertUtil.first_a2a_when_not_multi_stream(inputs_list)

        for i in range(num_local_experts):
            output_list_for_each_multi_data = []
            for j in range(multi_data):
                if sequence_parallel:
                    if not multi_stream:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_not_multi_stream(inputs_list, i, j, True)
                    else:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_multi_stream(inputs_list, i, j, True)

                    PipeExpertUtil.get_fw_ag_event()[i * multi_data + j].wait()
                else:
                    PipeExpertUtil.get_first_a2a_event()[i * multi_data + j].wait()

                input_detach_before_expert = inputs_list[i * multi_data + j].detach()
                input_detach_before_expert.requires_grad = True
                input_list_before_expert.append(input_detach_before_expert)

                with torch.enable_grad():
                    output_expert = Experts.experts[i](input_list_before_expert[i * multi_data + j])
                if sequence_parallel:
                    get_fw_ag_output().pop(0)

                if isinstance(output_expert, tuple):
                    output_expert = output_expert[0]

                output_list_after_expert.append(output_expert)
                output_detach_after_expert = output_expert.detach()

                if not multi_stream:
                    PipeExpertUtil.a2a_after_ar_rs(i, j, output_list_for_each_multi_data,
                                                   outputs_list_for_each_local_expert)

                    output_detach_after_expert, handle = async_fw_ar_rs(output_detach_after_expert, sequence_parallel)
                    output_list_for_each_multi_data.append(output_detach_after_expert)
                    PipeExpertUtil.get_ar_rs_event().append(handle)
                else:
                    # all2all allgather wait release memory
                    PipeExpertUtil.get_first_a2a_event()[i * multi_data + j].wait()
                    PipeExpertUtil.get_fw_ag_event()[i * multi_data + j].wait()

                    output_detach_after_expert, handle = async_fw_ar_rs(output_detach_after_expert, sequence_parallel)
                    PipeExpertUtil.get_ar_rs_event().append(handle)
                    output_detach_after_expert, handle = async_all_to_all(output_detach_after_expert,
                                                                          PipeExpertUtil.get_ar_rs_event()[
                                                                              i * multi_data + j])
                    output_list_for_each_multi_data.append(output_detach_after_expert)
                    PipeExpertUtil.get_second_a2a_event().append(handle)

            outputs_list_for_each_local_expert.append(output_list_for_each_multi_data)

        if not multi_stream:
            PipeExpertUtil.a2a_for_final_data(outputs_list_for_each_local_expert)

        for i in range(num_local_experts):
            for j in range(multi_data):
                PipeExpertUtil.get_second_a2a_event()[i * multi_data + j].wait()
                # reduce scatter
                PipeExpertUtil.get_ar_rs_event()[i * multi_data + j].wait()

        PipeExpertUtil.get_first_a2a_event().clear()
        PipeExpertUtil.get_second_a2a_event().clear()
        PipeExpertUtil.get_fw_ag_event().clear()
        PipeExpertUtil.get_ar_rs_event().clear()

        output_forward = torch.cat(
            [torch.cat((outputs_list_for_each_local_expert[i]), dim=1) for i in range(num_local_experts)], dim=0)

        for tensor in output_list_after_expert:
            tensor.untyped_storage().resize_(0)

        ctx.save_for_backward(*tuple(input_list_before_expert), *tuple(output_list_after_expert))

        return output_forward

    @staticmethod
    def backward(ctx, *args):
        num_local_experts = ctx.num_local_experts
        sequence_parallel = ctx.sequence_parallel
        multi_stream = ctx.multi_stream
        multi_data = ctx.multi_data

        saved_tensors_list = list(ctx.saved_tensors)
        input_list_before_expert = saved_tensors_list[:len(saved_tensors_list) // 2]
        output_list_after_expert = saved_tensors_list[len(saved_tensors_list) // 2:]

        grad_outputs = args[0]
        global ASYNC_BW_ALL_GATHER_COUNT
        ASYNC_BW_ALL_GATHER_COUNT = 0

        grad_outputs_list = []
        grad_outputs_list_for_each_local_expert = []
        PipeExpertUtil.deal_data(grad_outputs, grad_outputs_list)
        grad_outputs.storage().resize_(0)

        if not multi_stream:
            PipeExpertUtil.first_a2a_when_not_multi_stream(grad_outputs_list)

        for i in range(num_local_experts):
            grad_output_list_for_each_multi_data = []
            for j in range(multi_data):
                if sequence_parallel:
                    if not multi_stream:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_not_multi_stream(grad_outputs_list, i, j, False)

                    else:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_multi_stream(grad_outputs_list, i, j, False)

                    PipeExpertUtil.get_bw_ag_event()[i * multi_data + j].wait()
                else:
                    PipeExpertUtil.get_first_a2a_event()[i * multi_data + j].wait()
                ASYNC_BW_ALL_GATHER_COUNT += 1
                output_list_after_expert[i * multi_data + j].backward(grad_outputs_list[i * multi_data + j])
                grads_expert_output = input_list_before_expert[i * multi_data + j].grad

                grads_expert_output, handle = async_all_to_all(grads_expert_output)
                grad_output_list_for_each_multi_data.append(grads_expert_output)
                PipeExpertUtil.get_second_a2a_event().append(handle)
            grad_outputs_list_for_each_local_expert.append(grad_output_list_for_each_multi_data)

        for i in range(num_local_experts):
            for j in range(multi_data):
                PipeExpertUtil.get_second_a2a_event()[i * multi_data + j].wait()

        for event in PipeExpertUtil.get_first_a2a_event():
            event.wait()

        for event in PipeExpertUtil.get_bw_ag_event():
            event.wait()

        PipeExpertUtil.get_second_a2a_event().clear()
        PipeExpertUtil.get_first_a2a_event().clear()
        PipeExpertUtil.get_bw_ag_event().clear()
        grad_output = torch.cat(
            [torch.cat((grad_outputs_list_for_each_local_expert[i]), dim=1) for i in range(num_local_experts)], dim=0)

        return None, grad_output, None, None, None, None, None
