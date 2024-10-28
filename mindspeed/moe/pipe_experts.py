# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd. All rights reserved.

import torch

from megatron.training import get_args
from mindspeed.core.weight_grad_store import WeightGradStore

from .async_comm_utils import (async_all_to_all, async_fw_ar_rs, get_fw_ag_output, async_fw_all_gather,
                               async_all_gather)

ASYNC_BW_ALL_GATHER_COUNT = 0
FLAG_GRAD_REDUCE = True


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
                output_data.append(origin_data[i * cls.ep_size: (i + 1) * cls.ep_size,
                                   j * cls.slice_seq_size: (j + 1) * cls.slice_seq_size].clone().contiguous())

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
        index = num_local_experts_index * cls.multi_data + multi_data_index
        if index == 0 and get_args().ampipe_degree <= 1:
            cls.first_a2a_event[index].wait()
            if is_fw_ag:
                input_data_list[index], handle = async_fw_all_gather(input_data_list[index])
                cls.fw_ag_event.append(handle)
            else:
                if get_args().use_nanopipe and WeightGradStore.is_decoupleBlock:
                    WeightGradStore.save_grad_output(input_data_list[num_local_experts_index * cls.multi_data + multi_data_index].clone().detach())
                input_data_list[index], handle = async_all_gather(input_data_list[index], is_bwd=True)
                cls.bw_ag_event.append(handle)
        if index < (cls.num_local_experts * cls.multi_data - 1):
            cls.first_a2a_event[index + 1].wait()
            if is_fw_ag:
                if index == 0 and not get_args().use_nanopipe:
                    input_data_list[index + 1], handle = async_fw_all_gather(input_data_list[index + 1], None, True)
                else:
                    input_data_list[index + 1], handle = async_fw_all_gather(input_data_list[index + 1])
                cls.fw_ag_event.append(handle)
            else:
                if get_args().use_nanopipe and WeightGradStore.is_decoupleBlock:
                    WeightGradStore.save_grad_output(input_data_list[num_local_experts_index * cls.multi_data + multi_data_index + 1].clone().detach())
                if index == 0 and not get_args().use_nanopipe:
                    input_data_list[index + 1], handle = async_all_gather(input_data_list[index + 1], None, True, True)
                else:
                    input_data_list[index + 1], handle = async_all_gather(input_data_list[index + 1], is_bwd=True)
                cls.bw_ag_event.append(handle)

    @classmethod
    def fw_bw_ag_after_first_a2a_when_multi_stream(cls, input_data_list, num_local_experts_index, multi_data_index,
                                                   is_fw_ag):
        index = num_local_experts_index * cls.multi_data + multi_data_index
        if index == 0:
            input_data_list[index], handle = async_all_to_all(input_data_list[index])
            cls.first_a2a_event.append(handle)
            if is_fw_ag:
                input_data_list[index], handle = async_fw_all_gather(
                    input_data_list[index], cls.first_a2a_event[index])
                cls.fw_ag_event.append(handle)
            else:
                input_data_list[index], handle = async_all_gather(
                    input_data_list[index], cls.first_a2a_event[index], is_bwd=True)
                cls.bw_ag_event.append(handle)
        if index < (cls.num_local_experts * cls.multi_data - 1):
            if is_fw_ag:
                input_data_list[index + 1], handle = async_all_to_all(
                    input_data_list[index + 1], cls.fw_ag_event[index])
                cls.first_a2a_event.append(handle)
                if index == 0 and not get_args().use_nanopipe:
                    input_data_list[index + 1], handle = async_fw_all_gather(
                        input_data_list[index + 1], cls.first_a2a_event[index + 1], True)
                else:
                    input_data_list[index + 1], handle = async_fw_all_gather(
                        input_data_list[index + 1], cls.first_a2a_event[index + 1])
                cls.fw_ag_event.append(handle)
            else:
                input_data_list[index + 1], handle = async_all_to_all(
                    input_data_list[index + 1], cls.bw_ag_event[index])
                cls.first_a2a_event.append(handle)
                if index == 0 and not get_args().use_nanopipe:
                    input_data_list[index + 1], handle = async_all_gather(
                        input_data_list[index + 1], cls.first_a2a_event[index + 1], True, True)
                else:
                    input_data_list[index + 1], handle = async_all_gather(
                        input_data_list[index + 1], cls.first_a2a_event[index + 1], is_bwd=True)
                cls.bw_ag_event.append(handle)

    @classmethod
    def fw_a2a_after_ar_rs_when_not_multi_stream(cls, num_local_experts_index, multi_data_index,
                                                 output_list_for_each_multi_data, outputs_list_for_each_local_expert):
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
    def fw_a2a_for_final_data_when_not_multi_stream(cls, outputs_list_for_each_local_expert):
        cls.ar_rs_event[cls.num_local_experts * cls.multi_data - 1].wait()
        outputs_list_for_each_local_expert[cls.num_local_experts - 1][
            cls.multi_data - 1], handle = async_all_to_all(
            outputs_list_for_each_local_expert[cls.num_local_experts - 1][cls.multi_data - 1])
        cls.second_a2a_event.append(handle)


class PipeExpert(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Experts, *args):
        inputs = args[0]
        ep_size = args[1]
        num_local_experts = args[2]
        sequence_parallel = args[3]
        multi_data = args[4]
        multi_stream = args[5]

        ctx.num_local_experts = num_local_experts
        ctx.sequence_parallel = sequence_parallel
        ctx.multi_data = multi_data
        ctx.multi_stream = multi_stream

        inputs_list = []
        ampipe_degree = get_args().ampipe_degree
        ctx.ampipe_degree = ampipe_degree
        if ampipe_degree > 1:
            PipeExpertUtil.first_a2a_event = args[6]
            PipeExpertUtil.second_a2a_event = args[7]
            PipeExpertUtil.fw_ag_event = args[8]
            ctx.hidden_size = hidden_size = args[9]
            save_tensors_list = args[10]
            inputs_list = inputs
            slice_seq_size = 0
        else:
            input_shape = list(inputs.size())
            if multi_data > input_shape[1]:
                raise ValueError('--pipe-experts-multi-data cannot be greater than experts capacity')
            slice_seq_size = input_shape[1] // multi_data
            if input_shape[1] % multi_data != 0:
                slice_seq_size += 1

        outputs_list_for_each_local_expert = []
        input_list_before_expert = []
        output_list_after_expert = []
        PipeExpertUtil.set_parameters(args, slice_seq_size)

        if ampipe_degree <= 1:
            PipeExpertUtil.deal_data(inputs, inputs_list)
            inputs.untyped_storage().resize_(0)

        if not multi_stream and ampipe_degree <= 1:
            PipeExpertUtil.first_a2a_when_not_multi_stream(inputs_list)

        for i in range(num_local_experts):
            output_list_for_each_multi_data = []
            for j in range(multi_data):
                if sequence_parallel:
                    if not multi_stream:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_not_multi_stream(inputs_list, i, j, True)
                    elif ampipe_degree <= 1:
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
                    output_expert, bias = output_expert
                    if bias is not None:
                        with torch.enable_grad():
                            output_expert = output_expert + bias

                output_list_after_expert.append(output_expert)
                output_detach_after_expert = output_expert.detach()

                if not multi_stream:
                    PipeExpertUtil.fw_a2a_after_ar_rs_when_not_multi_stream(i, j, output_list_for_each_multi_data,
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
            PipeExpertUtil.fw_a2a_for_final_data_when_not_multi_stream(outputs_list_for_each_local_expert)

        for i in range(num_local_experts):
            for j in range(multi_data):
                PipeExpertUtil.get_second_a2a_event()[i * multi_data + j].wait()
                # reduce scatter
                PipeExpertUtil.get_ar_rs_event()[i * multi_data + j].wait()

        PipeExpertUtil.get_first_a2a_event().clear()
        PipeExpertUtil.get_second_a2a_event().clear()
        PipeExpertUtil.get_fw_ag_event().clear()
        PipeExpertUtil.get_ar_rs_event().clear()

        for tensor in output_list_after_expert:
            tensor.untyped_storage().resize_(0)

        ctx.input_list_before_expert = input_list_before_expert

        if 1 < ampipe_degree <= multi_data:
            save_tensors_list.extend(output_list_after_expert)
            output_list = []
            for i in range(num_local_experts):
                exp_out_list = []
                for j in range(ampipe_degree):
                    ampipe_tokens = outputs_list_for_each_local_expert[i][
                                    j * multi_data // ampipe_degree:(j + 1) * multi_data // ampipe_degree]
                    ampipe_tokens = torch.cat(ampipe_tokens, dim=1)
                    exp_out_list.append(ampipe_tokens)
                output_list.append(exp_out_list)
            output_forward = [
                torch.cat([i[j] for i in output_list], dim=1).reshape(num_local_experts * ep_size, -1, hidden_size) for
                j in range(ampipe_degree)]

        else:
            ctx.save_for_backward(*tuple(output_list_after_expert))
            output_forward = torch.cat([torch.cat((outputs_list_for_each_local_expert[i]), dim=1) for i in range(num_local_experts)], dim=0)

        return output_forward

    @staticmethod
    def backward(ctx, *args):
        num_local_experts = ctx.num_local_experts
        sequence_parallel = ctx.sequence_parallel
        multi_stream = ctx.multi_stream
        multi_data = ctx.multi_data
        ampipe_degree = ctx.ampipe_degree

        grad_outputs = args[0]
        global ASYNC_BW_ALL_GATHER_COUNT
        ASYNC_BW_ALL_GATHER_COUNT = 0

        grad_outputs_list = []
        grad_outputs_list_for_each_local_expert = []
        if ampipe_degree > 1:
            PipeExpertUtil.first_a2a_event = args[1]
            PipeExpertUtil.bw_ag_event = args[2]
            PipeExpertUtil.second_a2a_event = args[3]
            output_list_after_expert = args[4]
            grad_outputs_list = grad_outputs
        else:
            output_list_after_expert = list(ctx.saved_tensors)

        if ampipe_degree <= 1:
            PipeExpertUtil.deal_data(grad_outputs, grad_outputs_list)
            grad_outputs.storage().resize_(0)

        if not multi_stream and ampipe_degree <= 1:
            PipeExpertUtil.first_a2a_when_not_multi_stream(grad_outputs_list)

        for i in range(num_local_experts):
            grad_output_list_for_each_multi_data = []
            global FLAG_GRAD_REDUCE
            FLAG_GRAD_REDUCE = False
            for j in range(multi_data):
                if sequence_parallel:
                    if not multi_stream:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_not_multi_stream(grad_outputs_list, i, j, False)

                    elif ampipe_degree <= 1:
                        PipeExpertUtil.fw_bw_ag_after_first_a2a_when_multi_stream(grad_outputs_list, i, j, False)

                    PipeExpertUtil.get_bw_ag_event()[i * multi_data + j].wait()
                else:
                    PipeExpertUtil.get_first_a2a_event()[i * multi_data + j].wait()
                ASYNC_BW_ALL_GATHER_COUNT += 1
                if j == multi_data - 1:
                    FLAG_GRAD_REDUCE = True
                output_list_after_expert[i * multi_data + (multi_data // ampipe_degree + j) % multi_data].backward(
                    grad_outputs_list[i * multi_data + j])
                grads_expert_output = ctx.input_list_before_expert[
                    i * multi_data + (multi_data // ampipe_degree + j) % multi_data].grad

                grads_expert_output, handle = async_all_to_all(grads_expert_output)
                grad_output_list_for_each_multi_data.append(grads_expert_output)
                PipeExpertUtil.get_second_a2a_event().append(handle)
            grad_outputs_list_for_each_local_expert.append(grad_output_list_for_each_multi_data)

        if 1 < ampipe_degree <= multi_data:
            for i in range(num_local_experts):
                for j in range(multi_data):
                    index = i * multi_data + j
                    if index < len(PipeExpertUtil.get_second_a2a_event()) - 1:
                        PipeExpertUtil.get_second_a2a_event()[index].wait()

            for event in PipeExpertUtil.get_first_a2a_event():
                event.wait()

            for event in PipeExpertUtil.get_bw_ag_event():
                event.wait()

            PipeExpertUtil.get_first_a2a_event().clear()
            PipeExpertUtil.get_bw_ag_event().clear()

            output_list = []
            for i in range(num_local_experts):
                exp_out_list = []
                for j in range(ampipe_degree):
                    ampipe_tokens = grad_outputs_list_for_each_local_expert[i][
                                    j * multi_data // ampipe_degree:(j + 1) * multi_data // ampipe_degree]
                    exp_out_list.append(ampipe_tokens)
                output_list.append(exp_out_list)
            second_ampipe_grad_input = torch.cat([torch.cat(i[0], dim=1) for i in output_list], dim=1)
            second_ampipe_grad_input = second_ampipe_grad_input.reshape((num_local_experts * PipeExpertUtil.ep_size), -1,
                                                                        ctx.hidden_size)
            first_ampipe_grad_inputs = [i[1] for i in output_list]

            return [first_ampipe_grad_inputs, second_ampipe_grad_input]
        else:
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

            ctx.input_list_before_expert = None
            return None, grad_output, None, None, None, None, None
