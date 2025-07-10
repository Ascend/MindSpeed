# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from ast import literal_eval
import math
import numpy as np
from sklearn.linear_model import LinearRegression

from mindspeed.auto_tuning.config.search_config import SearchConfig


def cal_new_shape_new(cal_arr, search_cfg: SearchConfig, seq_ratio: int):
    tp = search_cfg.tp
    cp = search_cfg.cp
    ep = search_cfg.ep or 1
    mbs = search_cfg.mbs
    num_experts = search_cfg.num_experts or 1
    cal_arr = literal_eval(cal_arr)
    result_arr = []
    mbs_flag = False
    if mbs > 1:
        mbs_flag = True
    for inner_arr in cal_arr:
        result = []
        for item in inner_arr:
            temp = int(item) * seq_ratio
            # dis = int(10 * (item - float(int(item))))可能会出现浮点数精度问题
            # 为解决浮点数问题，使用以下代码
            dis = str(10 * (item - float(int(item))))
            dis = int(dis[0]) + int(1 if int(dis[2]) >= 5 else 0)

            if 1 & dis == 1:  # 按位与运算，代表ep存在
                temp = temp * ep / num_experts
            if 2 & dis == 2:  # cp存在
                temp = temp / cp
                if mbs_flag:
                    temp = temp * mbs
            if 4 & dis == 4:  # tp存在
                temp = temp / tp
            result.append(math.ceil(temp))
        result_arr.append(result)
    return result_arr


def cal_new_shape_tce(cal_arr, search_cfg: SearchConfig, seq_ratio: int):
    result_cal_arr = cal_new_shape_new(cal_arr, search_cfg, seq_ratio)
    result_str = ';'.join([','.join(map(str, arr)) if arr else '' for arr in result_cal_arr])
    return result_str


def mul_shape(shape):
    result = 1
    for item in shape:
        if item != 0:
            result *= item
    return result


def model_operator_with_shape(history_result_list):
    if len(history_result_list) <= 0:
        return 0, 0
    x_arr = []
    y_arr = []
    for history in history_result_list:
        x_arr.append([cal_operator_flops(history.input_shape, history.output_shape, history.types)])
        y_arr.append([history.duration])
    shape_model_w, shape_model_b = linear_regression(x_arr, y_arr)
    return shape_model_w, shape_model_b


def cal_operator_flops(input_shape, output_shape, types):
    input_shape_arr_before = []
    output_shape_arr = []
    if len(input_shape) < 1 or input_shape == ';':
        return 1
    for str_num in input_shape.split(';')[0].split(','):
        if str_num == '':
            return 1
        else:
            input_shape_arr_before.append(int(str_num))
    if len(output_shape) < 1 or output_shape == ';':
        return 1
    for str_num in output_shape.split(';')[0].split(','):
        if str_num == '':
            return 1
        else:
            output_shape_arr.append(int(str_num))
    # other operator flops
    x_item = mul_shape(input_shape_arr_before)

    # FLOPs(BatchMatMul) = b*x*y*n; [b, x, n] * [b, n, y] == [b, x, y]
    if types in ['BatchMatMul']:
        x_item = mul_shape(output_shape_arr)
        if input_shape_arr_before[1] in output_shape_arr:
            x_item *= input_shape_arr_before[2]
        else:
            x_item *= input_shape_arr_before[1]

    # FLOPs(MatMul) = x*y*n; [x, n] * [n, y] == [x, y]
    if types in ['MatMul', 'MatMulCommon', 'MatMulV3', 'MatMulV2', 'MatMulVX']:
        input_shape_arr_after = [int(str_num) for str_num in input_shape.split(';')[1].split(',')]
        x_item = 2 * mul_shape(output_shape_arr)
        if input_shape_arr_before[0] in output_shape_arr:
            x_item *= input_shape_arr_before[1]
        else:
            x_item *= input_shape_arr_before[0]
        # 输入矩阵A需要进行转置，产生额外FLOPs
        if output_shape_arr[0] != input_shape_arr_before[0]:
            x_item += 2 * mul_shape(input_shape_arr_before)
        # 输入矩阵B需要进行转置，产生额外FLOPs
        if output_shape_arr[1] != input_shape_arr_after[1]:
            x_item += 2 * mul_shape(input_shape_arr_after)

    if types in ['Mul', 'MulAiCore', 'ConcatD']:
        x_item = 0
        str_arr = input_shape.split(';')
        for arr in str_arr:
            if len(arr) > 0:
                int_arr = [int(str_num) for str_num in arr.split(',')]
                x_item += mul_shape(int_arr)

    if types in ['Slice', 'SliceAiCore']:
        x_item = 0
        str_arr = output_shape.split(';')
        for arr in str_arr:
            if len(arr) > 0:
                int_arr = [int(str_num) for str_num in arr.split(',')]
                x_item += mul_shape(int_arr)

    if types in ['FlashAttentionScore', 'FlashAttentionScoreGrad']:
        x_item = mul_shape(input_shape_arr_before)
        input_shape_arr_after_flash = []
        for str_num in input_shape.split(';')[1].split(','):
            if str_num != '':
                input_shape_arr_after_flash.append(int(str_num))
        x_tmp = input_shape_arr_after_flash[0] * x_item
        x_item += x_tmp

    return x_item


def cal_operator_duration_with_shape(shape_model_w, shape_model_b, flops):
    result_duration = float(shape_model_w) * flops + float(shape_model_b)
    if result_duration < 0:
        return 0
    return result_duration


def model_operator_with_tp(operator_notes_index_list):
    """
        方式一：相同TP的同一个index-name算子duration随TP线性下降duration ~ w / tp
        分别求出TP的比例a1，CP的比例a2
        最后结果是 d = model_w_tp / TP + model_w_cp / CP
    """
    result_tp = 0
    for operator_notes_index in operator_notes_index_list:
        result_tp = result_tp + operator_notes_index.tp * operator_notes_index.duration
    model_w_tp = result_tp / len(operator_notes_index_list)

    return model_w_tp, 0


def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    model = LinearRegression()
    model.fit(x, y)
    w = model.coef_[0]
    b = model.intercept_
    return w[0], b[0]


def print_operator_shape(print_list, information_level=0):
    levels = {
        1: ['name', 'index_name'],
        2: ['index_name', 'input_shape', 'output_shape'],
        3: ['index_name', 'duration', 'input_shape', 'output_shape']
    }
    attributes_list = levels.get(information_level, [])

    result = ", ".join(
        f"{attr}: {getattr(obj, attr)}" for obj in print_list for attr in attributes_list if hasattr(obj, attr))
    print(result)


def print_operator_cal_shape(print_list, information_level=0):
    levels = {
        1: ['index_name', 'input_cal', 'output_cal'],
        2: ['index_name', 'duration', 'input_cal', 'output_cal'],
        3: ['index_name', 'duration', 'input_cal', 'output_cal', 'input_shape', 'output_shape', 'types']
    }
    attributes_list = levels.get(information_level, [])

    result = ", ".join(
        f"{attr}: {getattr(obj, attr)}" for obj in print_list for attr in attributes_list if hasattr(obj, attr))
    print(result)


def print_operator_note_shape(print_list, information_level=0):
    levels = {
        1: ['index_name', 'input_shape', 'output_shape'],
        2: ['tp', 'cp', 'ep', 'index_name', 'input_shape', 'output_shape'],
    }
    attributes_list = levels.get(information_level, [])

    result = ", ".join(
        f"{attr}: {getattr(obj, attr)}" for obj in print_list for attr in attributes_list if hasattr(obj, attr))
    print(result)
