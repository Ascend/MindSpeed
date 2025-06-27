from copy import deepcopy
import math
import numpy as np
from mindspeed.auto_tuning.utils.logger import get_logger

logger = get_logger('operator_shape_analysis')


def get_default_shape_change(param):
    rows = param.split(';')
    arr = []
    for row in rows:
        nums = []
        for num in row.split(','):
            if num != '':
                nums.append(int(num))
        arr.append(nums)
    return arr


def is_not_zero(i):
    # 由于浮点数误差，使用0.01作为判断值是否为0的阈值
    return abs(i) > 0.01


def linear_regression_single(list_a, list_b):
    # 取对数将乘除转换为加减
    list_a = np.log2(list_a)
    list_b = np.log2(list_b)
    diff = 0
    index = []
    # 当输入结果值全部相同，默认为0
    if not np.sum(is_not_zero(list_b[:] - list_b[0])):
        base = round(2 ** list_b[0])
        return index, float(int(base) + float(diff))

    res = np.linalg.pinv(np.asmatrix(list_a.T @ list_a)) @ list_a.T @ list_b
    base = round(2 ** res[0, 3])

    # tp
    if (is_not_zero(abs(res[0, 0]))):
        diff = diff + 0.4
    # cp
    if (is_not_zero(abs(res[0, 1]))):
        diff = diff + 0.2
    # ep
    if (is_not_zero(abs(res[0, 2]))):
        diff = diff + 0.1

    count = 0
    # 异常处理：斜率与预设严重偏差
    if res[0, 0] > 0.01 or res[0, 1] < -1.5:
        count += 1
        list_a = np.delete(list_a, 0, axis=1)
        diff = diff - 0.4
    if res[0, 1] > 0.01 or res[0, 1] < -1.5:
        list_a = np.delete(list_a, 1 - count, axis=1)
        count += 1
        diff = diff - 0.2
    if res[0, 2] < -0.01 or res[0, 1] > 1.5:
        list_a = np.delete(list_a, 2 - count, axis=1)
        count += 1
        diff = diff - 0.1
    if count > 0:
        new_res = np.linalg.pinv(np.asmatrix(list_a.T @ list_a)) @ list_a.T @ list_b
        base = math.ceil(2 ** new_res[0, 3 - count])

    return index, float(int(base) + float(diff))


def linear_regression(list_a, shape, resultgroup):
    shape_elements = []
    indexs = []
    for i in range(element_length(shape)):
        index, shape_element = linear_regression_single(list_a, resultgroup[:, i])
        indexs.append(index)
        shape_elements.append(shape_element)
    return shape_elements


def separate_ep_tp_cp(results):
    a = []
    for result in results:
        a.append(deepcopy(result))
        a[-1].input_shape = get_default_shape_change(result.input_shape)
        a[-1].output_shape = get_default_shape_change(result.output_shape)

    list_a, inputgroup, outputgroup, \
        inputshape, outputshape = result_group(a)
    input_shape_element = linear_regression(list_a, a[0].input_shape, inputgroup)
    output_shape_element = linear_regression(list_a, a[0].output_shape, outputgroup)

    return new_shape(inputshape, input_shape_element), \
        new_shape(outputshape, output_shape_element)


def element_length(element):
    count = 0
    for input_shape in element:
        count = count + len(input_shape)
    return count


def is_uniform_length(two_d_list):
    if not two_d_list:  # 如果列表为空，可以认为长度一致
        return True
    first_length = len(two_d_list[0])
    return all(len(sublist) == first_length for sublist in two_d_list)


def reset_input_output_shape(results, list_a, inputgroup, outputgroup):
    # RemainedValue:检索input_shape、outputgroup的所有长度
    remainedvalue = []
    for i in inputgroup:
        remainedvalue.append(len(i))
    remainedvaluedict = {}
    for key in remainedvalue:
        if key in remainedvaluedict:
            remainedvaluedict[key] += 1
        else:
            remainedvaluedict[key] = 1
    max_count_inputgroup = max(
        [key for key, value in remainedvaluedict.items() if value == max(remainedvaluedict.values())])
    i = 0
    # 处理input_shape不符的算子，保留相同input_shape长度最多的值
    while i < len(inputgroup):
        if len(inputgroup[i]) != max_count_inputgroup:
            list_a.pop(i)
            inputgroup.pop(i)
            outputgroup.pop(i)
            results.pop(i)
            continue
        i = i + 1


def reset_bad_shape(results, list_a, inputgroup, outputgroup):
    reset_input_output_shape(results, list_a, inputgroup, outputgroup)
    reset_input_output_shape(results, list_a, outputgroup, inputgroup)
    return np.array(list_a), np.array(inputgroup), \
        np.array(outputgroup), results[0].input_shape, \
        results[0].output_shape


def result_group(results):
    inputgroup = []
    outputgroup = []
    list_a = []
    for result in results:
        resultlist = []
        # 在LinarRegression中需要对数组整体取log
        # 期待最小二乘时输入值为1（截距项），故而输入2
        list_a.append([result.tp, result.cp, result.ep, 2])
        for input_shape in result.input_shape:
            for i in input_shape:
                resultlist.append(i)
        inputgroup.append(resultlist)
        resultlist = []
        for output_shape in result.output_shape:
            for i in output_shape:
                resultlist.append(i)
        outputgroup.append(resultlist)

    try:
        if not is_uniform_length(inputgroup):
            raise ValueError(
                "Inconsistent input_shape. Highest number of the same input_shape reserved, others ignored.")
        return np.array(list_a), np.array(inputgroup), \
            np.array(outputgroup), results[0].input_shape, \
            results[0].output_shape
    except ValueError:
        # 异常处理1：输入异常处理-输入input_shape不一致
        logger.debug("Inconsistent input_shape. Highest number of the same input_shape reserved, others ignored.")
        logger.debug(f"Corresponding index_name: {results[0].index_name}")
        return reset_bad_shape(results, list_a, inputgroup, outputgroup)


def new_shape(shape, shape_element):
    count = 0
    shape_new = shape
    for x, _ in enumerate(shape_new):
        for y, _ in enumerate(shape_new[x]):
            shape_new[x][y] = shape_element[count]
            count = count + 1
    return shape_new
