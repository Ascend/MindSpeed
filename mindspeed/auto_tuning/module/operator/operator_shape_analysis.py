from mindspeed.auto_tuning.utils.logger import get_logger

logger = get_logger('operator_shape_analysis')


class DataEp:
    def __init__(self):
        self.tp = 0
        self.cp = 0
        self.ep = 0
        self.input_shape = ""
        self.output_shape = ""


def separate_ep(results):
    diff_idx_input = []
    diff_idx_output = []
    index_visit = [False] * len(results)
    flag = 0
    result = []
    for i, _ in enumerate(results):
        input_list = {}
        output_list = {}
        if index_visit[i]:
            continue
        index_visit[i] = True
        result1 = results[i]
        tp1 = str(result1.tp)
        cp1 = str(result1.cp)
        ep1 = str(result1.ep)
        seq_length1 = str(result1.seq_length)
        input_list[ep1] = get_default_shape_change(result1.input_shape)
        output_list[ep1] = get_default_shape_change(result1.output_shape)

        for j in range(i + 1, len(results)):
            if index_visit[j]:
                continue
            result2 = results[j]
            cp2 = str(result2.cp)
            tp2 = str(result2.tp)
            ep2 = str(result2.ep)
            seq_length2 = str(result2.seq_length)
            if tp1 != tp2 or cp1 != cp2 or seq_length1 != seq_length2:
                continue
            index_visit[j] = True
            input_list[ep2] = get_default_shape_change(result2.input_shape)
            output_list[ep2] = get_default_shape_change(result2.output_shape)
        # calculate linear relationship
        ep_arr = list(input_list.keys())
        # The first occurrence of ep is recorded, other ep operator shapes directly modify the relevant dimension and
        # insert into the dictionary.
        if flag == 0:
            diff_idx_input = [0] * count_num(input_list.get(str(ep1)))
            diff_idx_output = [0] * count_num(output_list.get(str(ep1)))
            input_cal_tmp, diff_idx_input = analyze_shape_arr_new(input_list, ep_arr, diff_idx_input, 2)
            output_cal_tmp, diff_idx_output = analyze_shape_arr_new(output_list, ep_arr, diff_idx_output, 2)
            if len(input_list) != 1:
                flag = 1
        else:
            input_cal_tmp = modify_by_index(input_list, diff_idx_input, ep_arr, mode=1)
            output_cal_tmp = modify_by_index(output_list, diff_idx_output, ep_arr, mode=1)
        tmp = DataEp()
        tmp.tp = tp1
        tmp.cp = cp1
        tmp.ep = ep1
        tmp.seq_length = seq_length1
        tmp.input_shape = input_cal_tmp
        tmp.output_shape = output_cal_tmp
        result.append(tmp)
    return result


def separate_cp_tp(results):
    input_shape_dic = {}
    output_shape_dic = {}
    index_visit = [False] * len(results)
    diff_idx_input = []
    diff_idx_output = []
    flag = 0
    for i, _ in enumerate(results):
        input_list = {}
        output_list = {}
        if index_visit[i]:
            continue
        index_visit[i] = True
        result1 = results[i]
        cp1 = str(result1.cp)
        tp1 = str(result1.tp)
        seq_length1 = str(result1.seq_length)
        input_list[tp1] = result1.input_shape
        output_list[tp1] = result1.output_shape
        for j in range(i + 1, len(results)):
            if index_visit[j]:
                continue
            result2 = results[j]
            cp2 = str(result2.cp)
            tp2 = str(result2.tp)
            seq_length2 = str(result2.seq_length)
            if cp1 != cp2 or seq_length1 != seq_length2:
                continue
            index_visit[j] = True
            input_list[tp2] = result2.input_shape
            output_list[tp2] = result2.output_shape
        # calculate linear relationship
        tp_arr = list(input_list.keys())
        if set(input_list.keys()) == {'8', '4'}:
            for index_i, sublist in enumerate(input_list.get('4')):
                for j, value in enumerate(sublist):
                    check_value = isinstance(value, float) and '.1' in str(value)
                    if (check_value and index_i < len(input_list.get('8'))
                            and j < len(input_list.get('4')[index_i])):
                        input_list.get('8')[index_i][j] = value
        # The first occurrence of cp is recorded, other cp operator shapes directly modify the relevant dimension
        if flag == 0:
            arr_in = input_list.get(str(tp1))
            arr_out = output_list.get(str(tp1))
            diff_idx_input = [0] * count_num(arr_in)
            diff_idx_output = [0] * count_num(arr_out)
            input_cal_tmp, diff_idx_input = analyze_shape_arr_new(input_list, tp_arr, diff_idx_input, 0)
            output_cal_tmp, diff_idx_output = analyze_shape_arr_new(output_list, tp_arr, diff_idx_output, 0)
            if len(input_list) != 1:
                flag = 1
        else:
            input_cal_tmp = modify_by_index(input_list, diff_idx_input, tp_arr, mode=2)
            output_cal_tmp = modify_by_index(output_list, diff_idx_output, tp_arr, mode=2)
        input_shape_dic[cp1] = input_cal_tmp
        output_shape_dic[cp1] = output_cal_tmp
    if set(input_shape_dic.keys()) == {'4', '2'}:
        for i, sublist in enumerate(input_shape_dic.get('2')):
            for j, value in enumerate(sublist):
                check_value = isinstance(value, float) and '.4' in str(value)
                if (check_value and
                        i < len(input_shape_dic.get('4')) and j < len(input_shape_dic.get('4')[i])):
                    input_shape_dic.get('4')[i][j] = value
    # calculate linear relationship
    cp_arr = list(input_shape_dic.keys())
    input_cal_arr, diff_idx_input = analyze_shape_arr_new(input_shape_dic, cp_arr, diff_idx_input, 1)
    output_cal_arr, diff_idx_output = analyze_shape_arr_new(output_shape_dic, cp_arr, diff_idx_output, 1)

    return input_cal_arr, output_cal_arr


def analyze_shape_arr_new(input_shape_list, tp_arr, diff, mode=0):
    # Data cleaning, removing some invalid data.
    input_shape_list, tp_arr = normal_list(input_shape_list, tp_arr)

    # Initialize the result array, initializing values for each position in the shape, defaulting value means unchanged.
    result_arr = input_shape_list.get(str(tp_arr[0]))

    # Compare the differences in shape between different TPs, and find the index of the differing columns
    diff_idx, diff_arr = analyze_shape_list(input_shape_list, str(tp_arr[0]))
    w_arr = []
    num = count_num(result_arr)
    if len(diff_idx) != 0 and len(diff) < num:
        diff = [0] * num
    for i in diff_idx:
        if mode == 0:
            diff[i] |= 1
        elif mode == 1:
            diff[i] += 1
        elif mode == 2:
            diff[i] = 1
    """
        tp cp ep
        1  1  1
        Only cut by TP with a suffix of 0.4, only CP is 0.2, only EP is 0.1.
        CP + EP binary corresponds to 0.3.
    """
    for index, _ in enumerate(diff_idx):
        # Calculate and record the pattern of changes based on the different data, with the default tp * shape_x
        i = diff_idx[index]
        if mode == 2:
            w = cal_shape_change_with_ep(diff_arr[index], tp_arr)
        else:
            w = cal_shape_change_with_tp_cp(diff_arr[index], tp_arr)
        flag = 0
        dis = float(float(w) - int(w))
        w = modify_special(w)
        if abs(dis - 0.1) < 0.001:
            flag = 1
        if diff[i] == 1:
            if mode == 0:
                if flag == 0:
                    # Only cut by TP 0.4
                    w_arr.append(float(w) + 0.4)
                elif flag == 1:
                    # tp + ep 0.5
                    w_arr.append(float(int(w)) + 0.5)
            elif mode == 1:
                if flag == 0:
                    # Only cut by CP 0.2
                    w_arr.append(float(w) + 0.2)
                elif flag == 1:
                    # cp + ep 0.3
                    w_arr.append(float(int(w)) + 0.3)
            elif mode == 2:
                # ep with suffix 0.1
                w_arr.append(float(w) + 0.1)
        elif diff[i] == 2:
            if flag == 0:
                # tp + cp 0.6
                w_arr.append(float(int(w)) + 0.6)
            elif flag == 1:
                # tp + cp + ep 0.7
                w_arr.append(float(int(w)) + 0.7)
        else:
            logger.warning("error")
    result_arr = convert_w_to_result_arr(result_arr, diff_idx, w_arr)
    return result_arr, diff


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


def analyze_shape_list(input_shape_list, row1_value):
    diff_index = []  # Save different column indices
    diff_arr = []  # Save different data
    # Compare the sublist within each list.
    column_index = 0

    for i in range(len(input_shape_list[row1_value])):
        for index_n in range(len(input_shape_list[row1_value][i])):
            tmp_list = []
            tmp_list_float = []
            for value in input_shape_list.values():
                tmp_list.append(int(value[i][index_n]))
                tmp_list_float.append(value[i][index_n])
            if len(set(tmp_list)) != 1:
                diff_arr.append(tmp_list_float)
                diff_index.append(column_index)
            column_index += 1

    return diff_index, diff_arr


def cal_shape_change_with_tp_cp(y_arr, x_arr):
    w_arr = []
    size = len(x_arr)
    h = float(y_arr[0] - int(y_arr[0]))
    for index in range(0, size):
        if abs(h) < 0.001:
            h = float(y_arr[index] - int(y_arr[index]))
        w_arr.append(int(y_arr[index]) * int(x_arr[index]))

    return w_arr[0] + h


def cal_shape_change_with_ep(y_arr, x_arr):
    w_arr = []
    size = len(x_arr)
    h = float(y_arr[0] - int(y_arr[0]))
    for index in range(0, size):
        if abs(h) < 0.001:
            h = float(y_arr[index] - int(y_arr[index]))
        w_arr.append(int(y_arr[index]) / float(x_arr[index]))

    return w_arr[0] + h


def convert_w_to_result_arr(result_arr, index_arr, w_arr):
    result_list = []
    column_index = 0
    index_index = 0
    for inner_arr in result_arr:
        result = []
        for item in inner_arr:
            if index_index < len(index_arr) and column_index == index_arr[index_index]:
                result.append(float(w_arr[index_index]))
                index_index = index_index + 1
            else:
                result.append(float(item))
            column_index = column_index + 1
        result_list.append(result)
        if len(inner_arr) == 0:
            column_index = column_index + 1
    return result_list


def check_array_format(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for i, _ in enumerate(arr1):
        if isinstance(arr1[i], list) and isinstance(arr2[i], list):
            if not check_array_format(arr1[i], arr2[i]):
                return False
    return True


def normal_list(input_shape_list, tp_arr):
    new_input_shape_list = {}
    new_tp_arr = []
    if len(input_shape_list) > 0 and len(tp_arr) > 0:
        new_input_shape_list[str(tp_arr[0])] = input_shape_list[str(tp_arr[0])]
        new_tp_arr.append(tp_arr[0])
        for index in range(1, len(tp_arr)):
            if check_array_format(input_shape_list[str(tp_arr[0])], input_shape_list[str(tp_arr[index])]):
                new_input_shape_list[str(tp_arr[index])] = input_shape_list[str(tp_arr[index])]
                new_tp_arr.append(tp_arr[index])
    else:
        logger.warning(f'Incorrect input_shape_list or tp_arr: {input_shape_list}, {tp_arr}')

    return new_input_shape_list, new_tp_arr


def modify_special(w):
    result = int(w)
    if result == 9016:
        result = 9024
    elif result == 1127:
        result = 1128

    return result


def count_num(arr):
    cnt = 0
    for i in arr:
        for _ in i:
            cnt += 1
    return cnt


def modify_by_index(shape_list, index_diff, tp_arr, mode=0):
    # Data cleaning, to remove invalid data elements, such as data that doesn't match the shape
    input_shape_list, tp_arr = normal_list(shape_list, tp_arr)

    input_list = shape_list[str(tp_arr[0])]
    result_list = []
    i_diff = 0
    column_index = 0
    for arr in input_list:
        result = []
        for item in arr:
            ans = 0.0
            if column_index < len(index_diff) and index_diff[column_index] == 1:
                # 修改
                if mode == 1:
                    ans = float(int(item) / float(tp_arr[0])) + 0.1
                elif mode == 2:
                    ans = float(int(item) * float(tp_arr[0])) + 0.4
                i_diff += 1
            else:
                ans = float(item)
            result.append(float(ans))
            column_index += 1
        result_list.append(result)

    return result_list
