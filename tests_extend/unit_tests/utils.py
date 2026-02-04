import copy
import datetime
import time

import numpy as np
import torch

debug_switch = False


def print_log(data=None, level="INFO"):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), level, data))


def data_compare(
    npu_output, cpu_output, diff_thd=0.005, pct_thd=0.005, max_diff_hd=0.1, rtol=0.005, atol=0.000025, output_dtype=None
):
    max_error_idx = 10000000
    real_data = np.array(npu_output.to(torch.float32).cpu().flatten())
    data_compe = np.array(cpu_output.to(torch.float32).cpu().flatten())
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log('The npu_output is [],and it is same as bm_output, the result of data_compare is "PASS"')
        return "PASS", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "FAIL"
    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal." % (real_data.size, data_compe.size)
        )
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10])
        )

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    has_nan_inf = False
    if "nan" in str(real_data) or "inf" in str(real_data) or "nan" in str(data_compe) or "inf" in str(data_compe):
        has_nan_inf = True

    # 默认精度比对方式
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    if has_nan_inf:
        err_diff, err_idx = cal_nan_inf_diff(real_data, data_compe, output_dtype, diff_abs, diff_thd)
    else:
        diff_index = np.where(diff_abs > 0)
        rdiff = cal_relative_diff_np(
            real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32), diff_thd
        )
        err_diff = rdiff[rdiff > diff_thd]
        diff_idx_list = diff_index[0]
        err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    fulfill_percent = float(split_count - err_diff.size) / float(split_count) * 100.0

    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "PASS" if (fulfill_percent >= pct_thd) else "FAIL"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error > max_diff_hd:
            result = "FAIL"
            print("max_error >= max_diff_hd")
    print_log("---------------------------------------------------------------------------------------")
    print_log("DiffThd  \t PctThd   \t PctRlt   \t Result")
    print_log("---------------------------------------------------------------------------------------")
    print_log("%.4f     \t %.2f%%   \t %.6f%%   \t %s" % (diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log("Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd))
    if result == "FAIL":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])

    return result, fulfill_percent


def cal_nan_inf_diff(real_data, expect_data, output_dtype, diff_abs, diff_thd):
    err_diff = []
    err_idx = []

    if output_dtype == "fp32":
        inf_value = 3.4028e38
    elif output_dtype == "bf16":
        inf_value = 3.38e38
    else:
        inf_value = 65504

    real_data_copy = copy.deepcopy(real_data)
    expect_data_copy = copy.deepcopy(expect_data)
    inf_idx = np.where(np.isinf(real_data_copy))[0]
    pos_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] > 0)[0]
    neg_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] < 0)[0]
    real_data_copy[inf_idx[pos_inf_idx]] = inf_value
    real_data_copy[inf_idx[neg_inf_idx]] = -inf_value
    inf_idx = np.where(np.isinf(expect_data_copy))[0]
    pos_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] > 0)[0]
    neg_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] < 0)[0]
    expect_data_copy[inf_idx[pos_inf_idx]] = inf_value
    expect_data_copy[inf_idx[neg_inf_idx]] = -inf_value

    num_idx = np.where(
        ~np.isnan(real_data_copy)
        & ~np.isinf(real_data_copy)
        & ~np.isnan(expect_data_copy)
        & ~np.isinf(expect_data_copy)
    )
    nan_inf_idx = np.setdiff1d(np.arange(len(real_data_copy)), num_idx)

    rdiff = cal_relative_diff_np(
        real_data_copy[num_idx].astype(np.float32), expect_data_copy[num_idx].astype(np.float32), diff_thd
    )
    num_err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = num_idx[0]
    num_err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    real_data_str = list(map(lambda x: str(x), real_data[nan_inf_idx].tolist()))
    expect_data_str = list(map(lambda x: str(x), expect_data[nan_inf_idx].tolist()))
    temp_err_idx = np.where(np.array(real_data_str) != np.array(expect_data_str))[0]
    nan_inf_err_idx = nan_inf_idx[temp_err_idx]
    nan_inf_err_diff = diff_abs[nan_inf_err_idx]

    err_idx = num_err_idx.tolist() + nan_inf_err_idx.tolist()
    err_diff = num_err_diff.tolist() + nan_inf_err_diff.tolist()

    return np.array(err_diff), np.array(err_idx)


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def display_output(real_data, expect_data, start, end, diff_thd, expect_fp32_data=None, if_mix=False):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff(expect_data[j], real_data[j], diff_thd)

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate)
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate)
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate)
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate)
                )

    print_log("---------------------------------------------------------------------------------------")
    if expect_fp32_data is not None:
        print_log("Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff")
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log("---------------------------------------------------------------------------------------")
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log("Error Line-----------------------------------------------------------------------------")
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log("---------------------------------------------------------------------------------------")
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log("%08s \t %07s \t %07s \t %07s \t %07s" % (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log("Max-RE line:---------------------------------------------------------------------------")
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log("---------------------------------------------------------------------------------------")


def cal_relative_diff(real_data, expect_data, diff_thd, type_str="fp16"):
    if "nan" in str(expect_data) or "inf" in str(expect_data):
        if type_str.lower() == "fp16":
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def multi_compare(output_npu, output_cpu, output_golden, bin_name, level="l1"):
    rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(bin_name))
    rst_npu.print_result()
    rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(bin_name))
    rst_gpu.print_result()
    if level == "l1":
        str1, str2 = rst_npu.l1_check(rst_gpu)
    else:
        str1, str2 = rst_npu.l0_check(rst_gpu)
    if 'error' in str1:
        res = 'FAIL'
    else:
        res = 'PASS'
    print('[INFO] device_{} 精度结果为：{}'.format(bin_name, res))
    print('[INFO] device_{} 精度计算结束：{} '.format(bin_name, time.strftime('%H:%M:%S', time.localtime())))
    return res


class Result:
    def __init__(self, result_name, total_big_num=0, total_big_ratio=0, diff_big_max=0, diff_big_avg=0, diff_big_sum=0,
                 total_small_num=0, total_small_ratio=0, err_small_num=0, err_small_ratio=0,
                 diff_rmse=0, rst_eb=0, diff_eb=0,
                 num_total_nan=0, err_total_nan=0, num_total_inf=0, err_total_inf=0, num_total_ninf=0,
                 err_total_ninf=0):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf

    # 打印精度结果细节
    def print_result(self):
        print(f"正在打印结果：{self.result_name}")
        print(f" 大值总数：{self.total_big_num}")
        print(f" 大值占比：{self.total_big_ratio:.2%}")
        print(f" 大值最大误差：{self.diff_big_max:.8f}")
        print(f" 大值平均误差：{self.diff_big_avg:.8f}")
        print(f" 大值误差总和：{self.diff_big_sum:.2f}")
        print(f" 小值总数：{self.total_small_num}")
        print(f" 小值占比：{self.total_small_ratio:.2%}")
        print(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print(f" 均衡性偏差计数：{self.rst_eb}")
        print(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0) or \
            (self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0) or True:
            print(f" golden nan总数：{self.num_total_nan}")
            print(f" nan误差数：{self.err_total_nan}")
            print(f" golden inf总数：{self.num_total_inf}")
            print(f" inf误差数：{self.err_total_inf}")
            print(f" golden -inf总数：{self.num_total_ninf}")
            print(f" -inf误差数：{self.err_total_ninf}")

    # 解析精度报错细节
    def check_result_debug(self, benchmark):
        reason_str = ''
        if self.diff_big_max > benchmark.diff_big_max * 10:
            reason_str += ' diff_big_max error,'
        elif self.diff_big_max > benchmark.diff_big_max:
            reason_str += ' diff_big_max warning,'
        if self.diff_big_avg > benchmark.diff_big_avg * 2:
            reason_str += ' diff_big_avg error,'
        elif self.diff_big_avg > benchmark.diff_big_avg:
            reason_str += ' diff_big_avg warning,'
        if self.diff_big_sum > benchmark.diff_big_sum * 2:
            reason_str += ' diff_big_sum error,'
        elif self.diff_big_sum > benchmark.diff_big_sum:
            reason_str += ' diff_big_sum warning,'

        if self.err_small_num > benchmark.err_small_num * 2:
            reason_str += ' err_small_num error,'
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += ' err_small_num warning,'

        if self.diff_rmse > benchmark.diff_rmse * 2:
            reason_str += ' diff_rmse error,'
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += ' diff_rmse warning,'

        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += ' err_total_nan error,'
        elif self.err_total_nan > 0:
            reason_str += ' err_total_nan warning,'
        if self.err_total_inf > benchmark.err_total_inf or self.err_total_ninf > benchmark.err_total_ninf:
            reason_str += ' err_total_inf error,'
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += ' err_total_inf warning,'

        return reason_str

    # 与竞品对比精度结果，benchmark传入gpu竞品数据或基线版本数据，返回检查结果与检查不通过原因
    def l1_check(self, benchmark):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        small_num_total_atol = 100000
        if self.total_small_num >= small_num_total_atol:
            if self.diff_big_max > benchmark.diff_big_max * 5 or \
                self.diff_big_avg > benchmark.diff_big_avg * 1.5 or \
                self.diff_big_sum > benchmark.diff_big_sum * 1.5 or \
                self.err_small_num > benchmark.err_small_num * 1.5 or \
                self.diff_rmse > benchmark.diff_rmse * 1.5:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 5)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 1.5)
                print('err_small_num(大于0即error)', self.err_small_num - benchmark.err_small_num * 1.5)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 1.5)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        else:
            if self.diff_big_max > benchmark.diff_big_max * 5 or \
                self.diff_big_avg > benchmark.diff_big_avg * 1.5 or \
                self.diff_big_sum > benchmark.diff_big_sum * 1.5 or \
                self.diff_rmse > benchmark.diff_rmse * 1.5:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 5)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 1.5)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 1.5)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        print(self.result_name + 'compare result: ok')
        return 'ok', ''

    def l0_check(self, benchmark):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        small_num_total_atol = 100000
        if self.total_small_num >= small_num_total_atol:
            if self.diff_big_max > benchmark.diff_big_max * 10 or \
                self.diff_big_avg > benchmark.diff_big_avg * 2 or \
                self.diff_big_sum > benchmark.diff_big_sum * 2 or \
                self.err_small_num > benchmark.err_small_num * 2 or \
                self.diff_rmse > benchmark.diff_rmse * 2:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 10)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 2)
                print('err_small_num(大于0即error)', self.err_small_num - benchmark.err_small_num * 2)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 2)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        else:
            if self.diff_big_max > benchmark.diff_big_max * 10 or \
                self.diff_big_avg > benchmark.diff_big_avg * 2 or \
                self.diff_big_sum > benchmark.diff_big_sum * 2 or \
                self.diff_rmse > benchmark.diff_rmse * 2:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 10)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 2)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 2)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        print(self.result_name + 'compare result: ok')
        return 'ok', ''

    # 与竞品对比精度结果，benchmark传入gpu竞品数据或基线版本数据，返回检查结果与检查不通过原因
    def check_result_moe(self, benchmark):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        small_err_num_atol = 100
        if benchmark.err_small_num >= small_err_num_atol:
            if self.diff_big_max > benchmark.diff_big_max * 10 or \
                self.diff_big_avg > benchmark.diff_big_avg * 2 or \
                self.diff_big_sum > benchmark.diff_big_sum * 2 or \
                self.err_small_num > benchmark.err_small_num * 2 or \
                self.diff_rmse > benchmark.diff_rmse * 2:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 10)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 2)
                print('err_small_num(大于0即error)', self.err_small_num - benchmark.err_small_num * 2)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 2)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        else:
            if self.diff_big_max > benchmark.diff_big_max * 10 or \
                self.diff_big_avg > benchmark.diff_big_avg * 2 or \
                self.diff_big_sum > benchmark.diff_big_sum * 2 or \
                self.diff_rmse > benchmark.diff_rmse * 2:
                print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 10)
                print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 2)
                print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 2)
                print(self.result_name + 'compare result: error')
                reason_str = self.check_result_debug(benchmark)
                return 'error', reason_str
        print(self.result_name + 'compare result: ok')
        return 'ok', ''


def checkResult(value, golden, name):
    print(f"info：开始计算 {name} 精度。")
    # 数据预处理,将golden中，超过value表示范围的值赋值成inf/-inf
    print("处理golden中，超过value表示范围的值，赋值为inf/-inf")
    golden[golden > torch.finfo(value.dtype).max] = torch.inf
    golden[golden < torch.finfo(value.dtype).min] = -torch.inf
    if value.shape == golden.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(value, golden)):
            print(f"info：{name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            if value.numel() == 0:
                return Result(name)
        # inf nan对比
        mask_golden_is_nan = torch.isnan(golden)
        mask_value_is_nan = torch.isnan(value)
        num_total_nan = torch.sum(mask_golden_is_nan)
        err_total_nan = torch.sum(mask_golden_is_nan.logical_xor(mask_value_is_nan))

        mask_golden_is_inf = torch.isinf(golden) & (golden > 0)
        mask_value_is_inf = torch.isinf(value) & (value > 0)
        num_total_inf = torch.sum(mask_golden_is_inf)
        err_total_inf = torch.sum(mask_golden_is_inf.logical_xor(mask_value_is_inf))

        mask_golden_is_ninf = torch.isinf(golden) & (golden < 0)
        mask_value_is_ninf = torch.isinf(value) & (value < 0)
        num_total_ninf = torch.sum(mask_golden_is_ninf)
        err_total_ninf = torch.sum(mask_golden_is_ninf.logical_xor(mask_value_is_ninf))

        # 将所有inf处理为边界值（inf误差转换为数值误差）
        golden[golden == torch.inf] = torch.finfo(value.dtype).max
        golden[golden == -torch.inf] = torch.finfo(value.dtype).min
        value[value == torch.inf] = torch.finfo(value.dtype).max
        value[value == -torch.inf] = torch.finfo(value.dtype).min

        if debug_switch:
            print(f" inf/nan总数：{num_total_nan + num_total_inf + num_total_ninf}")
            print(f" inf/nan误差数：{err_total_nan + err_total_inf + err_total_ninf}")

        # 对inf/nan统一赋1，忽略影响
        golden[torch.isinf(golden)] = 1
        value[torch.isinf(value)] = 1
        golden[torch.isnan(golden)] = 1
        value[torch.isnan(value)] = 1

        if value.dtype == torch.float16:
            small_value = 0.001
            small_value_atol = 0.00001
        elif value.dtype == torch.bfloat16:
            small_value = 0.001
            small_value_atol = 0.00001
        elif value.dtype == torch.float32:
            small_value = 0.000001
            small_value_atol = 0.000000001
        else:
            small_value = 0.000025
            small_value_atol = 0.000000001
        # 大值对比
        total_big_num = torch.sum(golden.abs() >= small_value)
        total_big_ratio = total_big_num / golden.numel()

        # 对小值统一赋1，忽略影响
        value_big = value.clone()
        value_big[golden.abs() < small_value] = 1
        golden_big = golden.clone()
        golden_big[golden.abs() < small_value] = 1

        diff_big = torch.abs(value_big.sub(golden_big))
        diff_big_max = diff_big.max()
        diff_big_sum = diff_big.sum()
        diff_big_avg = diff_big_sum / total_big_num
        diff_big_ratio = diff_big / golden_big.abs()

        if debug_switch:
            print(f" 大值总数：{total_big_num}")
            print(f" 大值占比：{total_big_ratio:.2%}")
            print(f" 大值最大误差：{diff_big_max:.8f}")
            print(f" 大值平均误差：{diff_big_avg:.8f}")
            print(f" 大值误差总和：{diff_big_sum:.2f}")

        # 小值对比
        total_small_num = torch.sum(golden.abs() < small_value)
        total_small_ratio = total_small_num / golden.numel()

        # 对大值统一赋1，忽略影响
        value_small = value.clone()
        value_small[golden.abs() > small_value] = 1
        golden_small = golden.clone()
        golden_small[golden.abs() > small_value] = 1

        diff_small = torch.abs(value_small.sub(golden_small))
        err_small_num = torch.sum(diff_small > small_value_atol)
        err_small_ratio = err_small_num / total_small_num

        if debug_switch:
            print(f" 小值总数：{total_small_num}")
            print(f" 小值占比：{total_small_ratio:.2%}")
            print(f" 小值错误数：{err_small_num}，占比{err_small_ratio:.2%}")

        # 计算均方根误差（rmse）
        diff = torch.abs(value.sub(golden))
        diff_rmse = torch.sqrt(torch.mean(torch.square(diff)))
        if debug_switch:
            print(f" 误差均方根（RMSE）：{diff_rmse:.8f}")

        # 计算误差均衡性（eb）
        eb_bigger = torch.sum(value > golden)
        eb_smaller = torch.sum(value < golden)
        rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
        diff_eb = torch.sum(value.sub(golden))
        if debug_switch:
            print(f" 均衡性偏差计数：{rst_eb}")
            print(f" 均衡性diff总和：{diff_eb:.8f}")

        return Result(name, total_big_num, total_big_ratio, diff_big_max, diff_big_avg, diff_big_sum,
                      total_small_num, total_small_ratio, err_small_num, err_small_ratio, diff_rmse, rst_eb,
                      diff_eb,
                      num_total_nan, err_total_nan, num_total_inf, err_total_inf, num_total_ninf, err_total_ninf)
    else:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: 输入shape {value.shape}")
        print(f"debug: 真值shape  {golden.shape}")
