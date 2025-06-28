import os
import sys

from collections import Counter
import torch

import matplotlib.pyplot as plt
import pandas as pd

from torch_npu.contrib import transfer_to_npu
from config import Config
from ops import Ops
from tests_extend.tools.coc_tools.utils import Utils


def plot_results(results, plot_file='plot.png'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
    ax_list = []
    ax_list.append(ax1)
    ax_list.append(ax2)
    ax_list.append(ax3)

    for op in set(results["operator"]):
        ax = ax_list.pop()
        for coc_index in set(results["index"]):
            shape_times = []
            coc = results["coc_type"][results["index"].index(coc_index)]
            for shape, operator, _, time, index in zip(results["shape"], results["operator"], results["coc_type"],
                                                         results["time"], results["index"]):
                if operator == op and index == coc_index:
                    shape_times.append((shape, time))

            shapes, times = zip(*shape_times)

            ax.plot(shapes, times, label=f"{coc}")
            ax.set_xlabel("Shape")
            ax.set_ylabel("Runtime in Seconds")
            ax.set_title(op)
            ax.legend(loc="upper right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(plot_file)


def save_to_excel(results, filename="matrix_multiplication_times.xlsx"):
    # 删除ascend turbo type的index字段
    del results["index"]
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(df)
    print(f"Results saved to {filename}")


def main():
    world_size = 8
    tp_size = 8

    Utils.initialize_distributed(world_size)
    tp_groups_list, tp_groups, all_list, all_group = Utils.get_tp_list_and_groups(tp_size, world_size)

    example = Ops(tp_size=tp_size, tp_groups=tp_groups, tp_groups_list=tp_groups_list)

    results = example.test_shapes(Config.ops, Config.comm_overlap_type, Config.shape_list, Config.data_type)

    cal_shape_list = []
    for (_, shape) in Config.model_ops_shape:
        cal_shape_list.append(shape)

    cal_results = example.test_shapes(Config.ops, Config.comm_overlap_type, cal_shape_list, Config.data_type)

    return results, cal_results


def cal_and_sort(results):
    cal_list = Config.model_ops_shape
    model_result = {}
    for (op, shape) in cal_list:
        shape_str = (f"{shape[0]}x{shape[1]}x{shape[2]}")
        cur = {}

        for s, op_, coc_type, time in zip(results["shape"], results["operator"], results["coc_type"],
                                                    results["time"]):
            if op_ == op and s == shape_str:
                cur[coc_type] = time

        A, B = Counter(model_result), Counter(cur)
        model_result = dict(A + B)
    sorted_results = sorted(model_result.items(), key=lambda kv: (kv[1], kv[0]))
    df = pd.DataFrame(sorted_results)
    names = ["comm_overlap_type", "execution_time"]
    df.columns = names
    print(df)


if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        results, cal_results = main()

        plot_results(results, plot_file="tests_extend/tools/coc_tools/plot.png")
        save_to_excel(results, filename="tests_extend/tools/coc_tools/matrix_multiplication_times.xlsx")
        cal_and_sort(cal_results)
    else:
        results = main()
