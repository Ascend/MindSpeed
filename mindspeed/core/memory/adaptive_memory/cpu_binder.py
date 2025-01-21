import os
import psutil
from megatron.training import print_rank_0


def _get_pcie_info(devices, keyword="PCIeBusInfo"):
    device_pcie_tbl = dict()
    for device in devices:
        pcie_info = os.popen(f"npu-smi info -t board -i {device}").read().strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                device_pcie_tbl[device] = line[len(keyword) + 1:]
                break

    return device_pcie_tbl


def _get_numa_info(pcie_tbl, keyword="NUMAnode"):
    device_numa_tbl = dict()  # key is device id, value is numa id
    numa_devices_tbl = dict()  # key is numa id, value is device id list

    for device, pcie_no in pcie_tbl.items():
        numa_info = os.popen(f"lspci -s {pcie_no} -vvv").read().strip().split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_numa_tbl[device] = numa_id

                devices = numa_devices_tbl.get(numa_id, None)
                if devices is None:
                    numa_devices_tbl[numa_id] = list()

                numa_devices_tbl[numa_id].append(device)
                break

    return device_numa_tbl, numa_devices_tbl


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = os.popen(f"lscpu").read().strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


# 可以用export CPU_BINDING_NUM设置每个进程绑的核数;如果不设置CPU_BINDING_NUM,
# 会根据ratio(numa利用率)进行计算,如果有64个核，0.5表示用一半，用32个核, 平分给亲和在这个numa上的npu
def bind_cpus(world_size, rank_id, device_id, ratio=0.5):
    devices = [_ for _ in range(device_id, device_id + world_size)]
    # 获取npu和pcie的对应关系
    device_pcie_tbl = _get_pcie_info(devices)
    # 根据pcie信息获取npu和numa的对应关系
    device_numa_tbl, numa_devices_tbl = _get_numa_info(device_pcie_tbl)
    # 获取使用的numa对应的cpu核分配信息
    cpu_idx_tbl = _get_cpu_info(list(numa_devices_tbl.keys()))

    # 当前rank的npu id
    cur_device = rank_id + device_id
    # 获取npu对应的numa id
    numa_id = device_numa_tbl[cur_device]

    # 获取共享该numa的npu信息
    shard_devices = numa_devices_tbl[numa_id]
    # 按照npu id进行排序
    shard_devices.sort()

    # 获取该numa上所有的cpu id信息
    all_cpus = cpu_idx_tbl[numa_id]

    cpu_nums = len(all_cpus)
    # 计算给该共享numa的npu分配的核的个数
    CPU_BINDING_NUM = os.environ.get("CPU_BINDING_NUM", None)
    if CPU_BINDING_NUM is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(CPU_BINDING_NUM)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            raise Exception(
                f"Cpu num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!")

    # 获取该npu的下标信息
    idx = shard_devices.index(cur_device)
    # 给该npu分配要绑定的cpu id
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    print_rank_0("Bind cpu successful!!!")
