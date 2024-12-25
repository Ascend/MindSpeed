# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
class SwapPolicyConfig:
    def __init__(self):
        self.warmup_step = 2  # 多少步之后进入策略查找阶段
        self.stable_step = 10  # 多少步之后进入稳定阶段（包含warm up）

        # 初始阶段连续stable_N个steps变化小于op_diff_thresh后开始进行policy采集，开始采集后op序列变化超过op_diff_thresh后需要更新stable_op_sequence
        self.op_diff_thresh = 0.05
        self.tensor_size_thresh = 2**31 - 1
        self.enable_custom_record_stream = True

        self.save_policy = False
        self.save_profiler_data = False

        self.tensor_freq_thresh = 8  # 设置oomv2下为防止内存碎片对tensor频次进行的过滤

        self.print_level = 1  # 设置print级别 DEBUG=0, INFO=1, NONE=2
        self.print_rank = 0  # 设置打印信息的卡, -1打印所有卡

        self.rank = 0  # 获取当前rank
        self.output_root_path = "./swap_output"

        # 带宽设置
        self.bandwidth_arg = 1

        self.D2H_bandwidth = 64 / 2.5 * 1000 * self.bandwidth_arg
        self.H2D_bandwidth = 64 / 2.5 * 1000 * self.bandwidth_arg
        self.free_stage_delay = 4  # 设置为1的话则表示在完成后的紧接着下一个stage释放
        self.swap_in_free_stage_delay = 2

        # 内存降低目标相关设置
        # OOM情况下，降低到 显卡最大内存 - redundant_memory
        #            如果降低了还会出发被动swap，target_memory 每步降低 adjust_memory
        # 非OOM情况下，target_mode = True 则降低到 target_memory 内存目标
        #              target_mode = False则就降低 reduction_memory 大小的内存
        self.target_mode = False
        self.reduction_memory = 3 * 1024 * 1024 * 1024  # 手动设置目标内存
        self.target_memory = 40 * 1024 * 1024 * 1024  # 手动设置目标内存
        self.tensor_size_filter = 20 * 1024 * 1024  # 设置tensor size的过滤，小于20MB的不会被选为candidate

        self.redundant_memory = 2 * 1024 * 1024 * 1024
        self.size_coverage_weight = 2  # 以coverage weight为1，size比之的比例
        self.adjust_memory = 300 * 1024 * 1024  # 自动化调整 redundant_memory
        self.adjust_step_duration = 1  # 自动化调整duration time，将得到的step duration乘以这个数值，并与历史的取最小值
        self.adjust_size_coverage_weight = 0  # size_coverage_weight 每次递增这个数值

    def __str__(self):
        return str(self.__dict__)


swap_policy_config = SwapPolicyConfig()
