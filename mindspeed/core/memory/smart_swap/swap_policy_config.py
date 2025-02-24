# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
class SwapPolicyConfig:
    def __init__(self):
        # utils
        self.rank = 0  # 获取当前rank

        self.save_policy = False
        self.save_profiler_data = False

        self.print_level = 1 # 设置print级别 DEBUG=0, INFO=1, NONE=2
        self.print_rank = 0 # 设置打印信息的卡, -1打印所有卡
        self.output_root_path = "./swap_output"

        # 执行
        self.warmup_step = 2 # 多少步之后进入SEARCHING_POLICY_STAGE
        self.stable_step = 10 # 多少步之后进入STABLE_STAGE

        self.op_diff_thresh = 0.05
        self.tensor_size_thresh = 2**31 - 1

        self.enable_custom_record_stream = True
        self.free_stage_delay = 4  # 表示将swap out任务的内存延后N个stage强制释放
        self.swap_in_free_stage_delay = 2 # 表示将swap in任务的内存延后N个stage强制释放

        # 带宽设置
        self.D2H_bandwidth = 64 / 2.5 * 1000
        self.H2D_bandwidth = 64 / 2.5 * 1000

        # 内存目标设置
        # OOM场景: 降低到 device最大内存 - redundant_memory 内存目标
        #          如果后续迭代中仍触发OOM swap, target_memory 将每步减少 adjust_memory 大小
        # 非OOM场景: target_mode = True 指降低至 target_memory 内存目标
        #              target_mode = False 指仅降低 reduction_memory 内存目标
        self.target_mode = False
        self.reduction_memory = 3 * 1024 * 1024 * 1024  # 手动设置目标内存
        self.target_memory = 40 * 1024 * 1024 * 1024  # 手动设置目标内存
        self.tensor_size_filter = 20 * 1024 * 1024  # 设置tensor size的过滤, 小于20MB的不会被选为candidate

        self.redundant_memory = 2 * 1024 * 1024 * 1024
        self.size_coverage_weight = 2  # 以coverage weight为1, size比之的比例
        self.adjust_memory = 300 * 1024 * 1024  # 自动化调整 redundant_memory
        self.adjust_step_duration = 1  # 自动化调整duration time, 将得到的step duration乘以这个数值, 并与历史的取最小值
        self.adjust_size_coverage_weight = 0  # size_coverage_weight 每次递增这个数值

    def __str__(self):
        return str(self.__dict__)


swap_policy_config = SwapPolicyConfig()
