import ast
import os
from enum import Enum
import torch
import torch_npu
import torch.nn.functional as F


def column_forward(self, input_, column_parallel_function=None, check_fcn=None):
    if check_fcn is not None:
        check_fcn()
    bias = self.bias if not self.skip_bias_add else None
    input_parallel = input_
    use_weight = self.weight
    if hasattr(self, "norm") and self.norm:
        use_weight = F.normalize(self.weight)
    output_parallel = column_parallel_function.apply(
        input_parallel,
        use_weight,
        bias
    )
    output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def row_forward(self, input_, row_parallel_function=None, check_fcn=None):
    if check_fcn is not None:
        check_fcn()
    input_parallel = input_
    output_parallel = row_parallel_function.apply(
        input_parallel,
        self.weight,
        None
    )
    output = output_parallel
    if not self.skip_bias_add:
        output = output + self.bias if self.bias is not None else output
        output_bias = None
    else:
        output_bias = self.bias
    return output, output_bias


class ModuleType(Enum):
    ORIGINAL_ALL_REDUCE = 0
    ORIGINAL_SEQ_PARALLEL = 1
    REWRITE_ALL_REDUCE = 2
    REWRITE_SEQ_PARALLEL = 3
    COC_FOR_ALL_REDUCE = 4
    COC_FOR_SEQ_PARALLEL = 5


class MinCommConfig:
    def __init__(self, coc_mode, coc_parallel_num, coc_fused_kernel):
        # basic settings acquired from environmental variables
        # default module_type is ModuleType.ORIGINAL_SEQ_PARALLEL
        self.module_type: ModuleType = ModuleType.ORIGINAL_SEQ_PARALLEL
        self.coc_mode = coc_mode
        self.parallel_num = coc_parallel_num
        self.coc_fused_kernel = coc_fused_kernel

        # configurations registered from framework
        self.ColumnParallelLinear = None
        self.RowParallelLinear = None
        self.column_parallel_forward = None
        self.row_parallel_forward = None
        self.tp_group_fcn = None
        self.tp_world_size_fcn = None
        self.tp_rank_fcn = None
        self.all_reduce = None
        self.reduce_scatter_along_first_dim = None
        self.gather_along_first_dim = None
        self.prefix = None
        self.check_fcn = None
        self.tp_enabled = True
        self.sequence_parallel_enabled = True

        # configurations manually set by users in user_config.py
        self.k_min = 1024
        self.k_max = 4096
        self.all_gather_recomputation_enabled = False
        self.print_tensor_value_enabled = False
        self.matmul_soc_friendly_enabled = True
        self.customized_coc_dict = {}
        self.enable_coc_in_column_backward = True

    def print_settings(self):
        if self.coc_fused_kernel:
            enable_coc_in_column_backward = True if self.enable_coc_in_column_backward else False
        else:
            enable_coc_in_column_backward = False
        if self.coc_fused_kernel:
            settings_dict = {
                "is coc turned on": True,
                "use script or use fused kernel": "fused kernel",
                "is sequence parallel enabled": self.sequence_parallel_enabled,
                "is coc enabled in column backward": enable_coc_in_column_backward
            }
        elif "ORIGINAL" in self.module_type.name:
            settings_dict = {
                "is coc turned on": False
            }
        else:
            settings_dict = {
                "is coc turned on": True,
                "use script or use fused kernel": "script",
                "coc mode": self.coc_mode,
                "parallel num": self.parallel_num,
                "module type": self.module_type.name,
                "is sequence parallel enabled": self.sequence_parallel_enabled,
                "if get aligned mm inputs": self.matmul_soc_friendly_enabled
            }
        if torch.npu.current_device() == 0:
            print("\n-----------------------------COC Settings: ------------------------------------")
            for key, value in settings_dict.items():
                print(f"{key}: {value}")
            print("-------------------------------------------------------------------------------\n")

    @property
    def tp_rank(self):
        return self.tp_rank_fcn()

    @property
    def tp_group(self):
        return self.tp_group_fcn()

    @property
    def tp_world_size(self):
        return self.tp_world_size_fcn()

    def register_tp_get_functions(self, tp_group_fcn, tp_world_size_fcn, tp_rank_fcn):
        self.tp_group_fcn = tp_group_fcn
        self.tp_world_size_fcn = tp_world_size_fcn
        self.tp_rank_fcn = tp_rank_fcn

    def register_class(self, column_parallel_linear, row_parallel_linear):
        self.ColumnParallelLinear = column_parallel_linear
        self.RowParallelLinear = row_parallel_linear

    def register_mappings(self, _all_reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim):
        self.all_reduce = _all_reduce
        self.reduce_scatter_along_first_dim = _reduce_scatter_along_first_dim
        self.gather_along_first_dim = _gather_along_first_dim

    def replace_forward_functions_by_autograd_class(self, column_autograd_class, row_autograd_class):
        def column_parallel_forward(x, y):
            return column_forward(x, y, column_parallel_function=column_autograd_class,
                                  check_fcn=self.check_fcn)

        def row_parallel_forward(x, y):
            return row_forward(x, y, row_parallel_function=row_autograd_class, check_fcn=self.check_fcn)

        self.column_parallel_forward = column_parallel_forward
        self.row_parallel_forward = row_parallel_forward
        self.ColumnParallelLinear.forward = self.column_parallel_forward
        self.RowParallelLinear.forward = self.row_parallel_forward

    def register_sequence_parallel_switch(self, sequence_parallel_enabled):
        self.sequence_parallel_enabled = sequence_parallel_enabled

    def register_check_fcn(self, check_fcn):
        self.check_fcn = check_fcn

    def register_customized_coc(self, customized_coc):
        if len(customized_coc) == 0:
            return
        for coc_shape_yaml_str in customized_coc.keys():
            key_list = ast.literal_eval(coc_shape_yaml_str)
            coc_shape_key_str = str(key_list)
            self.customized_coc_dict.update({coc_shape_key_str: customized_coc[coc_shape_yaml_str]})
        print("self.customized_coc_dict: ", self.customized_coc_dict)

    def register_matmul_soc_friendly_setting(self, matmul_soc_friendly, k_min, k_max):
        self.matmul_soc_friendly_enabled = matmul_soc_friendly
        self.k_min = k_min
        self.k_max = k_max

    def register_all_gather_recomputation_switch(self, all_gather_recomputation_enabled):
        self.all_gather_recomputation_enabled = all_gather_recomputation_enabled

    def register_print_tensor_value_switch(self, print_tensor_value_enabled):
        self.print_tensor_value_enabled = print_tensor_value_enabled

    def register_column_backward_coc_switch(self, enable_coc_in_column_backward):
        self.enable_coc_in_column_backward = enable_coc_in_column_backward

    def acquire_module_type(self, tp_size):
        sequence_parallel_types = [ModuleType.ORIGINAL_SEQ_PARALLEL,
                                   ModuleType.REWRITE_SEQ_PARALLEL,
                                   ModuleType.COC_FOR_SEQ_PARALLEL]
        all_reduce_types = [ModuleType.ORIGINAL_ALL_REDUCE,
                            ModuleType.REWRITE_ALL_REDUCE,
                            ModuleType.COC_FOR_ALL_REDUCE]

        if self.parallel_num not in [1, 2, 4, 8]:
            raise RuntimeError("COC_PARALLEL_NUM must be either 1, 2, 4 or 8. Current value not supported")
        if self.coc_mode not in [-1, 0, 1, 2]:
            raise RuntimeError("COC_MODE must be either 0, 1, or 2. Current value not supported")

        if self.coc_mode == -1:
            self.coc_mode = 0 if self.parallel_num == 1 else 2

        if tp_size == 1:
            self.coc_mode = 0
            self.parallel_num = 1

        if self.sequence_parallel_enabled:
            self.module_type = sequence_parallel_types[self.coc_mode]
        else:
            self.module_type = all_reduce_types[self.coc_mode]

        if "COC" in self.module_type.name:
            self.prefix = f"module_{self.module_type.name}_parallel_num_{self.parallel_num}"
        else:
            self.prefix = f"module_{self.module_type.name}"

        self.print_settings()


# 0 = original, 1 = rewrite, 2 = coc default
coc_mode_from_env = int(os.getenv("COC_MODE", -1))
coc_parallel_num_from_env = int(os.getenv("COC_PARALLEL_NUM", 1))
coc_fused_kernel_from_env = int(os.getenv("USE_COC_FUSED_KERNEL", 0))

min_comm_config = MinCommConfig(coc_mode=coc_mode_from_env, coc_parallel_num=coc_parallel_num_from_env,
                                coc_fused_kernel=coc_fused_kernel_from_env)
