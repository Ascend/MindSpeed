import ast
from enum import Enum
import torch


def print_on_device0(msg, isprint):
    if torch.npu.current_device() == 0 and isprint:
        print(msg)


class ModuleType(Enum):
    ORIGINAL_ALL_REDUCE = 0
    ORIGINAL_SEQ_PARALLEL = 1
    REWRITE_ALL_REDUCE = 2      # debug
    REWRITE_SEQ_PARALLEL = 3    # debug
    COC_FOR_ALL_REDUCE = 4
    COC_FOR_SEQ_PARALLEL = 5


class MinCommConfig:
    def __init__(self):
        # basic settings acquired from environmental variables
        # default module_type is ModuleType.ORIGINAL_SEQ_PARALLEL

        self.module_type: ModuleType = ModuleType.ORIGINAL_SEQ_PARALLEL
        self.coc_mode = None  # coc_mode degee (-1, 0, 1, 2)  -1:auto 0:close 1:debug(same like close) 2:open
        self.parallel_num = None  # Parallelism degree (-1, 1, 2, 4, 8)
        self.coc_fused_kernel = None

        # configurations registered from framework
        self.ColumnParallelLinear = None
        self.RowParallelLinear = None
        self.column_parallel_forward = None
        self.row_parallel_forward = None
        self.tp_group_fcn = None
        self.tp_world_size_fcn = None
        self.tp_rank_fcn = None
        self.tp_gather_func = None
        self.all_reduce = None
        self.reduce_scatter_along_first_dim = None
        self.gather_along_first_dim = None
        self.prefix = None
        self.tp_enabled = True
        self.sequence_parallel_enabled = True
        self.gradient_accumulation_fusion = False

        # configurations manually set by users in user_config.py
        self.k_min = 1024
        self.k_max = 4096
        self.all_gather_recomputation_enabled = False
        self.print_tensor_value_enabled = False
        self.matmul_soc_friendly_enabled = True
        self.customized_coc_dict = {}
        self.enable_coc_in_column_backward = True

        self.column_parallel_function = None
        self.row_parallel_function = None

        self.coc_print_enabled = True

    def print_settings(self):
        """Print current configuration settings."""
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
        """Current tensor parallel rank."""
        return self.tp_rank_fcn()

    @property
    def tp_group(self):
        """Tensor parallel group."""
        return self.tp_group_fcn()

    @property
    def tp_world_size(self):
        """Tensor parallel world size."""
        return self.tp_world_size_fcn()

    def register_tp_get_functions(self, tp_group_fcn, tp_world_size_fcn, tp_rank_fcn, tp_gather_func):
        """Register tensor parallel helper functions."""
        self.tp_group_fcn = tp_group_fcn
        self.tp_world_size_fcn = tp_world_size_fcn
        self.tp_rank_fcn = tp_rank_fcn
        self.tp_gather_func = tp_gather_func

    def register_coc_get_param(self, coc_mode, coc_parallel_num, coc_fused_kernel):
        """Register CoC optimization parameters."""
        self.coc_mode = coc_mode
        self.parallel_num = coc_parallel_num
        self.coc_fused_kernel = coc_fused_kernel

    def register_mappings(self, _all_reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim):
        """Register mapping helper functions."""
        self.all_reduce = _all_reduce
        self.reduce_scatter_along_first_dim = _reduce_scatter_along_first_dim
        self.gather_along_first_dim = _gather_along_first_dim

    def register_sequence_parallel_switch(self, sequence_parallel_enabled):
        """Enable/disable sequence parallelism."""
        self.sequence_parallel_enabled = sequence_parallel_enabled

    def register_gradient_accumulation_fusion(self, gradient_accumulation_fusion):
        """Enable/disable gradient accumulation fusion."""
        self.sequence_parallel_enabled = gradient_accumulation_fusion

    def register_customized_coc(self, customized_coc):
        """Register custom CoC optimization settings."""
        if len(customized_coc) == 0:
            return
        for coc_shape_yaml_str in customized_coc.keys():
            key_list = ast.literal_eval(coc_shape_yaml_str)
            coc_shape_key_str = str(key_list)
            self.customized_coc_dict.update({coc_shape_key_str: customized_coc[coc_shape_yaml_str]})
        print("self.customized_coc_dict: ", self.customized_coc_dict)

    def register_matmul_soc_friendly_setting(self, matmul_soc_friendly, k_min, k_max):
        """Configure memory-friendly matmul settings."""
        self.matmul_soc_friendly_enabled = matmul_soc_friendly
        self.k_min = k_min
        self.k_max = k_max

    def register_all_gather_recomputation_switch(self, all_gather_recomputation_enabled):
        """Enable/disable all-gather recomputation."""
        self.all_gather_recomputation_enabled = all_gather_recomputation_enabled

    def register_print_tensor_value_switch(self, print_tensor_value_enabled):
        """Enable/disable all-gather recomputation."""
        self.print_tensor_value_enabled = print_tensor_value_enabled

    def register_column_backward_coc_switch(self, enable_coc_in_column_backward):
        """Enable/disable CoC optimizations in column backward pass."""
        self.enable_coc_in_column_backward = enable_coc_in_column_backward

    def acquire_module_type(self, tp_size):
        """Determine and set the module type based on configuration."""
        sequence_parallel_types = [ModuleType.ORIGINAL_SEQ_PARALLEL,
                                   ModuleType.REWRITE_SEQ_PARALLEL,
                                   ModuleType.COC_FOR_SEQ_PARALLEL]
        all_reduce_types = [ModuleType.ORIGINAL_ALL_REDUCE,
                            ModuleType.REWRITE_ALL_REDUCE,
                            ModuleType.COC_FOR_ALL_REDUCE]

        if self.parallel_num not in [1, 2, 4, 8]:
            raise RuntimeError("coc_parallel_num must be either 1, 2, 4 or 8. Current value not supported")
        if self.coc_mode not in [-1, 0, 1, 2]:
            raise RuntimeError("coc_mode must be either 0, 1, or 2. Current value not supported")

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

        if self.coc_print_enabled:
            self.print_settings()

    def register_function(self):
        from .coc_parallel_linears_all_reduce_fused import FusedCOCRowAllReduceFunction
        from .coc_parallel_linears_all_reduce import COCColumnAllReduceFunction, COCRowAllReduceFunction
        from .coc_parallel_linears_sequence_parallel import COCColumnSeqParallelFunction, COCRowSeqParallelFunction
        from .rewrite_parallel_linears_all_reduce import RewriteColumnAllReduceFunction, RewriteRowAllReduceFunction
        from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, \
            RewriteRowSeqParallelFunction
        from .coc_parallel_linears_sequence_parallel_fused import FusedCOCColumnSeqParallelFunction, \
            FusedCOCRowSeqParallelFunction
        # not use fused op map
        map_type2autograd_class = {
            ModuleType.REWRITE_SEQ_PARALLEL: [RewriteColumnSeqParallelFunction,
                                              RewriteRowSeqParallelFunction],
            ModuleType.REWRITE_ALL_REDUCE: [RewriteColumnAllReduceFunction,
                                            RewriteRowAllReduceFunction],
            ModuleType.COC_FOR_SEQ_PARALLEL: [COCColumnSeqParallelFunction,
                                              COCRowSeqParallelFunction],
            ModuleType.COC_FOR_ALL_REDUCE: [COCColumnAllReduceFunction,
                                            COCRowAllReduceFunction]
        }
        # use fused op map
        map_fused_class = {
            True: [FusedCOCColumnSeqParallelFunction, FusedCOCRowSeqParallelFunction],
            False: [COCColumnAllReduceFunction, FusedCOCRowAllReduceFunction]
        }

        if self.coc_fused_kernel:
            print_on_device0("COC REPLACE WITH COC FUSED KERNEL SCRIPT!", self.coc_print_enabled)
            self.column_parallel_function, self.row_parallel_function = map_fused_class[self.sequence_parallel_enabled]
        elif "ORIGINAL" not in self.module_type.name:
            if "REWRITE" in self.module_type.name:
                print_on_device0("COC REPLACE WITH REWRITE SCRIPT!", self.coc_print_enabled)
            else:
                print_on_device0("COC REPLACE WITH COC SCRIPT!", self.coc_print_enabled)
            parallel_linear_autograd_class = map_type2autograd_class.get(self.module_type)
            if parallel_linear_autograd_class is None:
                raise RuntimeError("Module type is not matched.")
            self.column_parallel_function = parallel_linear_autograd_class[0]
            self.row_parallel_function = parallel_linear_autograd_class[1]
        else:
            print_on_device0("COC REPLACE NONE!", self.coc_print_enabled)
        self.coc_print_enabled = False


min_comm_config = MinCommConfig()
