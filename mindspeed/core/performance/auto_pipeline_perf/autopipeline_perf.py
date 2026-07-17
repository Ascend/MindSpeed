# The timestamp intentionally precedes heavyweight imports so it includes import time.
# ruff: noqa: E402
import time

_TRAIN_START_TIME = time.time()
import json
import os.path
import gc
import copy
import torch
import torch.nn
import torch_npu
from megatron.training import print_rank_0
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import training
from megatron.training.training import print_datetime
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options


profile_context = {"fwd_time": [], "bwd_time": []}


class AutoPipeline_Perf:
    autopipeline_perf = None

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.context = {'module': []}
        self.modules_hooks = []
        self.profiling_step = 0
        self.stop_profiling_step = 3
        self.unit_mb = 1024 * 1024

    @staticmethod
    def get_memory_status():
        used_memory = torch.npu.memory_allocated()
        reserved_memory = torch.npu.memory_reserved()
        return used_memory, reserved_memory

    def _cal_tensor_size(self, tensor):
        try:
            return tensor.numel() * tensor.element_size() / self.unit_mb
        except ZeroDivisionError:
            return 0

    def pre_hook_func(self, state, sync: bool, *args, **kargs):
        used_memory, _ = self.get_memory_status()
        torch.npu.reset_max_memory_allocated()
        state['memory'] = used_memory
        size = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                size += self._cal_tensor_size(arg)
            elif isinstance(arg, (list, tuple)):
                for t in arg:
                    if isinstance(t, torch.Tensor):
                        size += self._cal_tensor_size(t)
        state['input'] = size

    def post_hook_func(self, state, sync: bool, *args, **kargs):
        used_memory, _ = self.get_memory_status()
        max_mem = torch.npu.max_memory_allocated()
        state['peak_memory'] = max_mem - state['memory']
        state['memory'] = (used_memory - state['memory']) // self.unit_mb

    def forward_pre_hook(self, name, parent_ctx, ctx):
        if self.profiling_step < self.stop_profiling_step:
            ctx['name'] = name
            if 'layers' in parent_ctx:
                parent_ctx['layers'].append(ctx)

        def hook(module, *args, **kargs):
            if self.profiling_step < self.stop_profiling_step:
                if 'module' in self.context:
                    self.context['module'].append(ctx)
                self.pre_hook_func(ctx, True, *args, **kargs)

        return hook

    def forward_post_hook(self, ctx):
        def hook(module, *args, **kargs):
            if self.profiling_step < self.stop_profiling_step:
                self.post_hook_func(ctx, True, *args)
                if 'module' in self.context:
                    self.context['module'].pop()

        return hook

    def register_recursive_hook(self, prefix_name, model, ctx):
        for name, module in model.named_children():
            if 'layers' not in ctx:
                ctx['layers'] = []
            current_ctx = {}

            next_name = prefix_name + "." + name if prefix_name != "" else name
            if next_name == "module.module":
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(name, ctx, current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
                self.register_recursive_hook(next_name, module, current_ctx)

    def step_hook(self, model):
        self.profiling_step += 1

    def hook_step_func(self, step_func, models):
        def custom_step_func(*args, **kargs):
            result = step_func(*args, **kargs)
            if self.profiling_step < self.stop_profiling_step:
                used_memory, reserved_memory = self.get_memory_status()
                self.context['used_mem'] = used_memory // self.unit_mb
            if isinstance(models, list):
                for model in models:
                    self.step_hook(model)
            else:
                self.step_hook(models)
            return result

        return custom_step_func

    def remove_outliers(self, data, m=2):
        data = sorted(data)
        median = data[len(data) // 2]
        deviation = [x for x in data if median - m * median < x < median + m * median]
        return deviation

    def get_forward_context(self):
        global profile_context
        if "fwd_time" in profile_context:
            fwd_time_list = self.remove_outliers(profile_context["fwd_time"])
            try:
                self.context["fwd_time"] = sum(fwd_time_list) / len(fwd_time_list)
            except ZeroDivisionError:
                print("[Error] Divided by zero.")
        else:
            self.context["fwd_time"] = 0

    def get_backward_context(self):
        global profile_context
        if "bwd_time" in profile_context:
            bwd_time_list = self.remove_outliers(profile_context["bwd_time"])
            try:
                self.context["bwd_time"] = sum(bwd_time_list) / len(bwd_time_list)
            except ZeroDivisionError:
                print("[Error] Divided by zero.")
        else:
            self.context["bwd_time"] = 0

    def clear_global_context(self):
        global profile_context
        profile_context["fwd_time"] = []
        profile_context["bwd_time"] = []

    def get_comm_time(self, config, sync: bool):
        if torch.distributed.get_rank() == 0:
            if sync:
                torch.cuda.synchronize()
            input_tensor = torch.ones(self.args.seq_length, self.args.micro_batch_size, self.args.hidden_size)
            start_time = time.time()
            p2p_communication.send_backward(input_tensor, config)
            comm_time = (time.time() - start_time) * 1000
            self.context['comm_time'] = comm_time
        else:
            self.context['comm_time'] = 0.028

    def get_peak_memory(self, sync: bool):
        if sync:
            torch.cuda.synchronize()
        max_mem = torch.npu.max_memory_allocated() / (1 << 20)
        self.context['peak_memory'] = max_mem

    def get_smi_peak_memory(self, sync: bool):
        if sync:
            torch.cuda.synchronize()
        mem_infos = torch.npu.mem_get_info()
        smi_peak_memory = (mem_infos[1] - mem_infos[0]) / (1 << 20)
        self.context['smi_peak_memory'] = smi_peak_memory

    def get_smi_left_memory(self, sync: bool):
        if sync:
            torch.cuda.synchronize()
        mem_infos = torch.npu.mem_get_info()
        smi_left_memory = mem_infos[0] / (1 << 20)
        self.context['smi_left_memory'] = smi_left_memory

    def get_data_parallel_size(self, data_parallel_size):
        if data_parallel_size:
            self.context['data_parallel_size'] = data_parallel_size
        else:
            self.context['data_parallel_size'] = 1

    def broadcast_param_in_ranks(self, src_rank, param, init_memory):
        if torch.distributed.get_rank() == src_rank:
            try:
                param = torch.npu.max_memory_allocated() / self.unit_mb - init_memory
            except ZeroDivisionError:
                print("[Error] Divided by zero.")
        tmp_param = torch.cuda.IntTensor([param])
        torch.distributed.broadcast(tmp_param, src=src_rank)
        param = tmp_param.item()
        return param

    def update_args_for_profiling(self, micro_batch_size=None):
        args = get_args()
        args.train_iters = self.stop_profiling_step
        if micro_batch_size:
            args.micro_batch_size = micro_batch_size
            args.global_batch_size = args.micro_batch_size * 16
        args.save = False
        args.log_interval = 10

    def restore_args_for_training(self):
        args = get_args()
        if args.num_layers_per_virtual_pipeline_stage is None:
            args.num_layers = self.args.num_layers
            args.encoder_num_layers = self.args.num_layers
        args.train_iters = self.args.train_iters
        args.micro_batch_size = self.args.micro_batch_size
        args.global_batch_size = self.args.global_batch_size
        args.save = self.args.save
        args.log_interval = self.args.log_interval


def check_equal_model_configs(args, parsed_contents):
    model_index = 0
    for model_instance in parsed_contents:
        if (
            args.hidden_size == model_instance.get("model_configs", {}).get("hidden_size")
            and args.ffn_hidden_size == model_instance.get("model_configs", {}).get("ffn_hidden_size")
            and args.seq_length == model_instance.get("model_configs", {}).get("seq_length")
            and args.num_attention_heads == model_instance.get("model_configs", {}).get("num_attention_heads")
        ):
            return model_index
        else:
            model_index += 1
    return -1


def check_equal_parallel_configs(args, parsed_content):
    for parallel_instance in parsed_content.get("optimpipeline_policy"):
        if (
            args.num_layers == parallel_instance.get("num_layers")
            and args.pipeline_model_parallel_size == parallel_instance.get("pipeline_model_parallel_size")
            and args.tensor_model_parallel_size == parallel_instance.get("tensor_model_parallel_size")
            and args.micro_batch_size == parallel_instance.get("micro_batch_size")
            and args.global_batch_size == parallel_instance.get("global_batch_size")
        ):
            return (
                parallel_instance.get("enable_scheduler"),
                parallel_instance.get("optimized_mbs_list"),
                parallel_instance.get("pp_schedule_list"),
                parallel_instance.get("optimal_layers"),
            )
    return None, None, None, None


def check_skip_profiling(args, config_file):
    if os.path.exists(config_file):
        with open(config_file, encoding="utf-8") as config_json:
            config_contents = config_json.read()
        parsed_contents = json.loads(config_contents)
        index = check_equal_model_configs(args, parsed_contents)
        if index != -1:
            optimized_type, optimized_mbs_list, pp_schedule_list, optimal_layers = check_equal_parallel_configs(
                args, parsed_contents[index]
            )
            if optimized_mbs_list or pp_schedule_list:
                return True, (optimized_type, optimized_mbs_list, pp_schedule_list, optimal_layers)
    return False, (None, None, None, None)


def check_out_of_memory(args, context, mbs_tries):
    total_memory = torch_npu.npu.get_device_properties(0).total_memory / (1 << 20)
    per_activation_memory_allocated = context["layers"][0]["memory"] // mbs_tries
    predict_next_max_memory_allocated = (
        context["smi_peak_memory"] + per_activation_memory_allocated * args.pipeline_model_parallel_size + 1000
    )
    return predict_next_max_memory_allocated > total_memory


def broadcast_skip_in_ranks(src_rank, policy):
    is_skip = [False]
    if torch.distributed.get_rank() == src_rank:
        is_skip = [policy]
    tmp_is_skip = torch.cuda.BoolTensor(is_skip)
    torch.distributed.broadcast(tmp_is_skip, src=src_rank)
    return tmp_is_skip.item()


def calculate_num_of_activations(context):
    total_memory = torch_npu.npu.get_device_properties(0).total_memory / (1 << 20)
    activation_memory_allocated = context["layers"][0]["memory"]
    num_of_activations_left = (total_memory - context["smi_peak_memory"]) // activation_memory_allocated
    return int(num_of_activations_left)


def get_autopipeline_perf(args):
    AutoPipeline_Perf.autopipeline_perf = AutoPipeline_Perf(args)
    return AutoPipeline_Perf.autopipeline_perf


def autopipelineperf_profiling(
    cfg_container,
    mbs_tries,
    model_provider,
    model_type,
    forward_step_func,
    train_valid_test_dataset_provider,
    process_non_loss_data_func,
):
    args = get_args()
    configured_tokenizer_type = cfg_container.tokenizer.tokenizer_type
    if args.tokenizer_type is None:
        args.tokenizer_type = configured_tokenizer_type or 'GPT2BPETokenizer'
    initialize_megatron()
    set_jit_fusion_options()
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    args = get_args()
    pipelining = get_autopipeline_perf(args)
    pipelining.update_args_for_profiling(mbs_tries)
    models, optimizer, lr_scheduler = training.setup_model_and_optimizer(model_provider, model_type)
    optimizer.step = pipelining.hook_step_func(optimizer.step, models)
    config = training.get_model_config(models[0])

    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        for i in range(len(models)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = training.build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
    else:
        train_data_iterator, valid_data_iterator, _ = training.build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider
        )
    if isinstance(models, list):
        for model in models:
            pipelining.register_recursive_hook("module", model, pipelining.context)
    else:
        pipelining.register_recursive_hook("module", models, pipelining.context)
    checkpointing_context = {}
    training.train(
        forward_step_func,
        models,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
    )
    pipelining.get_smi_peak_memory(sync=True)
    pipelining.get_smi_left_memory(sync=True)
    pipelining.get_comm_time(config, sync=True)
    pipelining.get_peak_memory(sync=True)
    pipelining.get_data_parallel_size(args.data_parallel_size)
    pipelining.get_forward_context()
    pipelining.get_backward_context()
    pipelining.clear_global_context()

    timers = get_timers()
    if timers('interval-time'):
        timers('interval-time').stop(barrier=True)

    for hook_handle in pipelining.modules_hooks:
        hook_handle.remove()
    pipelining.modules_hooks.clear()
    pipelining.restore_args_for_training()

    if hasattr(optimizer, 'chained_optimizers'):
        for op in optimizer.chained_optimizers:
            for key, value in op.optimizer.state.items():
                key.detach()
                key.grad = None
                key.storage().resize_(0)
                if "momentum_buffer" in value:
                    value["momentum_buffer"].detach()
                    value["momentum_buffer"].grad = None
                    value["momentum_buffer"].storage().resize_(0)
            for ofg in op.param_groups:
                if "params" in ofg:
                    for og in ofg["params"]:
                        og.detach()
                        og.grad = None
                        og.storage().resize_(0)
    else:
        for key, value in optimizer.optimizer.state.items():
            key.detach()
            key.grad = None
            key.storage().resize_(0)
            if "momentum_buffer" in value:
                value["momentum_buffer"].detach()
                value["momentum_buffer"].grad = None
                value["momentum_buffer"].storage().resize_(0)
        for ofg in optimizer.param_groups:
            if "params" in ofg:
                for og in ofg["params"]:
                    og.detach()
                    og.grad = None
                    og.storage().resize_(0)
    for md in models:
        for param in md.parameters():
            param.detach()
            param.grad = None
            param.storage().resize_(0)
        for param_tensor in md.state_dict():
            if md.state_dict()[param_tensor] is not None:
                md.state_dict()[param_tensor].detach()
                md.state_dict()[param_tensor].grad = None
                md.state_dict()[param_tensor].storage().resize_(0)

    gc.collect()
    torch_npu.npu.empty_cache()
    return pipelining.context
