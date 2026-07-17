# The timestamp intentionally precedes heavyweight imports so it includes import time.
# ruff: noqa: E402
import time

_TRAIN_START_TIME = time.time()
import json
import os.path
import gc
import copy
from functools import wraps
import torch
import torch.nn
import torch_npu
from megatron.training import print_rank_0
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import training
from megatron.training.training import print_datetime
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core import mpu, tensor_parallel
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options


class AutoPipeline:
    auto_pipeline = None

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.context = {'module': []}
        self.modules_hooks = []
        self.profiling_step = 0
        self.stop_profiling_step = 5
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
        if sync:
            torch.npu.synchronize()
        used_memory, _ = self.get_memory_status()
        torch.npu.reset_max_memory_allocated()
        state['memory'] = used_memory
        torch.npu.synchronize()
        state['time'] = time.time()
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
        if sync:
            torch.npu.synchronize()
        used_memory, _ = self.get_memory_status()
        max_mem = torch.npu.max_memory_allocated()
        state['peak_memory'] = max_mem - state['memory']
        state['memory'] = (used_memory - state['memory']) // self.unit_mb
        if 'pre_total_time' in state:
            state['forward_cnt'] += 1
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] += state['time']
            try:
                state['time'] = state['pre_total_time'] / state['forward_cnt']
            except ZeroDivisionError:
                state['time'] = 0
        else:
            state['forward_cnt'] = 0
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] = 0

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

    def get_comm_time(self, config, sync: bool):
        if torch.distributed.get_rank() == 0:
            if sync:
                torch.npu.synchronize()
            input_tensor = torch.ones(self.args.seq_length, self.args.micro_batch_size, self.args.hidden_size)
            start_time = time.time()
            p2p_communication.send_backward(input_tensor, config)
            comm_time = (time.time() - start_time) * 1000
            self.context['comm_time'] = comm_time
        else:
            self.context['comm_time'] = 0.028

    def get_modules_params_by_stages(self, init_memory, sync: bool):
        if self.args.pipeline_model_parallel_size == 2:
            self.context['first_stage_embed'] = self.args.padded_vocab_size * self.args.hidden_size
            self.context['last_stage_embed'] = self.args.padded_vocab_size * self.args.hidden_size
            attention_block = (
                3
                * self.args.hidden_size
                * self.args.num_attention_heads
                * (self.args.hidden_size / self.args.num_attention_heads)
                + self.args.hidden_size * self.args.hidden_size
                + self.args.hidden_size
                + self.args.hidden_size
            )
            ffn_block = (
                3 * self.args.ffn_hidden_size * self.args.hidden_size + self.args.hidden_size + self.args.hidden_size
            )
            per_trans_layer_param = attention_block + ffn_block
            per_trans_layer_param /= self.args.tensor_model_parallel_size
            self.context['per_trans_layer_param'] = per_trans_layer_param

        else:
            first_stage_param = 0
            per_trans_layer_param = 0
            last_stage_param = 0
            if sync:
                torch.npu.synchronize()
            first_stage_rank = 0
            last_stage_rank = torch.distributed.get_world_size() - 1
            layer_stage_rank = self.args.tensor_model_parallel_size

            first_stage_param = self.broadcast_param_in_ranks(first_stage_rank, first_stage_param, init_memory)
            last_stage_param = self.broadcast_param_in_ranks(last_stage_rank, last_stage_param, init_memory)
            per_trans_layer_param = self.broadcast_param_in_ranks(layer_stage_rank, per_trans_layer_param, init_memory)

            self.context['first_stage_embed'] = first_stage_param - per_trans_layer_param
            self.context['last_stage_embed'] = last_stage_param - per_trans_layer_param
            self.context['per_trans_layer_param'] = per_trans_layer_param

    def broadcast_param_in_ranks(self, src_rank, param, init_memory):
        if torch.distributed.get_rank() == src_rank:
            param = torch.npu.max_memory_allocated() / self.unit_mb - init_memory
        tmp_param = torch.cuda.IntTensor([param])
        torch.distributed.broadcast(tmp_param, src=src_rank)
        param = tmp_param.item()
        return param

    def update_args_for_profiling(self):
        args = get_args()
        if args.num_layers_per_virtual_pipeline_stage is None:
            args.num_layers = self.args.pipeline_model_parallel_size
            args.encoder_num_layers = self.args.pipeline_model_parallel_size
        args.train_iters = self.stop_profiling_step
        args.save = False
        args.log_interval = 10

    def restore_args_for_training(self):
        args = get_args()
        if args.num_layers_per_virtual_pipeline_stage is None:
            args.num_layers = self.args.num_layers
            args.encoder_num_layers = self.args.num_layers
        args.train_iters = self.args.train_iters
        args.optimizer = self.args.optimizer
        args.save = self.args.save
        args.log_interval = self.args.log_interval


def check_equal_model_configs(args, parsed_contents):
    model_index = 0
    for model_instance in parsed_contents:
        if (
            args.hidden_size == model_instance["model_configs"]["hidden_size"]
            and args.ffn_hidden_size == model_instance["model_configs"]["ffn_hidden_size"]
            and args.seq_length == model_instance["model_configs"]["seq_length"]
            and args.num_attention_heads == model_instance["model_configs"]["num_attention_heads"]
        ):
            return model_index
        else:
            model_index += 1
    return -1


def check_equal_parallel_configs(args, parsed_content):
    for parallel_instance in parsed_content["autopipeline_policy"]:
        if (
            args.num_layers == parallel_instance["num_layers"]
            and args.pipeline_model_parallel_size == parallel_instance["pipeline_model_parallel_size"]
            and args.tensor_model_parallel_size == parallel_instance["tensor_model_parallel_size"]
            and args.save_memory_ratio == parallel_instance["ratio"]
        ):
            return (
                parallel_instance["num_layer_list"],
                parallel_instance["recompute_module_list"],
                parallel_instance["recompute_type"],
            )
    return None, None, None


def check_skip_profiling(args, config_file):
    if os.path.exists(config_file):
        with open(config_file, encoding="utf-8") as config_json:
            config_contents = config_json.read()
        parsed_contents = json.loads(config_contents)
        index = check_equal_model_configs(args, parsed_contents)
        if index != -1:
            num_layer_list, recompute_module_list, recompute_type = check_equal_parallel_configs(
                args, parsed_contents[index]
            )
            if num_layer_list:
                return True, [(num_layer_list, recompute_module_list, (0, [0]), recompute_type)]
    return False, None


def set_recompute_mode(models):
    for model in models:
        for name, module in model.named_modules():
            if str.isdigit(name) and name != "0":
                module.forward = hook_checkpoint_forward(module.forward)


def hook_checkpoint_forward(forward_func):
    def custom_forward(*args, **kargs):
        def inside_forward(*args):
            return forward_func(*args, **kargs)

        return tensor_parallel.checkpoint(inside_forward, None, *args)

    return custom_forward


def get_auto_pipeline(args):
    if AutoPipeline.auto_pipeline is None:
        AutoPipeline.auto_pipeline = AutoPipeline(args)
    return AutoPipeline.auto_pipeline


def initialize_cfg_from_args_wrapper(initialize_cfg_from_args):
    @wraps(initialize_cfg_from_args)
    def wrapper(*args, **kwargs):
        from mindspeed.core import training as mc_training

        argument = get_args()
        disable_mc2 = argument.automated_pipeline and not mc_training.POLICY
        if not disable_mc2:
            initialize_cfg_from_args(*args, **kwargs)

    return wrapper


def autopipeline_profiling(
    cfg_container,
    model_provider,
    model_type,
    forward_step_func,
    train_valid_test_dataset_provider,
    process_non_loss_data_func,
    args,
):
    is_skip, policy = check_skip_profiling(args, config_file="autopipeline_config.json")
    if not is_skip:
        if getattr(args, 'tokenizer_type', None) is None:
            args.tokenizer_type = cfg_container.tokenizer.tokenizer_type or 'GPT2BPETokenizer'
        initialize_megatron()
        set_jit_fusion_options()
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        args = get_args()
        pipelining = get_auto_pipeline(args)
        pipelining.update_args_for_profiling()
        init_memory = torch.npu.max_memory_allocated() / pipelining.unit_mb
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
        pipelining.get_modules_params_by_stages(init_memory, sync=True)
        set_recompute_mode(models)
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
        pipelining.get_comm_time(config, sync=True)

        timers = get_timers()
        if timers('interval-time'):
            timers('interval-time').stop(barrier=True)

        for hook_handle in pipelining.modules_hooks:
            hook_handle.remove()
        pipelining.modules_hooks.clear()
        pipelining.restore_args_for_training()

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
        time.sleep(5)
        return pipelining.context, policy
    else:
        print_rank_0("[INFO] Found existed automated pipeline policy, apply it directly.")
        return None, policy
