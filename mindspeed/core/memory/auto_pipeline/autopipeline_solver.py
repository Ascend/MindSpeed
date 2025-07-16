import os
import json
import statistics
import math
import time
import multiprocessing
from functools import wraps
import torch
import megatron.training.global_vars
from megatron.training import get_args
from megatron.training import print_rank_0
from .autopipeline import check_equal_model_configs
import mindspeed.model.transformer as mindspeed_transformer
import megatron.core.parallel_state as megatron_parallel_state
import mindspeed.core.parallel_state as mindspeed_parallel_state


class AutoPipelineSolver():
    def __init__(self, context):
        self.context = context
        self.MB_SIZE = 1024 * 1024
        # model configurations
        args = get_args()
        self.num_layers = args.num_layers
        self.vocab_size = args.padded_vocab_size
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.micro_batch_size = args.micro_batch_size
        self.global_batch_size = args.global_batch_size
        self.seq_length = args.seq_length
        self.num_attention_heads = args.num_attention_heads
        self.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        self.tensor_model_parallel_size = args.tensor_model_parallel_size

        self.first_stage_embed = 0
        self.last_stage_embed = 0
        self.per_trans_layer_param = 0
        self.embed_activation = 0

        self.forward_time = 0
        self.forward_activation = 0
        self.mlp_forward_time = 0
        self.comm_time = 0
        self.forward_mlp_activation = 0
        self.attention_forward_time = 0
        self.forward_attention_activation = 0
        self.layer_forward_time = 0
        self.forward_layer_activation = 0
        self.parse_profile()

        # hyper params settings
        self.ratio = args.save_memory_ratio if args.save_memory_ratio == 1.0 else 1 - args.save_memory_ratio
        self.min_layer, self.max_layer = self.get_min_max_layer()
        self.target_memory = self.set_target_memory()

        # auto pipeline search result
        self.ans = []
        self.backup = []
        self.backup_min_mem = 0
        # auto pipeline policy
        self.policy = []
        self.optimal_sch = []
        self.minn = []


    def find_target_profile(self, module, target, profile_type):
        context = self.context
        while module in context:
            for sub_context in context[module]:
                if sub_context["name"] == target:
                    return sub_context[profile_type]
                else:
                    context = sub_context
        return 0


    def get_min_max_layer(self):
        layer_avg = round(self.num_layers / self.pipeline_model_parallel_size)
        if 1 <= layer_avg <= 4:
            layer_range = 0
        elif 5 <= layer_avg < 8:
            layer_range = 1
        else:
            layer_range = 2
        return layer_avg - layer_range, layer_avg + layer_range


    def parse_profile(self):
        self.first_stage_embed = self.context["first_stage_embed"] * self.MB_SIZE
        self.last_stage_embed = self.context["last_stage_embed"] * self.MB_SIZE
        self.per_trans_layer_param = self.context["per_trans_layer_param"] * self.MB_SIZE
        self.embed_activation = self.find_target_profile("layers", "embedding", "memory") * self.MB_SIZE

        self.forward_time = self.find_target_profile("layers", "module", "time")
        self.mlp_forward_time = self.find_target_profile("layers", "mlp", "time")
        self.attention_forward_time = self.find_target_profile("layers", "self_attention", "time")
        self.layer_forward_time = self.find_target_profile("layers", "0", "time")
        self.comm_time = self.context["comm_time"]
        self.forward_activation = self.find_target_profile("layers", "module", "memory") * self.MB_SIZE
        self.forward_mlp_activation = self.find_target_profile("layers", "mlp", "memory") * self.MB_SIZE
        self.forward_attention_activation = self.find_target_profile("layers", "self_attention", "memory") * self.MB_SIZE
        self.forward_layer_activation = self.find_target_profile("layers", "0", "memory") * self.MB_SIZE


    def naive_search(self, module_type, answer_queue):

        def dfs_build_layers(prefix_n_layers, cur_layers_sum):

            if len(prefix_n_layers) > self.pipeline_model_parallel_size:
                return
            if cur_layers_sum > self.num_layers:
                return
            if 2 <= len(prefix_n_layers) < self.pipeline_model_parallel_size:
                if prefix_n_layers[-1] < prefix_n_layers[-2]:
                    return

            if len(prefix_n_layers) == self.pipeline_model_parallel_size and cur_layers_sum == self.num_layers:
                status, prefix_recomp_modules, mem_set = self.get_recompute_modules(prefix_n_layers, self.pipeline_model_parallel_size, module_type)
                if status:
                    answer_queue.append((prefix_n_layers, prefix_recomp_modules, mem_set, module_type))
                if len(answer_queue) == 0 and len(self.ans) == 0:
                    if len(self.backup) == 0:
                        self.backup.append((prefix_n_layers, prefix_recomp_modules, mem_set, module_type))
                    else:
                        temp_min_mem = min(mem_set[1])
                        if temp_min_mem < self.backup_min_mem:
                            self.backup_min_mem = temp_min_mem
                            self.backup[0] = (prefix_n_layers, prefix_recomp_modules, mem_set, module_type)
                return

            for cur_n_layer in range(self.max_layer, self.min_layer - 1, -1):
                dfs_build_layers(prefix_n_layers + [cur_n_layer], cur_layers_sum + cur_n_layer)

        for prefix_n_layer in range(self.max_layer, self.min_layer - 1, -1):
            dfs_build_layers([prefix_n_layer], prefix_n_layer)

        return answer_queue


    def main_search(self):
        mlp_answer_queue, attn_answer_queue, layer_answer_queue = [], [], []
        mlp_answer_queue = self.naive_search(0, mlp_answer_queue)
        self.ans = mlp_answer_queue
        attn_answer_queue = self.naive_search(1, attn_answer_queue)
        self.ans += attn_answer_queue
        layer_answer_queue = self.naive_search(2, layer_answer_queue)
        self.ans += layer_answer_queue

        return self.ans


    def cal_module_param(self, module_type):

        per_layer_activation_param = self.forward_activation
        per_recompute_module_param = 0
        if module_type == 0:
            # mlp activation param
            per_recompute_module_param = self.forward_mlp_activation
        if module_type == 1:
            # attn param
            per_recompute_module_param = self.forward_attention_activation
        if module_type == 2:
            # layer param
            per_recompute_module_param = 2 * self.seq_length * self.micro_batch_size * self.hidden_size

        return per_layer_activation_param, per_recompute_module_param


    def cal_model_mem(self, per_layer_activation_param, per_recompute_module_param, n_layer, n_recompute_module, parallel_num, \
                      stage_num):
        if stage_num == 0:
            stage_max_optimizer_mem = (self.first_stage_embed + self.per_trans_layer_param * n_layer) + self.embed_activation
            model_mem = self.first_stage_embed + self.per_trans_layer_param * n_layer \
                        + stage_max_optimizer_mem \
                        + per_layer_activation_param * n_layer * parallel_num
        elif stage_num == self.pipeline_model_parallel_size - 1:
            stage_max_optimizer_mem = (self.last_stage_embed + self.per_trans_layer_param * n_layer) + self.embed_activation
            model_mem = self.last_stage_embed + self.per_trans_layer_param * n_layer \
                        + stage_max_optimizer_mem \
                        + per_layer_activation_param * n_layer * parallel_num
        else:
            stage_max_optimizer_mem = self.per_trans_layer_param * n_layer
            model_mem = self.per_trans_layer_param * n_layer \
                        + stage_max_optimizer_mem \
                        + per_layer_activation_param * n_layer * parallel_num
        return model_mem


    def set_target_memory(self):
        per_layer_activation_param, per_recompute_module_param = self.cal_module_param(0)
        stage_num = 0
        default_n_layers_mems = []
        while stage_num < self.pipeline_model_parallel_size:
            default_layer_mem = self.cal_model_mem(per_layer_activation_param, per_recompute_module_param,
                               self.num_layers/self.pipeline_model_parallel_size, 0,
                               self.pipeline_model_parallel_size - stage_num, stage_num)
            default_n_layers_mems.append(default_layer_mem)
            stage_num += 1

        target_memory = sum(default_n_layers_mems)/len(default_n_layers_mems)
        if self.ratio < 1.0:
            target_memory = max(default_n_layers_mems)
        return target_memory


    def get_recompute_modules(self, n_layers, num_pp_stage, module_type):
        per_layer_activation_param, per_recompute_module_param = self.cal_module_param(module_type)
        init_recompute_modules = []
        new_n_layers_mems = []
        stage_num = 0
        status = True

        while stage_num < len(n_layers):
            init_layer_mem = self.cal_model_mem(per_layer_activation_param, per_recompute_module_param,\
                                                n_layers[stage_num], 0,
                                                num_pp_stage - stage_num, stage_num)
            if init_layer_mem <= self.target_memory * self.ratio:
                n_recompute_module = 0
                init_recompute_modules.append(n_recompute_module)
            else:
                if (per_recompute_module_param * (num_pp_stage - stage_num) / self.MB_SIZE) == 0:
                    n_recompute_module = 0
                else:
                    n_recompute_module = math.ceil((init_layer_mem / self.MB_SIZE - self.target_memory * self.ratio / self.MB_SIZE) / (per_recompute_module_param * (num_pp_stage - stage_num) / self.MB_SIZE))
                if n_recompute_module > n_layers[stage_num]:
                    status = False
                    n_recompute_module = n_layers[stage_num]
                    init_recompute_modules.append(n_recompute_module)
                else:
                    init_recompute_modules.append(n_recompute_module)

            init_layer_mem = self.cal_model_mem(per_layer_activation_param, per_recompute_module_param,
                                                n_layers[stage_num], n_recompute_module,
                                                num_pp_stage - stage_num, stage_num)
            init_layer_mem -= per_recompute_module_param*n_recompute_module
            init_layer_mem /= self.MB_SIZE
            new_n_layers_mems.append(init_layer_mem)
            stage_num += 1

        return status, init_recompute_modules, (self.target_memory/self.MB_SIZE, new_n_layers_mems)


    def dp(self, examples):
        # lookup duration via parallel params
        (Fwd, Bwd, ComFwd, ComBwd) = self.forward_time, self.forward_time * 1.3, self.comm_time, self.comm_time

        RecompFwd = 0
        module_type = examples[3]
        if module_type == 0:
            RecompFwd = self.mlp_forward_time
        elif module_type == 1:
            RecompFwd = self.attention_forward_time
        elif module_type == 2:
            RecompFwd = self.layer_forward_time

        # to remember that n_layers can be divided by num_pp_stage
        n_layers = [0] + examples[0]
        n_recompute_layers = [0] + examples[1]
        num_pp_stage = self.pipeline_model_parallel_size

        # number of micro-batch-size is 256
        mbs = [self.micro_batch_size for _ in range(self.global_batch_size)]
        num_microbatch = len(mbs)
        mbs = [0] + mbs

        SF = [[0 for i in range(num_microbatch + 1)] for _ in range(num_pp_stage + 1)]  # start of forward 图中蓝色的左边
        EF = [[0 for i in range(num_microbatch + 1)] for _ in range(num_pp_stage + 1)]  # end of forward 图中蓝色的右边

        SB = [[0 for i in range(num_microbatch + 1)] for _ in range(num_pp_stage + 1)]  # start of backward 图中绿色的左边
        EB = [[0 for i in range(num_microbatch + 1)] for _ in range(num_pp_stage + 1)]  # end of backward 图中绿色的右边

        warmup = [num_pp_stage - p - 1 for p in range(num_pp_stage)]
        remaining = [num_microbatch - warmup[p] for p in range(num_pp_stage)]

        # for dp, p and m start with 1
        # warmup: only forward processing, add activations
        for p in range(1, num_pp_stage + 1):
            for m in range(1, num_pp_stage - p + 1):
                SF[p][m] = max(EF[p][m - 1], EF[p - 1][m] + ComFwd)
                EF[p][m] = SF[p][m] + Fwd * n_layers[p]

        # 1f1b
        for num_1f1b in range(1, num_microbatch + 1):

            # # fwd of 1f1b
            for p in range(1, num_pp_stage + 1):
                if remaining[p - 1] < num_1f1b:
                    # this means it have to work for cool down phase
                    continue

                m = warmup[p - 1] + num_1f1b
                if p == 1:
                    EF[0][m] = max(EF[1]) - ComFwd

                SF[p][m] = max(EB[p][m + p - num_pp_stage - 1], EF[p - 1][m] + ComFwd)
                EF[p][m] = SF[p][m] + Fwd * n_layers[p]

            # bwd of 1f1b
            for p in range(num_pp_stage, 0, -1):
                m = num_1f1b
                if remaining[p - 1] < num_1f1b:
                    # this means it have to work for cool down phase
                    continue
                if p == num_pp_stage:
                    SB[p][m] = EF[p][m + num_pp_stage - p]
                else:
                    SB[p][m] = max(EF[p][m + num_pp_stage - p], EB[p + 1][m] + ComBwd)

                EB[p][m] = SB[p][m] + Bwd * n_layers[p] + RecompFwd * n_recompute_layers[p]

            # cooldown
            for p in range(num_pp_stage, 0, -1):
                m = num_1f1b
                if remaining[p - 1] >= num_1f1b:
                    continue
                SB[p][m] = max(EB[p][m - 1], EB[p + 1][m] + ComBwd)
                EB[p][m] = SB[p][m] + Bwd * n_layers[p] + RecompFwd * n_recompute_layers[p]

        itertime = max([max(EB[p]) for p in range(num_pp_stage)])
        self.policy.append((itertime, examples))
        return


    def find_top_optimal_schedule(self):
        self.main_search()
        for examples in self.ans:
            self.dp(examples)

        if len(self.policy) > 0:
            min_itertime = self.policy[0][0]
            self.minn.append(min_itertime)
            self.optimal_sch.append(self.policy[0][1])
            for idx, res in enumerate(self.policy):
                if res[0] < min_itertime:
                    min_itertime = res[0]
                    self.minn[0] = min_itertime
                    self.optimal_sch[0] = res[1]
        else:
            print_rank_0("[INFO] [Autopipeline Policy Time Searching Stage] No strategy is satisfied. We will apply the minimum memory strategy instead.")
            self.minn.append(0)
            self.optimal_sch.append(self.backup[0])

        return self.optimal_sch, self.minn


def broadcast_policy_in_ranks(src_rank, policy=None):
    args = get_args()
    num_layer_list = args.pipeline_model_parallel_size * [0]
    recompute_module_list = args.pipeline_model_parallel_size * [0]
    recompute_type = 0
    if torch.distributed.get_rank() == 0:
        num_layer_list = policy[0][0]
        recompute_module_list = policy[0][1]
        recompute_type = policy[0][3]

    tmp_layer_list = torch.cuda.IntTensor(num_layer_list)
    torch.distributed.broadcast(tmp_layer_list, src=src_rank)
    args.num_layer_list = tmp_layer_list.tolist()

    tmp_recompute_module_list = torch.cuda.IntTensor(recompute_module_list)
    torch.distributed.broadcast(tmp_recompute_module_list, src=src_rank)
    args.recompute_module_list = tmp_recompute_module_list.tolist()

    tmp_recompute_type = torch.cuda.IntTensor([recompute_type])
    torch.distributed.broadcast(tmp_recompute_type, src=src_rank)
    args.recompute_type = tmp_recompute_type.item()


def destroy_global_vars():
    megatron.training.global_vars._GLOBAL_ARGS = None
    megatron.training.global_vars._GLOBAL_RETRO_ARGS = None
    megatron.training.global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    megatron.training.global_vars._GLOBAL_TOKENIZER = None
    megatron.training.global_vars._GLOBAL_TENSORBOARD_WRITER = None
    megatron.training.global_vars._GLOBAL_WANDB_WRITER = None
    megatron.training.global_vars._GLOBAL_ADLR_AUTORESUME = None
    megatron.training.global_vars._GLOBAL_TIMERS = None
    megatron.training.global_vars._GLOBAL_SIGNAL_HANDLER = None
    megatron_parallel_state._EXPERT_PARALLEL_GROUP = None
    mindspeed_transformer._GLOBAL_ATTN_MASK = None


def destroy_global_parallel_group():
    global_parallel_group = [
        megatron_parallel_state._MODEL_PARALLEL_GROUP,
        megatron_parallel_state._TENSOR_MODEL_PARALLEL_GROUP,
        megatron_parallel_state._PIPELINE_MODEL_PARALLEL_GROUP,
        mindspeed_parallel_state._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM,
        megatron_parallel_state._DATA_PARALLEL_GROUP,
        megatron_parallel_state._DATA_PARALLEL_GROUP_WITH_CP,
        megatron_parallel_state._CONTEXT_PARALLEL_GROUP,
        megatron_parallel_state._EMBEDDING_GROUP,
        megatron_parallel_state._POSITION_EMBEDDING_GROUP,
        megatron_parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP,
        megatron_parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP,
        megatron_parallel_state._EXPERT_MODEL_PARALLEL_GROUP,
        megatron_parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP,
        megatron_parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP
    ]
    for gid in range(len(global_parallel_group)):
        if global_parallel_group[gid]:
            torch.distributed.destroy_process_group(global_parallel_group[gid])
        torch.distributed.barrier()

    megatron_parallel_state._MODEL_PARALLEL_GROUP = None
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GROUP = None
    megatron_parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = None
    mindspeed_parallel_state._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None
    megatron_parallel_state._DATA_PARALLEL_GROUP = None
    megatron_parallel_state._DATA_PARALLEL_GROUP_WITH_CP = None
    megatron_parallel_state._CONTEXT_PARALLEL_GROUP = None
    megatron_parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = None
    megatron_parallel_state._EMBEDDING_GROUP = None
    megatron_parallel_state._POSITION_EMBEDDING_GROUP = None
    megatron_parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = None
    megatron_parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    megatron_parallel_state._EXPERT_MODEL_PARALLEL_GROUP = None
    megatron_parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    megatron_parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    megatron_parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    megatron_parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    megatron_parallel_state._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = None
    megatron_parallel_state._MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    megatron_parallel_state._GLOBAL_MEMORY_BUFFER = None
    megatron_parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    megatron_parallel_state._MPU_EXPERT_MODEL_PARALLEL_RANK = None


def destroy_model_parallel_profiling_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper(*args, **kwargs):
        argument = get_args()
        enable_profiling_destroy = (argument.automated_pipeline and not argument.num_layer_list) \
                                   or (argument.automated_pipeline_perf and not argument.optimized_mbs_list)
        if enable_profiling_destroy:
            destroy_global_parallel_group()
        else:
            destroy_model_parallel(*args, **kwargs)
    return wrapper


def get_profiling_data(policy, args):
    instance = {"model_configs": {
        "vocab_size": args.padded_vocab_size,
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "seq_length": args.seq_length,
        "num_attention_heads": args.num_attention_heads
    }, "autopipeline_policy": [{
        "num_layers": args.num_layers,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "ratio": args.save_memory_ratio,
        "num_layer_list": policy[0][0],
        "recompute_module_list": policy[0][1],
        "recompute_type": policy[0][3]
    }]}
    return instance


def save_profiling_data(policy, config_file):

    args = get_args()
    instance = get_profiling_data(policy, args)
    if os.path.exists(config_file):
        with open(config_file, "r") as config_json:
            config_contents = config_json.read()
        parsed_contents = json.loads(config_contents)
        index = check_equal_model_configs(args, parsed_contents)
        if index != -1:
            if "autopipeline_policy" in parsed_contents[index]:
                parsed_contents[index]["autopipeline_policy"].append(instance["autopipeline_policy"][0])
        else:
            parsed_contents.append(instance)
        with open(config_file, "w") as f:
            json.dump(parsed_contents, f, ensure_ascii=False)
            os.chmod(config_file, 0o640)
    else:
        with open(config_file, "w") as f:
            json.dump([instance], f, ensure_ascii=False)
            os.chmod(config_file, 0o640)


def solve_autopipeline(context):
    if torch.distributed.get_rank() == 0:
        solver = AutoPipelineSolver(context)
        solver.find_top_optimal_schedule()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print_rank_0("=============================================")
        print_rank_0("[INFO] Applied Automated Pipeline Policy: ")
        print_rank_0("network layer setting: " + str(solver.optimal_sch[0][0]))
        print_rank_0("recompute policy setting: " + str(solver.optimal_sch[0][1]))
        print_rank_0("recompute type: " + str(solver.optimal_sch[0][3]))
        print_rank_0("=============================================")
        save_profiling_data(solver.optimal_sch, config_file="autopipeline_config.json")
        return solver.optimal_sch
    else:
        return [([0], [0], (0, [0]), 0)]