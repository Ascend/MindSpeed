import itertools
import copy
import torch
import numpy as np
from megatron.training import get_args
from megatron.core import mpu


def expert_placement_init(self, ep_size):
    global_args = get_args()
    self.expert_load_prediction = torch.zeros(self.config.num_moe_experts, device=torch.cuda.current_device())
    self.num_tokens_per_expert_origin = torch.zeros(self.config.num_moe_experts, device=torch.cuda.current_device())
    if getattr(global_args, "enable_expert_placement", False):
        self.expert_mapping = torch.arange(self.config.num_moe_experts, dtype=torch.int, device=torch.cuda.current_device())
        self.ep_world_size = ep_size
        self.input_splits_for_expert_placement = np.zeros(self.ep_world_size, dtype=np.int32)
        self.output_splits_for_expert_placement = np.zeros(self.ep_world_size, dtype=np.int32)

        self.local_resorted_relative_mapping_indices = torch.zeros(self.experts.num_local_experts, 
                                                                   dtype=torch.int, device=torch.cuda.current_device())
        self.new_local_expert_sorted_indices = torch.zeros(self.experts.num_local_experts, 
                                                           dtype=torch.int, device=torch.cuda.current_device())
        self.expert_placement_optimizer = ExpertDynamicplacement()


def predict_expert_load(self, num_tokens_per_expert):
    with torch.no_grad():
        global_args = get_args()
        if getattr(global_args, "enable_expert_placement", False):
            if global_args.curr_iteration % global_args.expert_placement_freq == global_args.expert_placement_freq - 1:
                return
            # Map back the num_tokens to the original order of experts
            self.num_tokens_per_expert_origin = num_tokens_per_expert[self.expert_mapping]
        else:
            self.num_tokens_per_expert_origin = num_tokens_per_expert
        # Predict expert load by exponential moving average method
        ema_weight = 0.9
        self.expert_load_prediction = ema_weight * self.expert_load_prediction \
                                        + (1 - ema_weight) * self.num_tokens_per_expert_origin


def gather_expert_load_data_parallel(num_tokens_per_expert):
    data_parallel_group = mpu.get_data_modulo_expert_parallel_group()    # test allreduce
    world_size = torch.distributed.get_world_size(group=data_parallel_group)
    if world_size > 1:
        dim_size = torch.Size((world_size, num_tokens_per_expert.shape[0]))

        num_tokens_per_expert_overall = torch.empty(
                                    dim_size, dtype=num_tokens_per_expert.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(num_tokens_per_expert_overall, 
                                            num_tokens_per_expert.contiguous(), group=data_parallel_group)
        num_tokens_per_expert_overall = num_tokens_per_expert_overall.sum(axis=0) / world_size
    else:
        num_tokens_per_expert_overall = num_tokens_per_expert.clone()
    return num_tokens_per_expert_overall


def print_expert_load(args, model, iteration):
    dp = mpu.get_data_parallel_rank()
    pp = mpu.get_pipeline_model_parallel_rank()
    vpp = mpu.get_virtual_pipeline_model_parallel_rank()
    ep = mpu.get_expert_model_parallel_rank()
    tp = mpu.get_tensor_model_parallel_rank()
    if ep == 0 and tp == 0:
        for model_chunk_index, model_chunk in enumerate(model):
            decoder = model_chunk.module.module.decoder
            for layer_index, layer in enumerate(decoder.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                    num_local_experts = layer.mlp.experts.num_local_experts
                    expert_load_per_device = layer.mlp.expert_load_prediction.reshape(-1, num_local_experts).sum(-1)
                    min_load = expert_load_per_device.min().item()
                    max_load = expert_load_per_device.max().item()
                    std_load = expert_load_per_device.std().item()
                    expert_load = expert_load_per_device.int().tolist()
                    if not getattr(args, "enable_expert_placement", False):
                        print(
                            f"[expert load statistics] iter {iteration}: "
                            f"rank {torch.distributed.get_rank()} dp{dp} pp{pp}, vpp{vpp}, tp{tp}, ep{ep}, "
                            f"model chunk {model_chunk_index}, layer {layer_index}, "
                            f"device expert load: min {min_load:.2f}, "
                            f"max {max_load:.2f}, std {std_load:.2f}, "
                            f"details {expert_load}\n"
                        )

                    else:
                        sorted_expert_mapping, sorted_expert_mapping_indices = torch.sort(layer.mlp.expert_mapping)
                        expert_load_per_device_after_placement = layer.mlp.expert_load_prediction[
                            sorted_expert_mapping_indices].reshape(-1, num_local_experts).sum(-1)
                        min_load_after_placement = expert_load_per_device_after_placement.min().item()
                        max_load_after_placement = expert_load_per_device_after_placement.max().item()
                        std_load_after_placement = expert_load_per_device_after_placement.std().item()
                        expert_load_after_placement = expert_load_per_device_after_placement.int().tolist()
                        print(
                            f"[expert load statistics before expert placement] iter {iteration}: "
                            f"rank {torch.distributed.get_rank()} dp{dp} pp{pp}, vpp{vpp}, tp{tp}, ep{ep}, "
                            f"model chunk {model_chunk_index}, layer {layer_index}, "
                            f"device expert load: min {min_load:.2f}, "
                            f"max {max_load:.2f}, std {std_load:.2f}, "
                            f"details {expert_load}\n"
                        )
                        print(
                            f"[expert load statistics after expert placement] iter {iteration}: "
                            f"rank {torch.distributed.get_rank()} dp{dp} pp{pp}, vpp{vpp}, tp{tp}, ep{ep}, "
                            f"model chunk {model_chunk_index}, layer {layer_index}, "
                            f"device expert load: min {min_load_after_placement:.2f}, "
                            f"max {max_load_after_placement:.2f}, std {std_load_after_placement:.2f}, "
                            f"details {expert_load_after_placement}\n"
                        )

 
class ExpertDynamicplacement:
    @staticmethod
    def expert_placement_greedy(experts, devices):
        num_experts = len(experts)
        num_devices = len(devices)
        expert_keys = list(experts.keys())
        # Initialize the current number of samples per device
        samples_each_device = dict.fromkeys(devices, 0)
        # Initialize the current number of experts per device
        experts_each_device = dict.fromkeys(devices, 0)
        # Initialize the placement of experts
        P = {key: [] for key in expert_keys}
        
        # Sort the experts in descending order of their costs and get the sorted indices
        sorted_indices = sorted(expert_keys, key=lambda i: experts[i], reverse=True)
 
        for i in sorted_indices:
            Ci = experts[i]
            # Initialize the minimum number of samples to infinity
            Tmin = float('inf')
            q = -1
            # Decide the placement of the current expert
            for device in devices:
                if experts_each_device[device] < num_experts / num_devices and samples_each_device[device] < Tmin:
                    Tmin = samples_each_device[device]
                    q = device
            # Place the current expert on the selected device
            P[i].append(q)
            # Update the number of samples on the selected device
            samples_each_device[q] += Ci
            # Update the number of experts on the selected device
            experts_each_device[q] += 1
        
        Q = {key: [] for key in devices}
        P_duplicate = copy.deepcopy(P)
        for i in expert_keys:
            device = P_duplicate[i].pop()
            Q[device].append(i)
 
        return P, Q
    
    @staticmethod
    def expert_placement_dp(experts, devices, expert_per_device=0):
        # Generate all possible subsets of experts
        num_experts = len(experts)
        num_devices = len(devices)
        expert_keys = list(experts.keys())
        subsets = []
        for r in range(num_experts + 1):
            subsets.extend(itertools.combinations(experts, r))
        subsets = [set(s) for s in subsets]
 
        feasible_subsets = []
        if expert_per_device:
            feasible_subsets.extend(itertools.combinations(experts, expert_per_device))
        else:
            for r in range(num_experts + 1):
                feasible_subsets.extend(itertools.combinations(experts, r))
        feasible_subsets = [set(s) for s in feasible_subsets]
 
        # Initialize the dp table
        dp = {}
        placement = {}
 
        # Base case: when using 0 devices, the cost is 0
        for s in subsets:
            dp[(0, frozenset(s))] = 0
 
        # Fill the dp table
        for i in range(1, num_devices + 1):
            for s in subsets:
                min_val = float('inf')
                best_s0 = None
                if i == 1:
                    remaining_experts = s
                    remaining_cost = sum(experts[e] for e in remaining_experts)
                    dp[(i, frozenset(s))] = remaining_cost
                    placement[(i, frozenset(s))] = set()
                    continue
                for s0 in subsets:
                    if s0.issubset(s):
                        remaining_experts = s - s0
                        if remaining_experts not in feasible_subsets:
                            continue
                        dp_key = (i - 1, frozenset(s0))
                        if dp_key not in dp:
                            raise KeyError(
                                f"DP key not found: (devices={i-1}, experts={s0}). "
                                f"Ensure base cases are properly initialized."
                            )
                        remaining_cost = sum(experts[e] for e in remaining_experts) if remaining_experts else 0
                        current_val = max(dp[dp_key], remaining_cost)
                        if current_val < min_val:
                            min_val = current_val
                            best_s0 = s0
                dp[(i, frozenset(s))] = min_val
                placement[(i, frozenset(s))] = best_s0
                
        # Backtracking to find the placement
        P = {key: [] for key in expert_keys}
        current_subset = set(experts)
        current_devices = devices.copy()
 
        for i in range(num_devices, 0, -1):
            placement_key = (i, frozenset(current_subset))
            if placement_key not in placement:
                raise KeyError(
                    f"Placement key not found: (device={i}, experts={current_subset}). "
                    f"Valid keys: {list(placement.keys())}"
                )

            s0 = placement[placement_key]
            remaining_experts = current_subset - s0
            current_device = current_devices.pop()
            for e in remaining_experts:
                P[e].append(current_device)
            current_subset = s0
        
        Q = {key: [] for key in devices}
        P_duplicate = copy.deepcopy(P)
        for i in expert_keys:
            device = P_duplicate[i].pop()
            Q[device].append(i)
 
        return P, Q
    
    @staticmethod
    def expert_placement_mix(experts, devices, num_virtual_devices):
        num_devices = len(devices)
 
        num_devices_within = int(num_devices / num_virtual_devices)
        virtual_devices = list(range(num_virtual_devices))
 
        # Call the expert_placement_greedy method to perform expert placement
        _, greedy_device_result = ExpertDynamicplacement.expert_placement_greedy(experts, virtual_devices)
        P = {}
        Q = {}
        for virtual_device in greedy_device_result:
            # collecting experts and cost in the each virtual device
            experts_local = {}
            for expert in experts:
                if expert in greedy_device_result[virtual_device]:
                    experts_local[expert] = experts[expert]
            # placing local experts to devices in each virtual device
            devices_local = devices[virtual_device * num_devices_within:(virtual_device + 1) * num_devices_within]
            expert_per_device = int(len(experts_local) / num_devices_within)
            dp_expert_result, dp_device_result = ExpertDynamicplacement.expert_placement_dp(
                                                                        experts_local, devices_local, expert_per_device)
            
            P.update(dp_expert_result)
            Q.update(dp_device_result)
        return P, Q
