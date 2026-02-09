import os
import sys
import threading
import subprocess
import re
from itertools import combinations
from logging import getLogger

from torch.fx.experimental.migrate_gradual_types.constraint_transformation import generate_disj

from megatron.training import get_args
from mindspeed.core.multi_modal.dist_train.dist_parallel_state import get_tensor_model_parallel_world_size
from mindspeed.log_config import log_rank_0
from mindspeed.core.qos.domain_info import domains, generate_masked_orthogonal_rank_groups, \
    get_tensor_parallel_comm_domain, get_pipeline_parallel_comm_domain, get_data_parallel_comm_domain, \
    get_context_parallel_comm_domain, get_expert_parallel_comm_domain, get_overlap_time_dict, get_overlap_space_dict, \
    is_cross_boundary

LOG = getLogger()

_DEFAULT_QOS = 4
_DEFAULT_QOS_LOW = os.environ.get('QOS_LOW', 2)
_DEFAULT_QOS_MIDDLE = os.environ.get('QOS_MIDDLE', 4)
_DEFAULT_QOS_HIGH = os.environ.get('QOS_HIGH', 6)

qos_str_to_value = {
    'low': _DEFAULT_QOS_LOW,
    'middle': _DEFAULT_QOS_MIDDLE,
    'high': _DEFAULT_QOS_HIGH
}

_PARALLEL_TYPES = [
    'dp',
    'dp-cp',
    'intra-dp-cp',
    'inter-dp-cp',
    'cp',
    'mp'
    'tp',
    'pp',
    'embd',
    'pos-embd',
    'tp-dp-cp',
    'tp-dp',
    'tp-cp',
    'ep',
    'ep-tp',
    'tp-ep-mp'
    'tp-ep-pp',
    'ep-dp',
    'hcp'
]

domains = ('tp', 'dp', 'pp', 'ep', 'cp')


def mpamqos():
    if is_a3_version:
        import aiQos
        aiQos.init()
        card_id_list = [0, 1, 2, 3, 4, 5, 6, 7]
        device_id_list = [0, 1]
        for card_id in card_id_list:
            for device_id in device_id_list:
                aiQos.set_gbl_qos(card_id=card_id, device_id=device_id, mode=1)
                aiQos.set_bw(target=0, bw_low=10, bw_high=50, hardlimit=0, card_id=card_id,
                             device_id=device_id)


class Qos:
    _instance = None
    _lock = threading.Lock()
    _initialize = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                cls._instance = super(Qos, cls).__new__(cls)
        return cls._instance

    def __init__(self, queue_list=None):
        if Qos._initialize:
            return
        if queue_list is None:
            self.queue_list = [_DEFAULT_QOS_LOW, _DEFAULT_QOS_MIDDLE, _DEFAULT_QOS_HIGH]
        else:
            self.queue_list = queue_list

        self.args = get_args()
        self.aiqos_mode = self.args.aiqos_mode if hasattr(self.args,
                                                          'aiqos_mode') and self.args.aiqos_mode is not None else "auto"
        if self.aiqos_mode.lower() not in ['auto', 'manual']:
            raise ValueError('aiqos mode must be "auto or manual"')
        self.aiqos_schedule = {}
        self.init_qos()
        Qos._initialized = True

    def init_qos(self):
        def parse_args(arg_str, target_dict):
            # Check if the input argument string is empty or None
            if arg_str is None or arg_str.strip() == "":
                raise ValueError("aiqos-schedule parameter cannot be empty")

            # Remove leading and trailing whitespace characters
            clean_str = arg_str.strip()

            # Define valid string values and update regex pattern to match them (case-insensitive)
            valid_values = r'(high|low|middle)'
            # Regex pattern: matches {key:value, key:value,...} format with valid string values
            pattern = fr'^\{{\s*([a-zA-Z0-9_-]+:\s*{valid_values}\s*(,\s*[a-zA-Z0-9_-]+:\s*{valid_values}\s*)*)?\}}$'

            # Validate the overall format with case-insensitive matching
            if not re.match(pattern, clean_str, re.IGNORECASE):
                raise ValueError(
                    f"Invalid aiqos-schedule format. Only {{key:value,key:value,...}} is supported (value must be high/low/middle). Current input: {arg_str}\n"
                    "Correct example: {tp:high,pp:low,dp:middle}"
                )

            # Remove the leading '{' and trailing '}' from the string
            clean_str = clean_str.lstrip('{').rstrip('}')

            # Split the string by comma, remove empty items and strip whitespace for each item
            items = [item.strip() for item in clean_str.split(',') if item.strip()]

            # Define the set of allowed valid values (lowercase for unified validation)
            allowed_values = {'high', 'low', 'middle'}

            # Iterate through each key-value pair for further validation and processing
            for item in items:
                # Ensure each item has exactly one colon (standard key:value format)
                if item.count(':') != 1:
                    raise ValueError(f"Invalid key-value pair format. Must be key:value. Current: {item}")

                # Split the item into key (parallel_type) and value, split only at the first colon
                parallel_type, value = item.split(':', 1)

                # Convert key to lowercase and strip whitespace for unified storage
                parallel_type = parallel_type.strip().lower()
                # Convert value to lowercase and strip whitespace for unified validation
                value_str = value.strip().lower()

                # Validate if the value is in the allowed set
                if value_str not in allowed_values:
                    raise ValueError(
                        f"Invalid value for {parallel_type}. Only 'high', 'low', 'middle' are allowed. Current: {value_str}"
                    )

                # Store the valid key-value pair into the target dictionary
                target_dict[parallel_type] = value_str

        if self.aiqos_mode.lower() == 'manual':
            parse_args(self.args.aiqos_schedule, self.aiqos_schedule)
            for key, priority_str in self.aiqos_schedule.items():
                priority_str_lower = priority_str.strip().lower()
                if priority_str_lower not in qos_str_to_value:
                    raise ValueError(
                        f"Invalid QoS priority string: {priority_str}, only 'high'/'low'/'middle' are allowed")
                self.aiqos_schedule[key] = qos_str_to_value[priority_str_lower]
        elif self.aiqos_mode.lower() == 'auto':
            self.cal_auto_qos()
            self.init_domain_qos_schedule_rules()

        log_rank_0(LOG.info, f'qos schedule: {self.aiqos_schedule}')

    def set_parallel_qos(self, parallel_type):
        if parallel_type is None:
            return _DEFAULT_QOS
        if parallel_type.lower() not in _PARALLEL_TYPES or parallel_type.lower() not in self.aiqos_schedule:
            return _DEFAULT_QOS
        return self.aiqos_schedule[parallel_type.lower()]

    def cal_auto_qos(self):
        parallel_comm_domain_list = [get_tensor_parallel_comm_domain(), get_data_parallel_comm_domain(),
                                     get_pipeline_parallel_comm_domain(), get_expert_parallel_comm_domain(),
                                     get_context_parallel_comm_domain(),
                                     ]
        domain_partition_information = {
            key: domain.rank_list
            for key, domain in zip(domains, parallel_comm_domain_list)
        }
        qos_res = self.combination(parallel_comm_domain_list, domain_partition_information)
        for parallel_type, qos in qos_res.items():
            self.aiqos_schedule[parallel_type] = qos
        return

    def combination(self, parallel_comm_domain_list=None, domain_partition_information=None):
        if parallel_comm_domain_list is None or domain_partition_information is None:
            raise ValueError("parallel_comm_domain_list  or domain_partition_information is None")
        domain_nums = len(parallel_comm_domain_list)
        queue_nums = len(self.queue_list)
        time_overlap = get_overlap_time_dict()
        space_overlap = get_overlap_space_dict(domain_partition_information)
        comb = generate_distributions(domain_nums, queue_nums)
        min_single_comb = comb[0]
        degree = sys.maxsize
        for each_comb in comb:
            cur_degree = cal_conflict_degree(each_comb, parallel_comm_domain_list, time_overlap, space_overlap)
            if cur_degree < degree:
                degree = cur_degree
                min_single_comb = each_comb
        min_single_comb_log_info = [[domains[idx] for idx in num_list] for num_list in min_single_comb]
        log_rank_0(LOG.info, f'min_single_comb: {min_single_comb_log_info}')
        return self.auto_qos_priority(min_single_comb, parallel_comm_domain_list)

    def auto_qos_priority(self, min_single_comb, parallel_comm_domain_list):
        rate = [0] * len(min_single_comb)
        for each_queue in min_single_comb:
            sum_comm_amount = 0
            sum_comm_amount_no_overlap = 0
            for each_flow in each_queue:
                sum_comm_amount += parallel_comm_domain_list[each_flow].comm_amount
                sum_comm_amount_no_overlap += parallel_comm_domain_list[each_flow].comm_amount_no_overlap
            if sum_comm_amount == 0:
                continue
            else:
                cur_rate = sum_comm_amount_no_overlap / sum_comm_amount
            rate[min_single_comb.index(each_queue)] = cur_rate
        sorted_rate = dict(sorted(zip(rate, min_single_comb), key=lambda x: x[0], reverse=True))
        qos_res = {}
        queue = self.queue_list[0]
        for value in sorted_rate.values():
            for flow in value:
                qos_res[domains[flow]] = queue
            queue += 1
        return qos_res

    def init_domain_qos_schedule_rules(self):
        if self.args.num_experts is None:
            self.aiqos_schedule['dp-cp'] = self.aiqos_schedule['dp']
            self.aiqos_schedule['mp'] = self.aiqos_schedule['pp']
        else:
            self.aiqos_schedule['dp-cp'] = self.aiqos_schedule['dp']
            self.aiqos_schedule['tp-ep-mp'] = self.aiqos_schedule['tp']
            self.aiqos_schedule['ep-dp'] = self.aiqos_schedule['dp']
            self.aiqos_schedule['mp'] = self.aiqos_schedule['pp']
            self.aiqos_schedule['tp-ep-pp'] = self.aiqos_schedule['pp']
            self.aiqos_schedule['tp-ep-mp'] = self.aiqos_schedule['pp']


def generate_distributions(m, n):
    if m < 0 or n < 0:
        return []

    results = []
    groups = []

    def dfs(idx):
        if idx == m:
            results.append([group[:] for group in groups])
            return

        for group in groups:
            if group and group[-1] < idx:
                group.append(idx)
                dfs(idx + 1)
                group.pop()

        if len(groups) < n:
            groups.append([idx])
            dfs(idx + 1)
            groups.pop()

    dfs(0)
    return results


def cal_conflict_degree(single_comb_info, parallel_comm_domain_list, time_overlap, space_overlap):
    degree = 0
    for each_queue in single_comb_info:
        for i, elem_i in enumerate(each_queue):
            for j, elem_j in enumerate(each_queue[i + 1:], start=i + 1):
                key = (domains[elem_i], domains[elem_j])
                conflict_state = time_overlap[key] * space_overlap[key]
                if conflict_state == 1:
                    degree += min(
                        parallel_comm_domain_list[i].comm_amount,
                        parallel_comm_domain_list[j].comm_amount
                    )
    return degree
