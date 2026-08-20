from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import mpu


def build_pretraining_data_loader_decorator(build_pretraining_data_loader):
    @wraps(build_pretraining_data_loader)
    def wrapper(*args, **kwargs):
        if args[0] is None:
            return None
        argument = get_args()
        if argument.dataloader_type == 'single' and argument.automated_pipeline_perf and argument.optimized_mbs_list:
            batch_sampler = DynamicMicroBatchPretrainingSampler(
                total_samples=len(args[0]),
                consumed_samples=args[1],
                micro_batch_size=argument.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
            return torch.utils.data.DataLoader(
                args[0], batch_sampler=batch_sampler, num_workers=argument.num_workers, pin_memory=True
            )
        else:
            dataloader = build_pretraining_data_loader(*args, **kwargs)
            return dataloader

    return wrapper


class DynamicMicroBatchPretrainingSampler:
    def __init__(
        self, total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, drop_last=True
    ):
        args = get_args()
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.drop_last = drop_last
        self.dynamic_micro_batch_size = args.optimized_mbs_list
        self.micro_batch_times_data_parallel_size = [
            self.dynamic_micro_batch_size[i] * data_parallel_size for i in range(len(self.dynamic_micro_batch_size))
        ]

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self, n_mbs):
        start_idx = self.data_parallel_rank * self.dynamic_micro_batch_size[n_mbs]
        end_idx = start_idx + self.dynamic_micro_batch_size[n_mbs]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        n_mbs = 0
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size[n_mbs]:
                start_idx, end_idx = self.get_start_end_idx(n_mbs)
                yield batch[start_idx:end_idx]
                batch = []
                n_mbs = (n_mbs + 1) % len(self.micro_batch_times_data_parallel_size)

        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx(n_mbs)
            yield batch[start_idx:end_idx]
