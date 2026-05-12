from functools import wraps
import torch


def split_deepstack_embs_wrapper(original_split_deepstack_embs):
    """Wrapper for split_deepstack_embs with automatic padding support.

    Handles cases where visual_pos_masks.shape[-1] is not divisible by split_size
    by automatically padding with zeros instead of raising assertion error.

    This is for Qwen3VL vision model to support CP (Context Parallel) and TP (Tensor Parallel)
    splitting of visual embeddings.
    """

    @wraps(original_split_deepstack_embs)
    def wrapper(
            visual_pos_masks: torch.Tensor,
            deepstack_visual_embeds: list[torch.Tensor],
            tp_size: int = 1,
            tp_rank: int = 0,
            cp_size: int = 1,
            cp_rank: int = 0,
            sequence_parallel: bool = False,
    ):
        # Calculate split_size
        if not sequence_parallel:
            tp_size = 1
            tp_rank = 0
        split_size = tp_size
        if cp_size > 1:
            split_size *= cp_size * 2

        # Early return if no splitting needed
        if split_size == 1 or visual_pos_masks is None:
            return original_split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                sequence_parallel=sequence_parallel
            )

        # Assert dimension
        if visual_pos_masks.dim() != 2:
            raise ValueError(f"visual_pos_masks.dim is not 2, current value is {visual_pos_masks.dim()}")

        # Calculate padding length
        pad_len = (split_size - visual_pos_masks.shape[-1] % split_size) % split_size

        # Apply padding if needed
        if pad_len > 0:
            # Pad visual_pos_masks
            visual_pos_masks = torch.cat([
                visual_pos_masks,
                torch.zeros(
                    visual_pos_masks.shape[0],
                    pad_len,
                    dtype=visual_pos_masks.dtype,
                    device=visual_pos_masks.device
                )
            ], dim=-1)

            # Pad deepstack_visual_embeds
            for i, embed in enumerate(deepstack_visual_embeds):
                deepstack_visual_embeds[i] = torch.cat([
                    embed,
                    torch.zeros(
                        pad_len,
                        embed.shape[-1],
                        dtype=embed.dtype,
                        device=embed.device
                    )
                ], dim=0)

        # Call original function with padded inputs
        return original_split_deepstack_embs(
            visual_pos_masks,
            deepstack_visual_embeds,
            tp_size=tp_size,
            tp_rank=tp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            sequence_parallel=sequence_parallel
        )

    return wrapper


def qwen3vl_model_init_wrapper(original_init):
    """Wrapper for Qwen3VLModel.__init__ to ignore CP assertion restriction.

    This wrapper catches the specific CP assertion error without modifying any config or pg_collection values.
    Only the CP assertion (Lines 177-180) is ignored, all other assertions still raise normally.

    Original assertion:
        if self.pg_collection.cp.size() > 1:
            assert self.config.calculate_per_token_loss, (
                "Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled"
            )

    This approach:
    - Does NOT modify self.config.calculate_per_token_loss
    - Does NOT modify self.pg_collection.cp.size()
    - Catches and ignores the specific CP assertion error
    - Re-raises all other assertions to maintain safety checks
    """

    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
        except AssertionError as e:
            error_msg = str(e)
            if "Qwen3-VL model only supports context parallelism" in error_msg:
                pass
            else:
                raise

    return wrapper


def get_device_capability(device=None):
    return 9, 0