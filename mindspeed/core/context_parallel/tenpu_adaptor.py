"""MindSpeed-to-TENPU runtime options for context-parallel attention."""

from functools import wraps


def set_tenpu_cp_runtime_options(attention, args):
    """Attach MindSpeed-only CP options to a TENPU attention instance.

    Megatron forwards the public CP process-group arguments to TENPU, but the
    double-ring window and send/recv-overlap options are MindSpeed extensions.
    TENPU's ring preparation path reads these attributes from the attention
    instance immediately before the first forward pass.
    """
    attention.cp_window_size = int(getattr(args, 'cp_window_size', 1))
    attention.use_cp_send_recv_overlap = bool(getattr(args, 'use_cp_send_recv_overlap', False))

    # ``a2a+p2p`` applies the window to its p2p subgroup.  Preserve both the
    # public Megatron hierarchy and the legacy spelling so TENPU can derive
    # that subgroup without consulting MindSpeed global arguments.
    hierarchy = getattr(args, 'hierarchical_context_parallel_sizes', None)
    attention.hierarchical_context_parallel_sizes = hierarchy
    attention.ulysses_degree_in_cp = (
        hierarchy[0]
        if hierarchy and len(hierarchy) == 2
        else getattr(args, 'ulysses_degree_in_cp', None)
    )

def te_dot_product_attention_init_wrapper(init_func):
    """Publish MindSpeed CP options after Megatron creates TENPU attention."""

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        init_func(self, *args, **kwargs)

        # Import lazily: this module is registered before Megatron finishes
        # importing, while the global argument namespace is available by the
        # time a model attention module is constructed.
        from megatron.training import get_args

        set_tenpu_cp_runtime_options(self, get_args())

    return wrapper
