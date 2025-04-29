_COC_CFGS = {
    'recompute_all_gather': True,
    'matmul_soc_friendly': True,
    'print_tensor_value_open': False,
    'customized_coc': {},
    'enable_coc_in_column_backward': False,
    'k_min': 1024,
    'k_max': 4096,
}


def get_value_from_cfg(attr_name):
    return _COC_CFGS[attr_name]
