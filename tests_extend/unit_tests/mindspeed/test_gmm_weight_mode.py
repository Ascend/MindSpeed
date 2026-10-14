# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""CPU configuration contracts; no NPU numerical coverage."""

import ast
import importlib.util
import sys
from copy import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def gmm_env(monkeypatch):
    spec = importlib.util.find_spec("mindspeed")
    root = (
        Path(spec.origin).resolve().parent
        if spec and spec.origin else Path(__file__).resolve().parents[3] / "mindspeed"
    )
    feature = ModuleType("mindspeed.features_manager.feature")
    feature.MindSpeedFeature = object
    args_utils = ModuleType("mindspeed.args_utils")
    args = SimpleNamespace(te_gmm_mode="compatible")
    args_utils.get_full_args = lambda: args
    monkeypatch.setitem(sys.modules, feature.__name__, feature)
    monkeypatch.setitem(sys.modules, args_utils.__name__, args_utils)
    spec = importlib.util.spec_from_file_location("_gmm_weight_mode_test", root / "features_manager/moe/gmm.py")
    gmm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gmm)

    # Load the real nested FBO expert builder without importing NPU modules.
    source = root / "core/transformer/moe/moe_feature/fb_overlap/adaptor.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    builder = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "fb_overlap_experts_builder")

    def build_fbo(experts_builder, config):
        namespace = {"copy": copy, "MindSpeedFbOverlapGmmExperts": experts_builder,
                     "grouped_mlp_submodules": object()}
        exec(compile(ast.Module(body=[builder], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["fb_overlap_experts_builder"](2, config)

    return SimpleNamespace(gmm=gmm, args=args, build_fbo=build_fbo)


@pytest.mark.parametrize("mode", ["compatible", "performance"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fbo", [False, True])
@pytest.mark.parametrize("mcore_forwards_layout", [False, True], ids=["te213", "te214"])
@pytest.mark.parametrize("parallel_mode", ["column", "row"])
def test_expert_paths_select_layout_independently_of_fusion(
    gmm_env, mode, fused, fbo, mcore_forwards_layout, parallel_mode
):
    env = gmm_env
    env.args.te_gmm_mode = mode
    packed = mode == "performance"
    config = SimpleNamespace(
        gemm_gradient_accumulation_fusion=fused,
        gradient_accumulation_fusion=not fused,
        moe_single_grouped_weight=not packed,
        delay_wgrad_compute=False,
        use_transformer_engine_op_fuser=True,
    )
    original_config = vars(config).copy()

    def tenpu_init(self, *, fuse_wgrad_accumulation=False, single_grouped_weight=False,
                   delay_wgrad_compute=False):
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.single_grouped_weight = single_grouped_weight
        self.delay_wgrad_compute = delay_wgrad_compute
        if single_grouped_weight:
            self.weight = SimpleNamespace()
        else:
            dim = 0 if parallel_mode == "column" else 1
            self.weight0 = SimpleNamespace(partition_dim=dim, partition_stride=1)

    tenpu_init = env.gmm.tenpu_grouped_linear_init_wrapper(tenpu_init)

    def mcore_init(self, config, parallel_mode):
        kwargs = dict(fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                      delay_wgrad_compute=config.delay_wgrad_compute)
        if mcore_forwards_layout:
            kwargs["single_grouped_weight"] = config.moe_single_grouped_weight
        tenpu_init(self, **kwargs)

    mcore_init = env.gmm.te_grouped_linear_init_wrapper(mcore_init)

    def experts_builder(num_experts, expert_config, **kwargs):
        assert num_experts == 2
        if fbo:
            assert expert_config.use_transformer_engine_op_fuser is False
        layer = SimpleNamespace()
        mcore_init(layer, expert_config, parallel_mode)
        return layer, expert_config

    if fbo:
        layer, expert_config = env.build_fbo(experts_builder, config)
    else:
        layer, expert_config = env.gmm._build_te_grouped_mlp(experts_builder, 2, config)
    assert expert_config is not config
    assert vars(config) == original_config
    assert expert_config.moe_single_grouped_weight is packed
    assert layer.single_grouped_weight is packed
    assert layer.fuse_wgrad_accumulation is fused
    assert layer.delay_wgrad_compute is fbo
    if packed:
        assert layer.weight.partition_dim == (1 if parallel_mode == "column" else 2)
        assert layer.weight.partition_stride == 1
    else:
        assert not hasattr(layer, "weight")
        assert layer.weight0.partition_dim == (0 if parallel_mode == "column" else 1)


@pytest.mark.parametrize("mode", ["compatible", "performance"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("explicit_layout", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_tenpu_preserves_explicit_layout(gmm_env, mode, fused, explicit_layout, positional):
    gmm_env.args.te_gmm_mode = mode

    def init(self, fuse_wgrad_accumulation=False, single_grouped_weight=False):
        return fuse_wgrad_accumulation, single_grouped_weight

    wrapped = gmm_env.gmm.tenpu_grouped_linear_init_wrapper(init)
    if positional:
        result = wrapped(object(), fused, explicit_layout)
    else:
        result = wrapped(object(), fuse_wgrad_accumulation=fused, single_grouped_weight=explicit_layout)
    assert result == (fused, explicit_layout)


def test_missing_mode_defaults_to_compatible_even_with_fusion(gmm_env):
    del gmm_env.args.te_gmm_mode

    def init(self, fuse_wgrad_accumulation=False, single_grouped_weight=False):
        return fuse_wgrad_accumulation, single_grouped_weight

    wrapped = gmm_env.gmm.tenpu_grouped_linear_init_wrapper(init)
    assert wrapped(object(), fuse_wgrad_accumulation=True) == (True, False)
