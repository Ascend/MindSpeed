import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


TEST_TREE_ROOT = Path(__file__).resolve().parents[3]


def _installed_package_roots(package_name):
    roots = []
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        roots.extend(
            Path(path).resolve().parent
            for path in spec.submodule_search_locations or ()
        )
        if spec.origin is not None:
            roots.append(Path(spec.origin).resolve().parent.parent)
    return roots


def _mindspeed_root():
    candidates = _installed_package_roots("mindspeed")
    candidates.extend(
        (
            TEST_TREE_ROOT,
            TEST_TREE_ROOT.parent / "MindSpeed",
            Path.cwd().resolve(),
            Path.cwd().resolve().parent / "MindSpeed",
        )
    )
    for candidate in candidates:
        if (candidate / "mindspeed" / "features_manager" / "feature.py").is_file():
            return candidate
    raise AssertionError(
        "MindSpeed sources are required; add the MindSpeed checkout to PYTHONPATH"
    )


MINDSPEED_ROOT = _mindspeed_root()
FEATURE_PATH = MINDSPEED_ROOT / "mindspeed" / "features_manager" / "feature.py"
MANAGER_PATH = MINDSPEED_ROOT / "mindspeed" / "features_manager" / "features_manager.py"
PATCH_MANAGER_PATH = MINDSPEED_ROOT / "mindspeed" / "patch_utils.py"
MOE_TP_EXTEND_EP_PATH = (
    MINDSPEED_ROOT / "mindspeed" / "features_manager" / "moe" / "tp_extend_ep.py"
)


def _package(name):
    package = ModuleType(name)
    package.__path__ = []
    return package


def _load_module(module_name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _megatron_adaptor_root():
    candidates = _installed_package_roots("megatron_adaptor")
    candidates.extend(
        (
            MINDSPEED_ROOT.parent / "MegatronAdaptor-dev",
            MINDSPEED_ROOT.parent / "MA" / "MegatronAdaptor",
            TEST_TREE_ROOT.parent / "MegatronAdaptor-dev",
            TEST_TREE_ROOT.parent / "MA" / "MegatronAdaptor",
        )
    )
    for candidate in candidates:
        if (
            candidate / "megatron_adaptor" / "features_manager" / "features_manager.py"
        ).is_file():
            return candidate
    raise AssertionError("A MegatronAdaptor checkout or installation is required")


def test_third_party_feature_and_manager_patch_without_affecting_ma(monkeypatch):
    """Mirror MindSpeed-LLM concrete-feature replacement against the real MA framework."""
    ma_root = _megatron_adaptor_root()
    ma_package_root = ma_root / "megatron_adaptor"
    modules = {
        "megatron_adaptor": _package("megatron_adaptor"),
        "megatron_adaptor.features_manager": _package(
            "megatron_adaptor.features_manager"
        ),
        "megatron_adaptor.patches": _package("megatron_adaptor.patches"),
        "mindspeed": _package("mindspeed"),
        "mindspeed.features_manager": _package("mindspeed.features_manager"),
        "mindspeed.features_manager.moe": _package("mindspeed.features_manager.moe"),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    ma_patch_module = _load_module(
        "megatron_adaptor.patches.patch_manager",
        ma_package_root / "patches" / "patch_manager.py",
        monkeypatch,
    )
    ma_feature_module = _load_module(
        "megatron_adaptor.features_manager.mindspeed_feature",
        ma_package_root / "features_manager" / "mindspeed_feature.py",
        monkeypatch,
    )
    ma_manager_module = _load_module(
        "megatron_adaptor.features_manager.features_manager",
        ma_package_root / "features_manager" / "features_manager.py",
        monkeypatch,
    )
    mindspeed_patch_module = _load_module(
        "mindspeed.patch_utils", PATCH_MANAGER_PATH, monkeypatch
    )
    _load_module("mindspeed.features_manager.feature", FEATURE_PATH, monkeypatch)
    mindspeed_manager_module = _load_module(
        "mindspeed.features_manager.features_manager", MANAGER_PATH, monkeypatch
    )
    mindspeed_moe_module = _load_module(
        "mindspeed.features_manager.moe.tp_extend_ep",
        MOE_TP_EXTEND_EP_PATH,
        monkeypatch,
    )

    MAFeature = ma_feature_module.MindSpeedFeature
    MAFeaturesManager = ma_manager_module.FeaturesManager
    MAPatchesManager = ma_patch_module.PatchesManager
    MindSpeedFeaturesManager = mindspeed_manager_module.MindSpeedFeaturesManager
    MindSpeedPatchesManager = mindspeed_patch_module.MindSpeedPatchesManager
    MindSpeedMoETpExtendEpFeature = mindspeed_moe_module.MoETpExtendEpFeature

    target_module_name = "_third_party_contract_patch_targets"
    target_module = ModuleType(target_module_name)
    target_module.MoELayer = lambda: "unpatched"
    target_module.MAOnlyBehavior = lambda: "ma_unpatched"
    monkeypatch.setitem(sys.modules, target_module_name, target_module)
    mindspeed_target = f"{target_module_name}.MoELayer"
    ma_target = f"{target_module_name}.MAOnlyBehavior"
    ma_register_calls = []
    mindspeed_register_calls = []
    mindspeed_original_calls = []
    third_party_calls = []

    ma_register_patch = MAPatchesManager.register_patch
    mindspeed_register_patch = MindSpeedPatchesManager.register_patch

    def register_ma_patch(
        target, replacement=None, force_patch=False, create_dummy=False
    ):
        assert force_patch is False
        ma_register_calls.append((target, force_patch))
        return ma_register_patch(target, replacement, force_patch, create_dummy)

    def register_mindspeed_patch(
        target, replacement=None, force_patch=False, create_dummy=False
    ):
        assert force_patch is False
        mindspeed_register_calls.append((target, force_patch))
        return mindspeed_register_patch(target, replacement, force_patch, create_dummy)

    MAPatchesManager.register_patch = staticmethod(register_ma_patch)
    MindSpeedPatchesManager.register_patch = staticmethod(register_mindspeed_patch)

    class MAOnlyFeature(MAFeature):
        def __init__(self):
            super().__init__("ma-only", 2)

        def register_patches(self, patch_manager, args):
            patch_manager.register_patch(ma_target, lambda: "ma")
            patch_manager.register_patch(mindspeed_target, lambda: "ma_base")

    def track_original_mindspeed_register_patches(self, patch_manager, args):
        mindspeed_original_calls.append((self, patch_manager, args))
        raise AssertionError("the replaced MindSpeed register_patches must not run")

    MindSpeedMoETpExtendEpFeature.register_patches = (
        track_original_mindspeed_register_patches
    )

    class ThirdPartyMoETpExtendEpFeature(MindSpeedMoETpExtendEpFeature):
        def register_patches(self, patch_manager, args):
            third_party_calls.append(("third_party", patch_manager))
            if (
                args.moe_token_dispatcher_type == "alltoall_seq"
                and args.moe_tp_extend_ep
                and not args.moe_alltoall_overlap_comm
            ):
                patch_manager.register_patch(mindspeed_target, lambda: "third_party")

    class ThirdPartyFeaturesManager(MindSpeedFeaturesManager):
        FEATURES_LIST = []

    args = SimpleNamespace(
        optimization_level=2,
        ma_only=True,
        moe_tp_extend_ep=True,
        moe_token_dispatcher_type="alltoall_seq",
        moe_alltoall_overlap_comm=False,
    )
    ma_feature = MAOnlyFeature()
    third_party_feature = ThirdPartyMoETpExtendEpFeature()

    assert issubclass(ThirdPartyMoETpExtendEpFeature, MindSpeedMoETpExtendEpFeature)
    assert issubclass(ThirdPartyMoETpExtendEpFeature, MAFeature)
    assert issubclass(ThirdPartyFeaturesManager, MindSpeedFeaturesManager)
    assert issubclass(ThirdPartyFeaturesManager, MAFeaturesManager)
    assert MindSpeedPatchesManager is not MAPatchesManager
    assert MindSpeedPatchesManager.patches_info is not MAPatchesManager.patches_info
    assert third_party_feature.feature_name == "moe_tp_extend_ep"
    assert target_module.MoELayer() == "unpatched"
    assert target_module.MAOnlyBehavior() == "ma_unpatched"

    MAFeaturesManager.set_features_list([ma_feature])
    MAFeaturesManager.apply_features_patches(args)
    ma_features_before = tuple(MAFeaturesManager.FEATURES_LIST)
    ma_patches_before = dict(MAPatchesManager.patches_info)
    ma_register_calls_before = list(ma_register_calls)

    assert target_module.MAOnlyBehavior() == "ma"
    assert target_module.MoELayer() == "ma_base"

    # A third-party repository composes its inherited manager with replacement feature instances.
    ThirdPartyFeaturesManager.set_features_list([third_party_feature])
    ThirdPartyFeaturesManager.apply_features_patches(args)

    assert mindspeed_original_calls == []
    assert third_party_calls == [("third_party", MindSpeedPatchesManager)]
    assert target_module.MoELayer() == "third_party"
    assert mindspeed_register_calls == [(mindspeed_target, False)]
    assert (
        MindSpeedPatchesManager.patches_info[mindspeed_target].patch_func()
        == "third_party"
    )

    # MA keeps its own feature list, registry, registration history, and effective behavior.
    assert tuple(MAFeaturesManager.FEATURES_LIST) == ma_features_before
    assert MAPatchesManager.patches_info == ma_patches_before
    assert ma_register_calls == ma_register_calls_before

    assert MAPatchesManager.patches_info is not MindSpeedPatchesManager.patches_info
    assert target_module.MAOnlyBehavior() == "ma"
    assert MAPatchesManager.patches_info[ma_target].patch_func() == "ma"

    # Removing the upper MindSpeed layer restores the MA implementation underneath it.
    MindSpeedPatchesManager.remove_patches()

    assert target_module.MoELayer() == "ma_base"
    assert target_module.MAOnlyBehavior() == "ma"
    assert tuple(MAFeaturesManager.FEATURES_LIST) == ma_features_before
    assert MAPatchesManager.patches_info == ma_patches_before
    assert ma_register_calls == ma_register_calls_before
