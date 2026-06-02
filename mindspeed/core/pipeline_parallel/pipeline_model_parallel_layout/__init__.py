"""Pipeline model parallel layout support."""


def __getattr__(name):
    if name == "LayerType":
        from .layout import LayerType

        return LayerType
    if name == "PipelineParallelLayerLayout":
        from .layout import PipelineParallelLayerLayout

        return PipelineParallelLayerLayout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
