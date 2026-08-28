from pathlib import Path

import pytest

TRITON_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="slow test; pass --run-all to execute")
    for item in items:
        item_path = getattr(item, "path", None) or item.fspath
        if TRITON_DIR not in Path(str(item_path)).resolve().parents:
            continue
        item.add_marker(pytest.mark.slow)
        if not config.getoption("--run-all"):
            item.add_marker(skip_slow)
