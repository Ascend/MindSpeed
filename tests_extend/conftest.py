# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/tests/conftest.py
# reworked/refactored some parts to make it run.
import os
import sys
from multiprocessing.pool import RUN
import pytest


def pytest_configure(config):
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True
    config.addinivalue_line("markers", "slow: tests excluded from the default CI gate")


def _is_slow_context_mapping_case(item):
    path = str(getattr(item, "path", item.fspath)).replace("\\", "/")
    if not path.endswith("mindspeed/core/context_parallel/test_mapping.py"):
        return False

    params = getattr(getattr(item, "callspec", None), "params", {})
    if params.get("gather_scatter_idx") not in (None, (2, 0)):
        return True
    if params.get("input_shape") not in (None, [32, 64, 32]):
        return True
    if params.get("dim") not in (None, 0):
        return True
    return "bfloat16" in str(params.get("dtype"))


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-all"):
        return

    skip_slow = pytest.mark.skip(reason="slow test; pass --run-all to execute")
    for item in items:
        if _is_slow_context_mapping_case(item):
            item.add_marker(pytest.mark.slow)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    if (
        item.cls is not None
        and getattr(item.cls, "reuse_dist_env", True)
        and (not nextitem or item.cls != nextitem.cls)
    ):
        dist_test_class = item.cls()
        if hasattr(dist_test_class, "_pool_cache"):
            for num_procs, pool in dist_test_class._pool_cache.items():
                if pool._state == RUN:  # pool not run when skip UT
                    dist_test_class._close_pool(pool, num_procs, force=True)
            dist_test_class._pool_cache.clear()


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)


TESTS_EXTEND_DIR = os.path.dirname(__file__)
MINDSPORE_TESTS_DIR = os.path.join(TESTS_EXTEND_DIR, "mindspore", "unit_tests")

if MINDSPORE_TESTS_DIR not in sys.path:
    sys.path.insert(0, MINDSPORE_TESTS_DIR)


def pytest_addoption(parser):
    parser.addoption("--ai-framework", action="store", default=None, help="Specify AI framework, e.g., mindspore")
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run the complete unit test suite, including tests marked as slow",
    )
