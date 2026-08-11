import os

import pytest

from setup_paths import setup_paths


setup_paths()
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="also run tests marked gpu (needs CUDA)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu (CUDA + scene build)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
