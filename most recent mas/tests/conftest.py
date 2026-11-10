"""Shared pytest options."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run tests marked slow (very long model-quality cases)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: very long case; runs only with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="very long case: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
