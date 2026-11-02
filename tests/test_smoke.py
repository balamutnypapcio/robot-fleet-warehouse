import importlib

import pytest


@pytest.mark.parametrize("package", ["fleet", "sim", "verification", "experiments"])
def test_package_imports(package):
    importlib.import_module(package)
