from __future__ import annotations

import builtins
import importlib
import sys


def test_cosmos3_adapter_import_does_not_require_imagewam(monkeypatch) -> None:
    """The simulator lane must not need the ImageWAM training package."""

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "imagewam" or name.startswith("imagewam."):
            raise ModuleNotFoundError("blocked ImageWAM import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("experiments.robofactory.cosmos3_robofactory_policy", None)
    sys.modules.pop("experiments.robofactory.imagewam_robofactory_policy", None)
    module = importlib.import_module(
        "experiments.robofactory.cosmos3_robofactory_policy"
    )
    assert module.Cosmos3RoboFactoryPolicy.__name__ == "Cosmos3RoboFactoryPolicy"
