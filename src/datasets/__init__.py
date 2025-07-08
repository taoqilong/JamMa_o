"""
Dataset registry & auto-discovery.

Usage
-----
from datasets import register, get_dataset_class

@register("whubuildings")
class WhuBuildingsDataset(torch.utils.data.Dataset):
    ...

dataset_cls = get_dataset_class("whubuildings")
ds = dataset_cls(...)
"""
from __future__ import annotations

import pkgutil
import importlib
from pathlib import Path
from typing import Dict, Type

__all__ = [
    "DATASET_REGISTRY",
    "register",
    "get_dataset_class",
]

# -----------------------------------------------------------------------------
# 1. Registry basics
# -----------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    """
    Decorator: @register("mydataset") registers a dataset class.

    Duplicate names (case-insensitive) will raise an error.
    """
    def _decorator(cls):
        key = name.lower()
        if key in DATASET_REGISTRY:
            raise ValueError(f"Dataset name '{name}' already registered!")
        DATASET_REGISTRY[key] = cls
        return cls
    return _decorator


def get_dataset_class(name: str) -> Type:
    """
    Retrieve dataset class by name (case-insensitive).

    Raises:
        KeyError – if the dataset has not been registered.
    """
    try:
        return DATASET_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset '{name}'. "
                       f"Available: {list(DATASET_REGISTRY.keys())}") from exc


# -----------------------------------------------------------------------------
# 2. Auto-import every module in this folder (excluding private / pkg dirs)
# -----------------------------------------------------------------------------
_pkg_dir = Path(__file__).resolve().parent

for mod_info in pkgutil.iter_modules([str(_pkg_dir)]):
    mod_name = mod_info.name
    if mod_name.startswith("_"):      # e.g. _utils.py
        continue
    importlib.import_module(f"{__name__}.{mod_name}")

