"""Explicit calibration context for high-level legacy analysis helpers."""
from types import ModuleType
from typing import Mapping, Any

def configure_module(module: ModuleType, context: Mapping[str, Any]) -> None:
    """Provide a helper module with notebook-selected calibration inputs.

    Low-level reusable APIs take normal function arguments. This adapter is only
    for high-level historical analyses whose many calibrated arrays are built in
    preceding notebook cells.
    """
    module.__dict__.update(context)
