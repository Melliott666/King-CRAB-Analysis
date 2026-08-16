"""Common user-facing configuration for King CRAB analyses."""
from dataclasses import dataclass, field
from typing import Literal
from .crab_gas_properties import PRESSURE_SCAN_BAR

Gas = Literal['argon','xenon']

@dataclass(frozen=True)
class CRABRunConfig:
    """Inputs common to gas-dependent notebooks.

    Fixed dimensions remain in :mod:`DATA.crab_nexus_geometry`; this object
    contains only run choices that a notebook user should normally change.
    """
    gas: Gas = 'xenon'
    pressure_bar: float = 10.0
    voltage_v: float = 900.0
    temperature_k: float = 293.15
    seed: int = 410000

    def __post_init__(self):
        gas=self.gas.lower()
        if gas not in ('argon','xenon'): raise ValueError("gas must be 'argon' or 'xenon'")
        if self.pressure_bar <= 0: raise ValueError('pressure_bar must be positive')
        object.__setattr__(self,'gas',gas)

STANDARD_PRESSURE_SCAN_BAR = PRESSURE_SCAN_BAR
