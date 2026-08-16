"""Reusable physics and data-analysis calculations for the CRAB detector."""
from .raytrace import Geometry, evaluate
from .analysis import (
    analytic_checks, lens1_capture_crosscheck, paraxial_system,
    projected_apertures, run_dense_convergence, run_gas_pressure_hybrid,
    throughput_profiles,
)
from .digitized import pmt_response
from .pmt import PMTProcessingConfig
from .muon_flux import CylinderGeometry, calculate_muon_flux

__all__ = [
    "Geometry", "evaluate", "analytic_checks", "lens1_capture_crosscheck",
    "paraxial_system", "projected_apertures", "run_dense_convergence",
    "run_gas_pressure_hybrid", "throughput_profiles", "pmt_response",
    "PMTProcessingConfig", "CylinderGeometry", "calculate_muon_flux",
]
