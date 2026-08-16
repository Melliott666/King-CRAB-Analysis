"""Loading and interpolation helpers for digitized PMT/vendor curves."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from DATA.crab_optical_properties import PMT_CHARACTERISTICS_FILE, PMT_SPECTRAL_RESPONSE_FILE
from SRC.KingCRAB.raytrace import emission_wavelength_nm


def load_digitized_datasets(path: str | Path) -> dict[str, np.ndarray]:
    """Return each WebPlotDigitizer dataset as a sorted ``(x, y)`` array."""
    raw = json.loads(Path(path).read_text())
    result = {}
    for dataset in raw.get("datasetColl", []):
        values = np.asarray(
            [point["value"][:2] for point in dataset.get("data", [])], dtype=float
        )
        if values.size:
            result[dataset.get("name", "Dataset")] = values[np.argsort(values[:, 0])]
    return result


def as_plot_datasets(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Return digitized data in the legacy plotting-notebook representation."""
    return [
        {"name": name, "x": values[:, 0], "y": values[:, 1]}
        for name, values in load_digitized_datasets(path).items()
    ]


def log_interpolate(points: np.ndarray, x: float) -> float:
    """Log-linear interpolation, rejecting extrapolation."""
    points = np.asarray(points, dtype=float)
    if not points[:, 0].min() <= x <= points[:, 0].max():
        raise ValueError(f"{x} lies outside the digitized data range")
    return float(np.exp(np.interp(x, points[:, 0], np.log(points[:, 1]))))


def log_linear_extrapolation(
    points: np.ndarray, x: float, n_points: int = 8, *, return_fit: bool = False
):
    """Extrapolate log(response) from the lowest-x digitized points."""
    frame = pd.DataFrame(points, columns=["x", "response"]).groupby("x", as_index=False).response.mean()
    fit = frame.nsmallest(n_points, "x")
    fit = fit.assign(wavelength_nm=fit.x)
    slope, intercept = np.polyfit(fit.x, np.log(fit.response), 1)
    prediction = float(np.exp(intercept + slope * x))
    return (prediction, fit, slope, intercept) if return_fit else prediction


def pmt_response(gas: str, pressure_bar: float, voltage_v: float = 900.0, n_low_points: int = 8):
    """Return gas/pressure-dependent PMT wavelength, QE estimate, gain, and dark current."""
    wavelength=emission_wavelength_nm(gas,pressure_bar)
    spectral=load_digitized_datasets(PMT_SPECTRAL_RESPONSE_FILE)
    characteristics=load_digitized_datasets(PMT_CHARACTERISTICS_FILE)
    qe=log_linear_extrapolation(spectral['Quantum Efficiency'],wavelength,n_low_points)
    radiant=log_linear_extrapolation(spectral['Cathode Radiant Sensitivity'],wavelength,n_low_points)
    qe_radiant=124.0*radiant/wavelength
    gain=characteristics['Gain'].copy(); gain[:,1]*=1e13
    dark=characteristics['Anode Dark Current'].copy(); dark[:,1]*=1e-12
    return dict(gas=gas.lower(),pressure_bar=pressure_bar,wavelength_nm=wavelength,qe_percent=qe,qe_from_radiant_percent=qe_radiant,pde=np.sqrt(qe*qe_radiant)/100,gain=log_interpolate(gain,voltage_v),dark_current_A=log_interpolate(dark,voltage_v))
