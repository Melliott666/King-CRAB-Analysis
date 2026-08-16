# King CRAB analysis

The project is organized so that notebooks contain embedded
figures, while reusable physics and detector values live in importable Python
modules.

## Layout

- `DATA/` — authoritative inputs and NEXUS-transcribed constants. The
  `crab_nexus_geometry.py`, `crab_gas_properties.py`, and
  `crab_optical_properties.py` modules are the only source of hard-coded CRAB
  periscope values.
- `SRC/KingCRAB/` — all reusable calculations. Notebooks contain no function or
  class definitions and import their analysis routines from here.
- `Notebooks/` — scientific analysis notebooks other than the authoritative
  periscope notebook.
- `CRAB_PERISCOPE_OPTIMIZATION.ipynb` — executed, self-contained optics report;
  all tables and figures are embedded.

The former `CODE/` scratch directory has been removed. Superseded periscope,
CaF2, focusing, and Photon-Yield scratch notebooks were retired after their
validated calculations were incorporated into the final implementation.

Empty/duplicate GammaGen, Muons, and pulse-height scratch notebooks were also
retired. Removed material remains recoverable from the dated cleanup folder in
the user's Trash.

## Configuration and shared APIs

Notebook inputs belong in their first configuration cell. Gas-dependent work
uses a common object:

```python
from DATA.crab_config import CRABRunConfig
run = CRABRunConfig(gas="xenon", pressure_bar=10.0, voltage_v=900.0)
```

Fixed NEXUS values are never copied into notebooks. They live in
`DATA/crab_nexus_geometry.py`, `DATA/crab_gas_properties.py`, and
`DATA/crab_optical_properties.py`.

The main reusable modules are:

- `pmt.py`: waveform loading, baseline correction, filtering, pulse charge,
  averaging, and voltage-folder discovery.
- `digitized.py`: vendor-table loading/interpolation and the gas/pressure-aware
  `pmt_response` calculation.
- `muon_flux.py`: Guan surface spectrum using the NEXUS vessel dimensions.
- `hdf5.py`: read-only NEXUS HDF5 inspection and detected-photon totals.
- `reflectivity.py`: loading and stitching measured mirror curves.
- `analysis.py` and `raytrace.py`: analytic and deterministic periscope optics.
- Domain modules such as `gain.py`, `pmt_noise.py`, `radioactive.py`, and
  `rga.py`: analysis-specific helpers formerly repeated inside notebooks.

Low-level functions take ordinary arguments. A few high-level historical
analyses construct calibrated arrays over several notebook cells; these use the
single explicit `context.configure_module` adapter before calling their helper
functions.

## Periscope API

```python
from SRC.KingCRAB import Geometry, evaluate
from SRC.KingCRAB.analysis import (
    lens1_geometrical_capture,
    paraxial_system,
    run_dense_convergence,
    run_gas_pressure_hybrid,
)
```

The ray integration is deterministic. Absolute efficiencies are probabilities
per isotropically emitted EL photon. Columns ending in
`post_L1_percent` divide by the geometrical Lens-1 capture integrated across the
complete finite EL region.

## Reproducibility

Run the final notebook in place:

```bash
python3 -m nbconvert --to notebook --execute --inplace \
  CRAB_PERISCOPE_OPTIMIZATION.ipynb --ExecutePreprocessor.timeout=3600
```

Repository-only execution has been verified for the periscope, PMT
characterization, mirror-reflectivity, beta/NEXUS, and cosmic-muon notebooks.
Notebooks referencing `/Volumes/Untitled` require that acquisition drive to be
mounted. Ba-133 uses its explicitly configured archive in `Downloads`.