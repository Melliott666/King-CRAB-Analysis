"""Read-only helpers for NEXUS HDF5 output."""
from pathlib import Path
import h5py
import numpy as np

def nexus_inventory(path):
    """Return dataset shape/dtype metadata without modifying the file."""
    inventory={}
    with h5py.File(Path(path),'r') as handle:
        handle.visititems(lambda name,obj: inventory.update({name:{'shape':obj.shape,'dtype':str(obj.dtype)}}) if isinstance(obj,h5py.Dataset) else None)
    return inventory

def read_nexus_dataset(path,dataset):
    with h5py.File(Path(path),'r') as handle:return handle[dataset][:]

def detected_photons(path,dataset='/MC/sns_response',field='charge'):
    data=read_nexus_dataset(path,dataset)
    if data.dtype.names and field in data.dtype.names:return float(np.sum(data[field]))
    return float(np.sum(data))
