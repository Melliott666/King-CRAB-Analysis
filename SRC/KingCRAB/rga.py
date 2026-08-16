"""Notebook-independent helpers extracted from Xe_Recapture.ipynb."""
from __future__ import annotations
from pathlib import Path
import os, re, json, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
 from scipy.integrate import trapezoid, quad
 from scipy.optimize import curve_fit
 from scipy.signal import butter, filtfilt, welch
 from scipy.stats import norm, binom
except ImportError:
 pass


def ReadFile(file_path):

    # Open and read the file, skipping lines until we reach the data section
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find where the data starts
    for i, line in enumerate(lines):
        if line.strip() == "Analog Scan Data":
            data_start = i + 3 # Skip the header and seperator
            break

    # Load only the numerical data into a DataFrame
    df = pd.read_csv(file_path, skiprows=data_start, delimiter=",")

    # Display the first few rows
    print(df.head())

    return df

def normalize_area(df):
    #Trapezodial integration along the curve
    area = np.trapezoid(df["Intensity"], df["Mass"])
    if area == 0:
        return df
    df_norm = df.copy()
    #Normalize the new data to 1
    df_norm["Intensity"] = df["Intensity"] / area
    return df_norm

def subtract_background(df, bg):
    #The data is not necessaryly along the same axis therefore I want to get them onto the same axis
    bg_interp = np.interp(df["Mass"], bg["Mass"], bg["Intensity"])
    df_corr = df.copy()
    #Subtract the background from the data frame
    df_corr["Intensity"] = df_corr["Intensity"] - bg_interp
    return df_corr

def ROI(df, lo, hi):
    """Integrate intensity over [lo, hi], clipping negatives to 0."""
    m = df["Mass"].values
    y = df["Intensity"].values
    mask = (m >= lo) & (m <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(np.clip(y[mask], 0, None), m[mask]))

def compositions(df_bgsub, windows=None):
    if windows is None: windows=WINDOWS
    """Return absolute areas (parts) and % fractions (fracs) for each ROI."""
    parts = {name: ROI(df_bgsub, *w) for name, w in windows.items()}
    total_all = ROI(df_bgsub, df_bgsub["Mass"].min(), df_bgsub["Mass"].max())
    other = max(total_all - sum(parts.values()), 0.0)  # never negative
    parts["Other"] = other

    denom = sum(parts.values())
    if denom <= 0:
        fracs = {k: 0.0 for k in parts}
    else:
        fracs = {k: 100.0 * v / denom for k, v in parts.items()}
    return parts, fracs

def highlight_windows(ax, windows=None, alpha=0.15):
    if windows is None: windows=WINDOWS
    """
    Shade ROI bands on an existing matplotlib Axes.
    Call after plotting m/z vs intensity.
    """
    for name, (lo, hi) in windows.items():
        ax.axvspan(lo, hi, alpha=alpha, label=name)
    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False)

def xe_percent(df_bs, windows=None):
    if windows is None: windows=WINDOWS
    parts = compositions(df_bs, windows)[0]  # <-- take Parts only
    Xe = parts["Xe_Single"] + parts["Xe_Double"]
    Ar = parts["Ar_Single"] + parts["Ar_Double"]
    denom = Xe + Ar
    if denom <= 0:
        raise ValueError("Xe + Ar is zero or negative; check your windows or data.")
    return 100 * Xe / denom

def xe_percent_series(files, backgrounds, windows=None, names=None):
    if windows is None: windows=WINDOWS
    if names is None:
        names = [Path(f).stem for f in files]
    out = []
    for nm, fp in zip(names, files):
        df = read_rga(fp)
        df_bs = subtract_background(df, backgrounds)
        xp = xe_percent(df_bs, windows)
        out.append((nm, xp))
    return pd.DataFrame(out, columns=["Dataset", "Xe_%"])

def read_rga(fp):
        df = ReadFile(fp)  # your existing reader
        # match your earlier renames
        df = df.rename(columns={
            "              Mass (AMU)": "Mass",
            "         Intensity (Torr)": "Intensity",
            "Mass (AMU)": "Mass",
            "Intensity (Torr)": "Intensity",
            "Partial Pressure": "Intensity",
        })
        # keep only numeric Mass/Intensity, sorted
        df = df[["Mass", "Intensity"]].apply(pd.to_numeric, errors="coerce").dropna()
        df = df.sort_values("Mass").reset_index(drop=True)
        return df

def find_data_start(path):
    with open(path,"r",errors="ignore") as f:
        for i,line in enumerate(f):
            if line.strip()=="Analog Scan Data": return i+3
    return 0

def read_rga(path):
    df = pd.read_csv(path, skiprows=find_data_start(path), delimiter=",")
    df.columns = df.columns.astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    df = df.rename(columns={"Mass (AMU)":"Mass","Intensity (Torr)":"Intensity","Partial Pressure":"Intensity"})
    df = df[["Mass","Intensity"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("Mass").reset_index(drop=True)
    return df

def avg_background_intensity_at(masses, bkgs):
    vals = [np.interp(masses, b["Mass"].to_numpy(), b["Intensity"].to_numpy()) for b in bkgs]
    return np.mean(np.vstack(vals), axis=0)

def subtract_background(df, bkgs):
    m = df["Mass"].to_numpy()
    y = df["Intensity"].to_numpy()
    yb = avg_background_intensity_at(m, bkgs)
    out = df.copy()
    out["Intensity"] = y - yb
    return out

def integrate_window(df, lo, hi):
    m = df["Mass"].to_numpy()
    y = np.clip(df["Intensity"].to_numpy(), 0, None)
    mask = (m>=lo)&(m<=hi)
    return float(np.trapz(y[mask], m[mask])) if mask.any() else 0.0

def compositions(df_bs, WINDOWS):
    Parts = {name: integrate_window(df_bs, *win) for name,win in WINDOWS.items()}

    total_pos = integrate_window(df_bs, float(df_bs["Mass"].min()), float(df_bs["Mass"].max()))
    Parts["Other"] = max(total_pos - sum(Parts.values()), 0.0)

    Xe = Parts["Xe_Single"] + Parts["Xe_Double"]
    Ar = Parts["Ar_Single"] + Parts["Ar_Double"]

    Tot = Xe + Ar + Parts["Other"]

    Frac = {
        "Xe_Single": 100*Parts["Xe_Single"]/Tot if Tot>0 else 0.0,
        "Xe_Double": 100*Parts["Xe_Double"]/Tot if Tot>0 else 0.0,
        "Ar_Single": 100*Parts["Ar_Single"]/Tot if Tot>0 else 0.0,
        "Ar_Double": 100*Parts["Ar_Double"]/Tot if Tot>0 else 0.0,
        "Other":     100*Parts["Other"]/Tot     if Tot>0 else 0.0,}

    Frac_noOther = {
        "Xe": 100*Xe/(Xe+Ar) if (Xe+Ar)>0 else 0.0,
        "Ar": 100*Ar/(Xe+Ar) if (Xe+Ar)>0 else 0.0}

    Parts["Xe"] = Xe
    Parts["Ar"] = Ar
    Parts["Total"] = Tot

    return Parts, Frac, Frac_noOther
