"""Notebook-independent helpers extracted from Characterising_PMT.ipynb."""
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


def plot_two_axis_datasets(datasets, title, left_label, right_label=None, yscale="linear", xlim=None, ylim=None):
    fig, ax_left = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if len(datasets) == 0:
        raise ValueError(f"No datasets to plot for {title}")

    first = datasets[0]
    ax_left.plot(first["x"], first["y"], "o-", ms=4, lw=1.8, color=colors[0], label=first["name"])
    ax_left.set_xlabel("Wavelength / Voltage")
    ax_left.set_ylabel(left_label, color=colors[0])
    ax_left.tick_params(axis="y", labelcolor=colors[0])
    ax_left.grid(True, alpha=0.3, which="both")
    if yscale == "log":
        ax_left.set_yscale("log")
    if xlim is not None:
        ax_left.set_xlim(*xlim)
    if ylim is not None:
        ax_left.set_ylim(*ylim)

    axes = [ax_left]
    lines = ax_left.get_lines()

    if len(datasets) > 1:
        second = datasets[1]
        ax_right = ax_left.twinx()
        ax_right.plot(second["x"], second["y"], "s-", ms=4, lw=1.8, color=colors[1], label=second["name"])
        ax_right.set_ylabel(right_label or second["name"], color=colors[1])
        ax_right.tick_params(axis="y", labelcolor=colors[1])
        if yscale == "log":
            ax_right.set_yscale("log")
        if ylim is not None:
            ax_right.set_ylim(*ylim)
        axes.append(ax_right)
        lines += ax_right.get_lines()

    for extra, color in zip(datasets[2:], colors[2:]):
        ax_left.plot(extra["x"], extra["y"], "o-", ms=4, lw=1.5, color=color, label=extra["name"])
        lines += ax_left.get_lines()[-1:]

    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")
    ax_left.set_title(title)
    fig.tight_layout()
    plt.show()
