"""Numerical compatibility helpers shared across analyses."""
import numpy as np

trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:  # NumPy versions without either historical spelling.
    from scipy.integrate import trapezoid
