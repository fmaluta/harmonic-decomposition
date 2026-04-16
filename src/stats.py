# src/stats.py
from __future__ import annotations

import numpy as np

def masked_mean(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    if F.shape != S.shape:
        raise ValueError("masked_mean: F and S must have same shape")
    Sf = S.astype(float, copy=False)
    num = np.nansum(F * Sf, axis=2)
    den = np.nansum(Sf, axis=2)
    out = np.empty_like(num, dtype=np.float64)
    # safe divide, fill NaN where den==0
    np.divide(num, den, out=out, where=(den != 0))
    out[den == 0] = np.nan
    return out

def masked_rms(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    if F.shape != S.shape:
        raise ValueError("masked_rms: F and S must have same shape")
    Sf = S.astype(float, copy=False)
    num = np.nansum((F * F) * Sf, axis=2)
    den = np.nansum(Sf, axis=2)
    out = np.empty_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=(den != 0))
    out[den == 0] = np.nan
    return np.sqrt(out)

def masked_data(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    if F.shape != S.shape:
        raise ValueError("masked_data: F and S must have same shape")
    out = np.array(F, dtype=np.float64, copy=True)
    out[~S.astype(bool)] = np.nan
    return out
