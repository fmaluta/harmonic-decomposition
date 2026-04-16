# src/masking.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def inpolygon_mask(X: np.ndarray, Z: np.ndarray, poly_x: np.ndarray, poly_z: np.ndarray) -> np.ndarray:
    """
    Deterministic inpolygon for grid points (MATLAB-compatible for this project):
    - ray casting (crossing number) with half-open rule to avoid vertex double count
    Returns boolean mask (Nz,Nx).
    """
    if X.shape != Z.shape:
        raise ValueError("X and Z must have same shape")

    px = X.ravel()
    py = Z.ravel()

    vx = np.asarray(poly_x, dtype=float).ravel()
    vy = np.asarray(poly_z, dtype=float).ravel()

    # Close polygon if needed
    if vx[0] != vx[-1] or vy[0] != vy[-1]:
        vx = np.concatenate([vx, [vx[0]]])
        vy = np.concatenate([vy, [vy[0]]])

    inside = np.zeros(px.shape, dtype=bool)

    for i in range(len(vx) - 1):
        x0, y0 = vx[i], vy[i]
        x1, y1 = vx[i + 1], vy[i + 1]

        # Half-open rule: include lower end, exclude upper end
        cond = ((y0 <= py) & (py < y1)) | ((y1 <= py) & (py < y0))

        if np.any(cond):
            denom = (y1 - y0)  # nonzero when cond True for non-horizontal edges
            x_int = x0 + (py[cond] - y0) * (x1 - x0) / denom
            inside[cond] ^= (x_int > px[cond])

    return inside.reshape(X.shape)


def build_smask(s: np.ndarray, X: np.ndarray, Z: np.ndarray, freeSurf: Optional[object] = None) -> np.ndarray:
    """
    Equivalent of MATLAB buildSMask.m

    Inputs:
      s: (Nz,Nx,T) base indicator/mask (0/1 or logical-like)
      X,Z: (Nz,Nx)
      free_surf: None or dict with 'Points_0' (x) and 'Points_2' (z)
    Output:
      S: boolean (Nz,Nx,T)
    """
    if s.ndim != 3:
        raise ValueError("build_smask: s must be (Nz,Nx,T)")
    Nz, Nx, T = s.shape
    if X.shape != (Nz, Nx) or Z.shape != (Nz, Nx):
        raise ValueError("build_smask: X and Z must be (Nz,Nx) consistent with s")

    # Fixed exclusion polygon in (|x|, z)
    r_poly = np.array([0.0617, 0.0586, 0.0263, 0.0230, 0.0201, 0.0000, 0.0000, 0.0454, 0.0626], dtype=float)
    z_poly = np.array([0.0594, 0.0627, 0.0627, 0.0595, 0.0200, 0.0200, 0.0040, 0.0040, 0.0220], dtype=float)

    S = (s != 0)

    in_fixed = inpolygon_mask(np.abs(X), Z, r_poly, z_poly)

    # --- Exclusion 2: optional free-surface polygon in (x, z) ---
    # Accept:
    #   - None / {}  -> disabled
    #   - dict with Points_0 / Points_2
    #   - object with attributes Points_0 / Points_2 (MATLAB-like loaded structs)
    xsurf = zsurf = None

    if freeSurf is not None:
        if isinstance(freeSurf, dict):
            xsurf = freeSurf.get("Points_0", None)
            zsurf = freeSurf.get("Points_2", None)
            # also accept pythonic aliases
            if xsurf is None and "x" in freeSurf: xsurf = freeSurf["x"]
            if zsurf is None and "z" in freeSurf: zsurf = freeSurf["z"]
        else:
            # attribute-style (e.g. simple namespace / matlab-like)
            xsurf = getattr(freeSurf, "Points_0", None)
            zsurf = getattr(freeSurf, "Points_2", None)

    if xsurf is not None and zsurf is not None:
        xsurf = np.ravel(np.asarray(xsurf, dtype=float))
        zsurf = np.ravel(np.asarray(zsurf, dtype=float))

        idx = np.argsort(xsurf)
        xsurf = xsurf[idx]
        zsurf = zsurf[idx]

        x_ext = np.concatenate([xsurf, [0.25, 0.25, -0.25, -0.25]])
        z_ext = np.concatenate([zsurf, [zsurf[-1], 1.0, 1.0, zsurf[0]]])

        in_free = inpolygon_mask(X, Z, x_ext, z_ext)
    else:
        in_free = np.zeros((Nz, Nx), dtype=bool)

    excl = in_fixed | in_free
    S = S & (~excl[..., None])
    return S
