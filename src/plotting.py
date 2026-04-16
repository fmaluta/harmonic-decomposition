# src/plotting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class PlotHandles:
    fig: Any
    ax: Any
    im: Any
    cb: Any
    patch: Any


def plot_modes(
    x: np.ndarray,
    z: np.ndarray,
    M: np.ndarray,
    colorlim: Optional[Tuple[float, float]] = None,
    opts: Optional[Dict[str, Any]] = None,
    *,
    ax: Any = None,
) -> PlotHandles:
    """
    Equivalent of MATLAB plotModes.m (imagesc-like + polygon overlay).

    If ax is provided, plot into that axes and do NOT create a new figure.
    """

    if opts is None:
        opts = {}

    # Defaults
    def_poly_color = (0.7, 0.7, 0.7)
    rpolyg_p = np.array([0.0454, 0.0626, 0.0617, 0.0586, 0.0263, 0.0230, 0.0201], dtype=float)
    zpolyg_p = np.array([0.0040, 0.0220, 0.0594, 0.0627, 0.0627, 0.0595, 0.0200], dtype=float)
    def_poly_x = np.concatenate([rpolyg_p, -rpolyg_p[::-1]])
    def_poly_z = np.concatenate([zpolyg_p, zpolyg_p[::-1]])

    title = opts.get("title", "")
    xlabel = opts.get("xlabel", "x [m]")
    ylabel = opts.get("ylabel", "z [m]")
    cbar_label = opts.get("cbar_label", "")
    overlay_poly = bool(opts.get("overlay_poly", True))
    poly_x = np.array(opts.get("poly_x", def_poly_x), dtype=float)
    poly_z = np.array(opts.get("poly_z", def_poly_z), dtype=float)
    poly_color = opts.get("poly_color", def_poly_color)
    poly_lw = float(opts.get("poly_lw", 3.0))
    font_size = float(opts.get("font_size", 16))
    font_name = opts.get("font_name", "Liberation Serif")    
    cmap = opts.get("cmap", "viridis")
    cbar_size = opts.get("cbar_size", "3%")
    cbar_pad = float(opts.get("cbar_pad", 0.08))

    x = np.ravel(x).astype(float)
    z = np.ravel(z).astype(float)

    Nz, Nx = M.shape
    if x.size != Nx or z.size != Nz:
        raise ValueError(f"plot_modes: size mismatch. x:{x.size}, z:{z.size}, M:{M.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")
    else:
        fig = ax.figure

    # Main map
    Xg, Zg = np.meshgrid(x, z)
    im = ax.pcolormesh(Xg, Zg, M, shading="auto", cmap=cmap)
    ax.set_aspect("equal")

    if colorlim is not None and len(colorlim) == 2 and (colorlim[1] > colorlim[0]):
        im.set_clim(colorlim[0], colorlim[1])

    # Impeller / polygon
    patch = None
    if overlay_poly:
        patch = ax.fill(
            poly_x,
            poly_z,
            facecolor=poly_color,
            edgecolor=poly_color,
            linewidth=poly_lw,
            zorder=5,
        )[0]

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=font_size, fontname=font_name)
    ax.set_ylabel(ylabel, fontsize=font_size, fontname=font_name)
    if title:
        ax.set_title(title, fontsize=font_size, fontname=font_name)

    ax.tick_params(axis="both", labelsize=font_size)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)

    # Colorbar matched to plotted axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    cb = fig.colorbar(im, cax=cax)

    if cbar_label:
        cb.set_label(cbar_label, fontsize=font_size, fontname=font_name)

    cb.ax.tick_params(labelsize=font_size)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontname(font_name)
        
    #plt.tight_layout()

    return PlotHandles(fig=fig, ax=ax, im=im, cb=cb, patch=patch)
