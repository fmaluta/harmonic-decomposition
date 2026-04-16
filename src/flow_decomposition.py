# src/flow_decomposition.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, welch, get_window

from .stats import masked_mean, masked_rms
from .plotting import plot_modes


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS = {
    "psd": {"highpass_Hz": 0.0, "highpass_order": 4},  # kept for future parity; not used here
    "welch": {"Tw": None, "ovlp": 0.5, "nfft": None, "minL": 256},
    "harmonics": {
        "Nmax": None,
        "min_frac": 0.005,
        "guard_Hz": 0.15,
        "keep_shaft": True,
        "force_lines": [],
    },
    "contrast": {"Delta_bins": 2, "bg_bins": [3, 8]},
    "compute": {"show": True, "save_figs": True},
    "dpi": 200,
    "figures_dir": None,  # optional override
}


# =============================================================================
# Utilities
# =============================================================================

def _deep_update(base: Dict[str, Any], upd: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if not upd:
        return out
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _nextpow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return int(2 ** np.ceil(np.log2(n)))


def _figures_dir(opt: Dict[str, Any]) -> Path:
    if opt.get("figures_dir", None):
        d = Path(opt["figures_dir"]).resolve()
    else:
        root = Path(__file__).resolve().parents[1]
        d = root / "Figures"
    d.mkdir(exist_ok=True)
    return d


def _save_fig(fig: plt.Figure, filename: str, opt: Dict[str, Any]) -> None:
    if not opt["compute"]["save_figs"]:
        return
    path = _figures_dir(opt) / filename
    fig.savefig(path, dpi=int(opt.get("dpi", 600)), bbox_inches="tight")


def _maybe_show(fig: plt.Figure, opt: Dict[str, Any]) -> None:
    if opt["compute"]["show"]:
        plt.show()
    else:
        plt.close(fig)


def _plot_save_modes(
    X: np.ndarray,
    Z: np.ndarray,
    M: np.ndarray,
    *,
    title: str,
    filename: str,
    opt: Dict[str, Any],
    colorlim=(0.0, 0.3),
    overlay_poly=True,
    xlabel="x [m]",
    ylabel="z [m]",
    cbar_label="",
    font_size=16,
    font_name="Liberation Serif",
    poly_lw=3.0,
    quiver=None,
) -> None:
    x = X[0, :]
    z = Z[:, 0]

    h = plot_modes(
        x,
        z,
        M,
        colorlim=colorlim,
        opts={
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "cbar_label": cbar_label,
            "overlay_poly": overlay_poly,
            "font_size": font_size,
            "font_name": font_name,
            "poly_lw": poly_lw,
        },
    )

    if quiver is not None:
        h.ax.quiver(
            quiver["X"],
            quiver["Z"],
            quiver["U"],
            quiver["W"],
            color=quiver.get("color", "k"),
            scale=quiver.get("scale", None),
        )
        
    plt.tight_layout()

    _save_fig(h.fig, filename, opt)
    _maybe_show(h.fig, opt)
    
def _plot_triptych_modes(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    titles,
    suptitle: str,
    filename: str,
    opt: Dict[str, Any],
    colorlim=None,
    overlay_poly=True,
) -> None:
    x = X[0, :]
    z = Z[:, 0]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle)

    plot_modes(x, z, A, colorlim=colorlim, opts={"title": titles[0], "overlay_poly": overlay_poly}, ax=axs[0])
    plot_modes(x, z, B, colorlim=colorlim, opts={"title": titles[1], "overlay_poly": overlay_poly}, ax=axs[1])
    plot_modes(x, z, C, colorlim=colorlim, opts={"title": titles[2], "overlay_poly": overlay_poly}, ax=axs[2])

    _save_fig(fig, filename, opt)
    _maybe_show(fig, opt)
    

# =============================================================================
# Main function
# =============================================================================

def flow_decomposition(
    DATA: Dict[str, Any],
    META: Dict[str, Any],
    OPT: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    opt = _deep_update(DEFAULTS, OPT or {})

    # ---- Time ----
    t = np.asarray(DATA["t"], dtype=np.float64).ravel()
    dt = float(np.mean(np.diff(t)))
    Fs = 1.0 / dt
    Nyq = Fs / 2.0

    rpm = float(META["rpm"])
    nbl = int(META["nblades"])
    fShaft = rpm / 60.0
    fBPF = nbl * fShaft

    # ---- Fields ----
    X = np.asarray(DATA["X"], dtype=np.float64)
    Z = np.asarray(DATA["Z"], dtype=np.float64)
    U = np.asarray(DATA["U"], dtype=np.float64)
    V = np.asarray(DATA["V"], dtype=np.float64)
    W = np.asarray(DATA["W"], dtype=np.float64)
    S = np.asarray(DATA["S"], dtype=bool)

    # ---- Means ----
    Um = masked_mean(U, S)
    Vm = masked_mean(V, S)
    Wm = masked_mean(W, S)

    U0 = U - Um[:, :, None]
    V0 = V - Vm[:, :, None]
    W0 = W - Wm[:, :, None]

    # ---- Detection signal ----
    Sf = S.astype(np.float64, copy=False)
    K = 0.5 * (U0 * U0 + V0 * V0 + W0 * W0) * Sf
    Kfig = np.nanmean(np.nanmean(K, axis=0), axis=0)
    q = np.nanmean(np.nanmean(K, axis=0), axis=0)
    q = detrend(q, type="linear")

    # ---- Welch ----
    Tw = opt["welch"]["Tw"]
    if Tw is None:
        Tw = min(6.0, max(1.5, 20.0 / max(fShaft, 1e-12)))

    Lw = max(int(opt["welch"]["minL"]), int(round(float(Tw) * Fs)))
    Nov = int(round(float(opt["welch"]["ovlp"]) * Lw))
    nfft = opt["welch"]["nfft"]
    if nfft is None:
        nfft = _nextpow2(Lw)

    f_psd, P_psd = welch(
        q,
        fs=Fs,
        window=get_window("hann", Lw, fftbins=True),
        nperseg=Lw,
        noverlap=Nov,
        nfft=int(nfft),
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    # ---- Candidate grid ----
    if opt["harmonics"]["Nmax"] is None:
        Nmax = int(np.floor(min(Nyq / max(fShaft, 1e-12), nbl * 10)))
    else:
        Nmax = int(opt["harmonics"]["Nmax"])

    fgrid = np.arange(2, Nmax + 1, dtype=np.float64) * fShaft
    fgrid = fgrid[fgrid < Nyq]

    # ---- Contrast ----
    df_psd = float(f_psd[1] - f_psd[0])
    Delta = int(opt["contrast"]["Delta_bins"]) * df_psd
    W1 = int(opt["contrast"]["bg_bins"][0]) * df_psd
    W2 = int(opt["contrast"]["bg_bins"][1]) * df_psd

    Ck = np.zeros_like(fgrid, dtype=np.float64)
    f_min = float(f_psd[1])  # MATLAB f(2)
    f_max = float(f_psd[-1])

    for i, fk in enumerate(fgrid):
        f1 = max(f_min, float(fk) - Delta)
        f2 = min(f_max, float(fk) + Delta)

        band = (f_psd >= f1) & (f_psd <= f2)
        Pband = float(np.trapezoid(P_psd[band], f_psd[band])) if np.any(band) else 0.0

        Lmask = ((f_psd >= fk - W2) & (f_psd <= fk - W1)) | ((f_psd >= fk + W1) & (f_psd <= fk + W2))
        Pmed = float(np.median(P_psd[Lmask])) if np.any(Lmask) else float(np.median(P_psd))

        Ck[i] = max(0.0, Pband - Pmed * (f2 - f1))
        

    Cfrac = Ck / (float(np.sum(Ck)) + np.finfo(np.float64).eps)

    # ---- Keep lines ----
    kept = fgrid[Cfrac >= float(opt["harmonics"]["min_frac"])].copy()

    # Always keep BPF
    kept = np.unique(np.concatenate([kept, np.array([fBPF], dtype=np.float64)]))

    # Optionally drop shaft vicinity
    if not bool(opt["harmonics"]["keep_shaft"]):
        kept = kept[np.abs(kept - fShaft) > float(opt["harmonics"]["guard_Hz"])]

    kept.sort()

    # ---- Regression (full-coverage fast path; your mask is typically full except impeller) ----
    # Design matrix: [cos(w1 t), sin(w1 t), cos(w2 t), sin(w2 t), ...]
    Xreg = np.column_stack([func(2.0 * np.pi * fk * t) for fk in kept for func in (np.cos, np.sin)])
    XtX = Xreg.T @ Xreg
    lam = 1e-10 * float(np.trace(XtX)) / XtX.shape[0]
    G = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xreg.T)

    Nz, Nx, T = U.shape
    Upl = (Xreg @ (G @ U0.reshape(-1, T).T)).T.reshape(U.shape)
    Vpl = (Xreg @ (G @ V0.reshape(-1, T).T)).T.reshape(V.shape)
    Wpl = (Xreg @ (G @ W0.reshape(-1, T).T)).T.reshape(W.shape)

    Ures = U0 - Upl
    Vres = V0 - Vpl
    Wres = W0 - Wpl

    # Replace non-finite residuals with 0 at masked or empty locations    
    Ures[~np.isfinite(Ures)] = 0.0
    Vres[~np.isfinite(Vres)] = 0.0
    Wres[~np.isfinite(Wres)] = 0.0

    # ---- Energy triad ----
    occ = np.nanmean(Sf, axis=2)
    Emean = float(np.nansum((Um * Um + Vm * Vm + Wm * Wm) * occ) / (np.nansum(occ) + 1e-15))
    EPL = float(np.nanmean((Upl * Upl + Vpl * Vpl + Wpl * Wpl) * Sf))
    Eres = float(np.nanmean((Ures * Ures + Vres * Vres + Wres * Wres) * Sf))

    frac = np.array([Emean, EPL, Eres], dtype=np.float64)
    frac /= float(np.sum(frac) + 1e-15)

    # ---- Post-processing maps ----    
    Urms_pl = masked_rms(Upl, S)
    Vrms_pl = masked_rms(Vpl, S)
    Wrms_pl = masked_rms(Wpl, S)

    Urms_res = masked_rms(Ures, S)
    Vrms_res = masked_rms(Vres, S)
    Wrms_res = masked_rms(Wres, S)

    mag_mean = np.sqrt(Um**2 + Vm**2 + Wm**2)
    mag_pl   = np.sqrt(Urms_pl**2 + Vrms_pl**2 + Wrms_pl**2)
    mag_res  = np.sqrt(Urms_res**2 + Vrms_res**2 + Wrms_res**2)

    maps = {
        "mag_mean": mag_mean,
        "mag_pl": mag_pl,
        "mag_res": mag_res,
        "Urms_pl": Urms_pl, "Vrms_pl": Vrms_pl, "Wrms_pl": Wrms_pl,
        "Urms_res": Urms_res, "Vrms_res": Vrms_res, "Wrms_res": Wrms_res,
    }

    # ---- Figures ----
    # 1) Harmonic selection bars
    font_size = 16
    font_name = "Liberation Serif"
    fig, ax = plt.subplots()
    ax.bar(fgrid / fShaft, Cfrac, width=0.9, color='k')
    ax.set_xlabel("$f_k/f_{shaft}$",fontsize=font_size, fontname=font_name)
    ax.set_ylabel("$\chi(f_k)$",fontsize=font_size, fontname=font_name)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(True)
    ax.set_xlim(0,33)
    plt.tight_layout()

    _save_fig(fig, "phase_lines.png", opt)
    _maybe_show(fig, opt)

    # 2) PSD of q
    fig, ax = plt.subplots()
    ax.loglog(f_psd, P_psd,'-k')
    ax.set_xlabel("$f \, [Hz]$",fontsize=font_size, fontname=font_name)
    ax.set_ylabel("$S(f)$",fontsize=font_size, fontname=font_name)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_xlim(0.4,100)
    plt.tight_layout()

    _save_fig(fig, "q_psd.png", opt)
    _maybe_show(fig, opt)
    
    # 2A) k vs t
    fig, ax = plt.subplots()
    ax.plot(t-t[0],Kfig,'-k')
    ax.set_xlabel("$t \, [s]$",fontsize=font_size, fontname=font_name)
    ax.set_ylabel("$k(t) \, [m^2/s^2]$",fontsize=font_size, fontname=font_name)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_xlim(0,np.max(t-t[0]))
    plt.tight_layout()

    _save_fig(fig, "k_vs_t.png", opt)
    _maybe_show(fig, opt)

    # 3) Triad energy
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].bar([0, 1, 2], [Emean, EPL, Eres])
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(["Mean", "PL", "RES"])
    axs[0].set_ylabel("Absolute energy (plane)")
    axs[0].grid(True)

    axs[1].pie(frac, labels=["Mean", "PL", "RES"])
    fig.suptitle("Energy triad")
    _save_fig(fig, "triad_energy.png", opt)
    _maybe_show(fig, opt)

    # 4) Magnitude maps
    colorlim = (0.0, 0.3)
    _plot_save_modes(X, Z, mag_mean, title="Mean flow |U|", filename="mag_mean.png", opt=opt, colorlim=colorlim)
    _plot_save_modes(X, Z, mag_pl,   title="Phase-locked RMS |U_PL|", filename="mag_pl_rms.png", opt=opt, colorlim=colorlim)
    _plot_save_modes(X, Z, mag_res,  title="Residual RMS |U_RES|", filename="mag_res_rms.png", opt=opt, colorlim=colorlim)

    # 5) Component triptychs (requested)
    _plot_triptych_modes(
        X, Z,
        Urms_pl, Vrms_pl, Wrms_pl,
        titles=("U_PL RMS", "V_PL RMS", "W_PL RMS"),
        suptitle="Phase-locked RMS components",
        filename="pl_rms_components.png",
        opt=opt,
        colorlim=None,
    )

    _plot_triptych_modes(
        X, Z,
        Urms_res, Vrms_res, Wrms_res,
        titles=("U_RES RMS", "V_RES RMS", "W_RES RMS"),
        suptitle="Residual RMS components",
        filename="res_rms_components.png",
        opt=opt,
        colorlim=None,
    )

    return {
        "meta": {
            "Fs": Fs,
            "dt": dt,
            "rpm": rpm,
            "nblades": nbl,
            "fShaft": fShaft,
            "fBPF": fBPF,
            "Nyq": Nyq,
        },
        "psd": {"f": f_psd, "P": P_psd, "Lw": Lw, "Nov": Nov, "nfft": int(nfft)},
        "scan": {"fgrid": fgrid, "Cfrac": Cfrac, "red_grid": fgrid / fShaft},
        "kept": {"f": kept, "red": kept / fShaft},
        "PL": {"Upl": Upl, "Vpl": Vpl, "Wpl": Wpl, "kept_lines": kept},
        "residual": {"Ures": Ures, "Vres": Vres, "Wres": Wres},
        "triad": {"Emean": Emean, "Epl": EPL, "Eres": Eres, "frac": frac},
        "maps": maps,
        "figs_dir": str(_figures_dir(opt)),
    }
