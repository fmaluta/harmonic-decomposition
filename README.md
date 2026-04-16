# Harmonic-regression flow decomposition and energy triad analysis

This repository provides a Python implementation of a **harmonic-regression-based decomposition** of time-resolved velocity fields into:

* mean component
* phase-locked (periodic) component
* residual (stochastic) component

together with the corresponding **plane-averaged energy triad analysis**.

The method is designed for the analysis of impeller-driven flows (e.g. stirred tanks and single-use bioreactors), but is otherwise geometry-agnostic.

---

## Contents

```
harmonicDecomposition/
├─ src/
│  ├─ flow_decomposition.py   # main routine
│  ├─ io.py                  # data I/O utilities
│  ├─ masking.py             # spatial masking
│  ├─ plotting.py            # visualization utilities
│  └─ stats.py               # statistical helpers
├─ examples/
│  └─ example.py             # minimal working example
├─ data/
│  └─ (not included; see below)
```

---

## Method overview

The decomposition is based on the identification of coherent harmonics of the impeller shaft frequency from the power spectral density of a plane-averaged kinetic-energy proxy.

Selected harmonics are reconstructed via linear regression onto sine/cosine bases, yielding:

* mean flow component
* phase-locked (periodic) component
* residual component

A plane-averaged energy triad is then computed from these contributions.

The method:

* does **not** rely on phase averaging or blade-angle binning
* operates directly in the time domain
* extracts periodic content in a data-driven manner

---

## Demo dataset

The demonstration dataset is **not stored in this GitHub repository** due to size limitations.

It is archived separately on Zenodo.

**Demo dataset DOI:**
*(to be added)*

The dataset provided on Zenodo is a **reduced version** of the full CFD dataset, obtained via temporal subsampling (Δt increased by a factor 16) in order to:

* reduce storage requirements
* enable fast execution (~10 s on a standard machine)
* preserve the statistical and spectral features relevant to the decomposition

After downloading, place the dataset in the local `data/` folder:

```
data/
  dataset_demo.h5
```

---

## Running the example

From the repository root:

```bash
python examples/example.py
```

The script:

* loads the dataset
* builds the spatial mask
* performs the harmonic decomposition
* generates diagnostic figures
* saves the decomposition results to:

```
data/decomposition_demo.h5
```

---

## Data layout and conventions

All spatial fields follow the same canonical layout:

* rows correspond to the **z-direction** (`Nz`)
* columns correspond to the **x-direction** (`Nx`)

Required array sizes:

* `X, Z` : `[Nz × Nx]` coordinate grids
* `U, V, W` : `[Nz × Nx × T]` velocity components
* `s` : `[Nz × Nx × T]` validity mask (`1 = valid`, `0 = excluded`)
* `t` : `[T]` uniformly sampled time vector

---

## Requirements

The code depends on:

* `numpy`
* `scipy`
* `matplotlib`
* `h5py`

Install with:

```bash
pip install -r requirements.txt
```

---

## License

The code in this repository is released under the **MIT License** (see `LICENSE`).

---

## Citation

If you use this code in academic work, please cite the associated publication.

Details will be added here upon publication.

---

## Contact

For questions or comments, please contact the corresponding author of the associated publication.

