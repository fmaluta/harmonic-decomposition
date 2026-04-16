from __future__ import annotations

import os
from typing import Any, Dict

import h5py
import numpy as np


def load_h5_dataset_full(
    h5_path: str,
    time_slice: slice = slice(None),
) -> Dict[str, np.ndarray]:
    """
    Load canonical HDF5 dataset and return data dict:
      X, Z: (Nz, Nx)
      U, V, W, s: (Nz, Nx, Tsel)
      t: (Tsel,)
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        required = ["X", "Z", "t", "U", "V", "W", "s"]
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"Missing required datasets in HDF5 file: {missing}")

        X = np.asarray(f["X"][...], dtype=np.float32)
        Z = np.asarray(f["Z"][...], dtype=np.float32)
        t = np.ravel(np.asarray(f["t"][time_slice], dtype=np.float32))

        if X.ndim != 2 or Z.ndim != 2:
            raise ValueError(f"X and Z must be 2D arrays, got X{X.shape}, Z{Z.shape}")
        if X.shape != Z.shape:
            raise ValueError(f"X and Z shape mismatch: X{X.shape}, Z{Z.shape}")

        Nz, Nx = X.shape
        Tsel = t.size

        def load3(name: str) -> np.ndarray:
            arr = np.asarray(f[name][:, :, time_slice], dtype=np.float32)
            if arr.shape != (Nz, Nx, Tsel):
                raise ValueError(
                    f"Dataset '{name}' has shape {arr.shape}, expected {(Nz, Nx, Tsel)}"
                )
            return arr

        return {
            "X": X,
            "Z": Z,
            "t": t,
            "U": load3("U"),
            "V": load3("V"),
            "W": load3("W"),
            "s": load3("s"),
        }


def load_data(source: str, **kwargs: Any) -> Dict[str, np.ndarray]:
    if source == "h5":
        return load_h5_dataset_full(
            kwargs["h5_path"],
            time_slice=kwargs.get("time_slice", slice(None)),
        )
    raise ValueError("Only source='h5' is supported in this public repository.")


def _write_h5_item(h5group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, dict):
        subgrp = h5group.create_group(key)
        for subkey, subval in value.items():
            _write_h5_item(subgrp, subkey, subval)
        return

    if value is None:
        ds = h5group.create_dataset(key, data=np.empty((0,), dtype=np.float32))
        ds.attrs["__is_none__"] = True
        return

    if isinstance(value, str):
        dt = h5py.string_dtype(encoding="utf-8")
        h5group.create_dataset(key, data=value, dtype=dt)
        return

    if np.isscalar(value):
        h5group.create_dataset(key, data=value)
        return

    arr = np.asarray(value)
    h5group.create_dataset(key, data=arr)


def save_dict_h5(
    h5_path: str,
    data: dict[str, Any],
    root_name: str = "out",
    overwrite: bool = True,
) -> None:
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    if os.path.exists(h5_path):
        if overwrite:
            os.remove(h5_path)
        else:
            raise FileExistsError(h5_path)

    with h5py.File(h5_path, "w") as f:
        root = f.create_group(root_name)
        for key, value in data.items():
            _write_h5_item(root, key, value)