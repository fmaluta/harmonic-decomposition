import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.io import load_data, save_dict_h5
from src.masking import build_smask
from src.flow_decomposition import flow_decomposition

DATA = load_data("h5", h5_path=os.path.join("..", "data", "dataset_demo.h5"))

DATA["S"] = build_smask(DATA["s"], DATA["X"], DATA["Z"], freeSurf=None)

META = {
    "rpm": 110,
    "nblades": 4,
    "name": "DES 19cm",
}

OPT = {
    "compute": {
        "show": True,
        "save_figs": True,
    },
    "harmonics": {
        "min_frac": 0.2,
        "keep_shaft": False,
    },
}

out = flow_decomposition(DATA, META, OPT)

save_dict_h5(
    os.path.join("..", "data", "decomposition_demo.h5"),
    out,
    root_name="out",
    overwrite=True,
)