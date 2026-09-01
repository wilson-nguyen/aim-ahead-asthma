"""
export_historical_split_arrays.py — make historical split verification
clean-clone reproducible.

The frozen pre-correction snapshots (../aim-ahead-asthma-FROZEN-*/) hold the
outcome and survey-weight arrays that prove participant membership and
ordering never changed. Those snapshots live outside the repository, so a
clean clone could not run the historical checks (external review, 31 Aug).

This one-time exporter extracts exactly the arrays the verifier compares
(y_train/val/test, sw_train/val/test per frozen run) into small committed
.npz files. Run on the machine that has the frozen snapshots:

    python export_historical_split_arrays.py

Writes: outputs/historical_split_arrays/<runid>.npz  (commit these)
"""
import glob
import os
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "historical_split_arrays")


def main():
    # shims so old __main__ pickles resolve
    import __main__ as _m
    for _name in ("NHANESCleaner", "ClinicalFeatureEngineer", "ProtectedSelectKBest"):
        if not hasattr(_m, _name):
            setattr(_m, _name, type(_name, (), {}))

    pkls = sorted(glob.glob(os.path.join(
        HERE, "..", "aim-ahead-asthma-FROZEN-*", "notebooks",
        "tuning_results_*", "preprocessed_data.pkl")))
    if not pkls:
        sys.exit("No frozen snapshots found (../aim-ahead-asthma-FROZEN-*). "
                 "Run this on the machine that has them.")
    os.makedirs(OUT, exist_ok=True)
    for fp in pkls:
        run = os.path.basename(os.path.dirname(fp)).replace("tuning_results_", "")
        art = joblib.load(fp)
        arrays = {}
        for part in ("train", "val", "test"):
            arrays[f"y_{part}"] = np.asarray(art[f"y_{part}"], dtype=float)
            arrays[f"sw_{part}"] = np.asarray(art[f"sw_{part}"], dtype=float)
        path = os.path.join(OUT, f"{run}.npz")
        np.savez_compressed(path, **arrays)
        print(f"exported {run}: " + ", ".join(
            f"{k}={v.shape[0]}" for k, v in arrays.items() if k.startswith("y_"))
            + f" -> {path}")


if __name__ == "__main__":
    main()
