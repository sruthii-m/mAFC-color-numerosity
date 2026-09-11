"""
Compare CNN confidence to human confidence on the mAFC color-dominance task,
using the one stimulus set that has both: `old test/` images and Expt3's
fitted human behavioral data (dataForModeling_Expt3.mat).

Mapping images -> human data
-----------------------------
Expt3's behavioral data indexes trials by (condition, config):
  - `conditions` (12x3): magnitude triples, each already sorted descending
    (rank1 >= rank2 >= rank3).
  - `configs` (6x3): permutations of {1,2,3}. A config row [c_R, c_G, c_B]
    says "channel X receives the value of rank c_X" -- i.e.
        R = values[c_R - 1]; G = values[c_G - 1]; B = values[c_B - 1]
    (validated by reproducing two known folders: condition [98,84,72] +
    config [1,3,2] -> R=98,G=72,B=84, which matches folder "98-84-72", and
    the analogous check against "84-72-62").
  - `resp[condition, config]` / `conf[condition, config]` are per-subject
    arrays of trial responses / confidence ratings (1-4 scale) for that
    exact (R, G, B) stimulus, across all 15 subjects.

Each `old test/<R>-<B>-<G>/` folder name is written in R-B-G order (see
src/train_cnn.py for how this was established from raw pixel counts). We
recover the *actual* (R, G, B) per folder by counting pixels directly rather
than trusting the name, since one folder ("98-84-48") turned out to be
mislabeled/corrupted -- its images don't contain the color counts its name
claims, so it's dropped from the comparison automatically by the pixel-count
check below.
"""

import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

OLD_TEST_DIR = (
    "/Users/sruthi/iCloud Drive (Archive)/Desktop/Desktop - sruthi’s MacBook Air/mAFC/"
    "training + generating codes/old test"
)
EXPT3_MAT = (
    "/Users/sruthi/iCloud Drive (Archive)/Desktop/Desktop - sruthi’s MacBook Air/mAFC/"
    "training + generating codes/Attempt6/conf_model/dataForModeling_Expt3.mat"
)
PURE_R, PURE_G, PURE_B = (255, 0, 0), (0, 128, 0), (0, 0, 255)


def measured_rgb(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        return None
    arr = np.array(Image.open(files[0]).convert("RGB"))
    colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    d = dict(zip((tuple(c) for c in colors), counts))
    return int(d.get(PURE_R, 0)), int(d.get(PURE_G, 0)), int(d.get(PURE_B, 0))


def find_condition_config(rgb, conditions, configs):
    """Return (condition_idx, config_idx) that reproduces `rgb`, or None."""
    r, g, b = rgb
    for ci, cond in enumerate(conditions):
        values = sorted(cond, reverse=True)
        for gi, cfg in enumerate(configs):
            cr, cg, cb = cfg
            if (values[cr - 1], values[cg - 1], values[cb - 1]) == (r, g, b):
                return ci, gi
    return None


def main():
    d = sio.loadmat(EXPT3_MAT, struct_as_record=False, squeeze_me=True)
    info = d["info"]
    data = d["data"]  # 15 subjects
    conditions = info.conditions
    configs = info.configs

    matched = []
    for folder_name in sorted(os.listdir(OLD_TEST_DIR)):
        folder = os.path.join(OLD_TEST_DIR, folder_name)
        if not os.path.isdir(folder):
            continue
        rgb = measured_rgb(folder)
        if rgb is None:
            continue
        hit = find_condition_config(rgb, conditions, configs)
        if hit is None:
            print(f"SKIP {folder_name}: measured RGB={rgb} matches no Expt3 condition "
                  f"(folder name doesn't reflect its actual contents)")
            continue
        ci, gi = hit

        # Pool trial-level resp/conf across all subjects for this stimulus.
        all_resp, all_conf = [], []
        for sub in data:
            r_cell = sub.resp[ci, gi]
            c_cell = sub.conf[ci, gi]
            all_resp.extend(np.atleast_1d(r_cell).tolist())
            all_conf.extend(np.atleast_1d(c_cell).tolist())
        all_resp = np.array(all_resp)
        all_conf = np.array(all_conf)

        # resp encodes which color (by rank/config position) was chosen;
        # correct == choosing the position holding the largest value.
        cr, cg, cb = configs[gi]
        largest_rank_channel = 1 + int(np.argmin([cr, cg, cb]))  # rank 1 is largest
        human_acc = float(np.mean(all_resp == largest_rank_channel))
        human_conf_raw = float(np.mean(all_conf))          # 1-4 scale
        human_conf_norm = (human_conf_raw - 1) / 3          # -> 0-1 scale

        matched.append({
            "folder": folder_name,
            "rgb_measured": rgb,
            "condition_idx": int(ci),
            "config_idx": int(gi),
            "n_trials": int(len(all_resp)),
            "human_accuracy": human_acc,
            "human_confidence_raw_1to4": human_conf_raw,
            "human_confidence_0to1": human_conf_norm,
        })
        print(f"{folder_name}: RGB={rgb} -> Expt3 condition {ci}/config {gi}, "
              f"n_trials={len(all_resp)}, human_acc={human_acc:.3f}, "
              f"human_conf={human_conf_raw:.2f}/4")

    out_path = Path(__file__).resolve().parent / "expt3_human_matched.json"
    with open(out_path, "w") as f:
        json.dump(matched, f, indent=2)
    print(f"\nMatched {len(matched)} folders to Expt3 human data -> {out_path}")


if __name__ == "__main__":
    main()
