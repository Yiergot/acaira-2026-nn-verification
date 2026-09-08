"""Generate the synthetic triage dataset used in the hands-on.

Two label columns are produced from the same vitals:
  * `label`     — correct NEWS2 bands (used for triage-v2)
  * `label_v1`  — bands computed WITHOUT the "3 in a single parameter" rule (used for triage-v1)

Sampling mixes (a) a "typical ward" distribution and (b) uniform samples over the whole plausible box,
because a verifier reasons about the whole box, not just the typical distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from news2 import FEATURES, VALID_RANGE, triage_class


def sample(n: int, rng: np.random.Generator, uniform_fraction: float = 0.35) -> np.ndarray:
    n_uni = int(n * uniform_fraction)
    n_typ = n - n_uni
    lo = np.array([VALID_RANGE[f][0] for f in FEATURES])
    hi = np.array([VALID_RANGE[f][1] for f in FEATURES])

    # (a) typical ward patients
    rr = rng.normal(17, 4, n_typ)
    spo2 = np.where(
        rng.random(n_typ) < 0.70,
        rng.uniform(96, 100, n_typ),
        np.where(rng.random(n_typ) < 0.8, rng.uniform(88, 96, n_typ), rng.uniform(75, 88, n_typ)),
    )
    sbp = rng.normal(125, 22, n_typ)
    pulse = rng.normal(82, 18, n_typ)
    temp = rng.normal(37.0, 0.8, n_typ)
    o2 = (rng.random(n_typ) < 0.15).astype(float)
    cvpu = (rng.random(n_typ) < 0.05).astype(float)
    typical = np.stack([rr, spo2, sbp, pulse, temp, o2, cvpu], axis=1)

    # (b) uniform over the plausible box (flags rounded to 0/1)
    uni = rng.uniform(lo, hi, size=(n_uni, len(FEATURES)))
    uni[:, 5:] = np.round(uni[:, 5:])

    X = np.concatenate([typical, uni], axis=0)
    X = np.clip(X, lo, hi)
    X[:, 5:] = np.round(X[:, 5:])
    # Inputs stay continuous (the model must behave on every real-valued input the verifier can pick);
    # the labels come from `triage_class`, which rounds to recording resolution first.
    X = np.round(X, 2)
    rng.shuffle(X)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "triage-synthetic.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = sample(args.n, rng)
    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = triage_class(X, single_parameter_rule=True)
    df["label_v1"] = triage_class(X, single_parameter_rule=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print("class balance (correct labels):", np.bincount(df['label']).tolist())
    print("class balance (v1 labels):     ", np.bincount(df['label_v1']).tolist())
    print("rows where the two labellings disagree:", int((df['label'] != df['label_v1']).sum()))


if __name__ == "__main__":
    main()
