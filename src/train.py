"""Train the small triage networks and export them to ONNX with normalisation folded into layer 1.

  python train.py --variant v1   # flawed labels (the "3 in a single parameter" rule is missing); ordinary training
  python train.py --variant v2   # correct NEWS2 bands; ordinary training
  python train.py --variant v3   # correct NEWS2 bands + the specification's hard-safety properties as extra
                                 # hinge losses ("property-driven training"; Vehicle can generate such losses
                                 # automatically from a .vcl file, here we write them by hand for transparency)
                                 # + counterexample-guided repair: the verifier is run between fine-tuning
                                 # rounds and every counterexample it finds goes back into training.
                                 # Needs `vehicle` and `Marabou` (env VEHICLE_BIN / MARABOU_BIN override PATH).

The exported ONNX takes the raw clinical vector [rr, spo2, sbp, pulse, temp, o2, cvpu] with shape [1, 7]
and returns three scores (low, medium, high). Because normalisation is folded into the first Gemm,
the Vehicle specification can be written in clinical units and Marabou's ONNX parser only ever sees
Gemm and Relu nodes.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from news2 import FEATURES, VALID_RANGE, triage_class

ROOT = Path(__file__).resolve().parents[1]


class TriageNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def boundary_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """Extra samples hugging the NEWS2 thresholds, so the model learns the rule edges."""
    lo = np.array([VALID_RANGE[f][0] for f in FEATURES]); hi = np.array([VALID_RANGE[f][1] for f in FEATURES])
    X = rng.uniform(lo, hi, size=(n, 7))
    X[:, 5:] = np.round(X[:, 5:])
    edges = {0: [8, 11, 20, 24], 1: [91, 93, 95], 2: [90, 100, 110, 219], 3: [40, 50, 90, 110, 130], 4: [35.0, 36.0, 38.0, 39.0]}
    for col, es in edges.items():
        pick = rng.random(n) < 0.5
        e = rng.choice(es, size=n)
        X[pick, col] = e[pick] + rng.uniform(-1.5, 1.5, size=pick.sum())
    X = np.clip(X, lo, hi)
    return X


# The regions quantified over by the hard-safety properties in specs/triage.vcl (clinical units).
# Sampling them densely during training is "specification-guided data augmentation": the labels still
# come from the NEWS2 rule, we simply make sure the model has seen the whole region the verifier will explore.
SPEC_REGIONS = {
    #  name           : (box overrides, what the property demands there)
    "hypoxia":        ({"spo2": (50, 91), "o2": (0, 0), "cvpu": (0, 0)}, "never_low"),
    "hypoxiaOnO2":    ({"spo2": (50, 91), "o2": (1, 1), "cvpu": (0, 0)}, "never_low"),
    "shock":          ({"sbp": (40, 90), "o2": (0, 0), "cvpu": (0, 0)}, "never_low"),
    "notAlert":       ({"o2": (0, 0), "cvpu": (1, 1)}, "never_low"),
    "normalVitals":   ({"rr": (13, 19), "spo2": (97, 100), "sbp": (115, 210), "pulse": (55, 88), "temp": (36.3, 37.8), "o2": (0, 0), "cvpu": (0, 0)}, "always_low"),
}


def property_loss(scores: torch.Tensor, kind: str, margin: float) -> torch.Tensor:
    """Hinge version of the property: 0 once the property holds with `margin` logits to spare."""
    s_low = scores[:, 0]; s_other = scores[:, 1:].max(1).values
    if kind == "never_low":
        return torch.relu(margin - (s_other - s_low)).mean()
    return torch.relu(margin - (s_low - s_other)).mean()


def violations(model, mean, std, rng, n=40000) -> dict:
    """Empirical property violations on fresh uniform samples of each region (cheap sanity check; the proof is Marabou's job)."""
    out = {}
    with torch.no_grad():
        for name, (region, kind) in SPEC_REGIONS.items():
            Xs = region_samples(n, rng, region, edge_fraction=0.5, slack=True).astype(np.float32)
            pred = model(torch.tensor((Xs - mean) / std)).argmax(1).numpy()
            out[name] = int((pred == 0).sum() if kind == "never_low" else (pred != 0).sum())
    return out


# NEWS2 thresholds are integers (temperature: one decimal) but the verifier explores real numbers. The class of a
# real-valued input is the class of the observation it would be *recorded* as, by truncation (see news2.py), so the
# true label changes at the next integer: a spec region "sbp <= 90" is really safe up to 90.99. We train the property
# loss on the region widened by half of that gap, which gives the network margin exactly where the verifier probes
# without asking it for an infinitely sharp step at the label change.
TRAIN_SLACK = {"rr": 0.5, "spo2": 0.5, "sbp": 0.5, "pulse": 0.5, "temp": 0.05}


def region_samples(n: int, rng: np.random.Generator, region: dict, edge_fraction: float = 0.0, slack: bool = False) -> np.ndarray:
    """Uniform samples of a region; optionally a fraction of them within 2 units of each constrained edge
    (that is where a learned decision boundary is most likely to be a hair on the wrong side)."""
    lo = np.array([VALID_RANGE[f][0] for f in FEATURES]); hi = np.array([VALID_RANGE[f][1] for f in FEATURES])
    for f, (a, b) in region.items():
        i = FEATURES.index(f)
        if slack and f in TRAIN_SLACK:
            a = max(VALID_RANGE[f][0], a - TRAIN_SLACK[f]) if a > VALID_RANGE[f][0] else a
            b = min(VALID_RANGE[f][1], b + TRAIN_SLACK[f]) if b < VALID_RANGE[f][1] else b
        lo[i] = a; hi[i] = b
    X = rng.uniform(lo, hi, size=(n, 7))
    n_edge = int(n * edge_fraction)
    if n_edge:
        for f, (a, b) in region.items():
            i = FEATURES.index(f)
            if i >= 5 or b - a < 4:
                continue
            width = 0.2 if f == "temp" else 2.0
            rows = rng.choice(n, size=n_edge // 2, replace=False)
            X[rows, i] = b - rng.uniform(0, width, size=len(rows))
            rows = rng.choice(n, size=n_edge // 2, replace=False)
            X[rows, i] = a + rng.uniform(0, width, size=len(rows))
    X[:, 5:] = np.round(X[:, 5:])
    return X


HARD_PROPERTIES = ["hypoxiaNeverLow", "hypoxiaOnOxygenNeverLow", "shockNeverLow", "notAlertNeverLow", "normalVitalsAlwaysLow"]


def export_folded(model: nn.Module, mean: np.ndarray, std: np.ndarray, out: Path) -> None:
    """Export a copy of `model` whose first layer absorbs the input normalisation: W'x + b' = W((x-mean)/std) + b."""
    m = copy.deepcopy(model).eval()
    l1 = m.net[0]
    with torch.no_grad():
        W = l1.weight.clone(); b = l1.bias.clone()
        l1.weight.copy_(W / torch.tensor(std)); l1.bias.copy_(b - (W * torch.tensor(mean / std)).sum(1))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, torch.zeros(1, 7), str(out), input_names=["patient"], output_names=["scores"], opset_version=13, dynamo=False)


def verify_hard_properties(onnx_path: Path, spec: Path = ROOT / "specs" / "triage.vcl") -> dict:
    """Run the real verifier on each hard-safety property. Returns {property: ("verified"|"falsified"|"other", counterexample|None)}.
    Uses $VEHICLE_BIN (default `vehicle`) and $MARABOU_BIN (default `Marabou`) so it works on a Mac with an x86_64
    venv as well as inside the Docker/Colab environment where both are on PATH."""
    vehicle = os.environ.get("VEHICLE_BIN", "vehicle"); marabou = os.environ.get("MARABOU_BIN", "Marabou")
    results = {}
    for prop in HARD_PROPERTIES:
        cmd = [vehicle, "verify", "-s", str(spec), "-n", f"triage:{onnx_path}", "-y", prop, "--solver", marabou, "-a", "--timeout=60"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout + "\n"
        out = re.sub(r"\x1b\[[0-9;]*m", "", out).replace("\r", "\n")
        if "proved no counterexample" in out:
            results[prop] = ("verified", None)
        elif "found a counterexample" in out:
            m = re.search(r"x:\s*\[([^\]]*)\]", out)
            ce = np.array([float(v) for v in m.group(1).split(",")], dtype=np.float32) if m else None
            results[prop] = ("falsified", ce)
        else:
            results[prop] = ("other", None)
    return results


def train(variant: str, seed: int, epochs: int, hidden: int, out: Path, cegis_rounds: int = 0) -> None:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    df = pd.read_csv(ROOT / "data" / "triage-synthetic.csv")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    if variant == "v1":
        label_fn = lambda A: triage_class(A, single_parameter_rule=False)
    elif variant == "v2":
        label_fn = triage_class
    else:  # v3: conservative labels — worst NEWS2 score within +/- 1 unit of measurement uncertainty
        label_fn = lambda A: triage_class(A, uncertainty=1.0)
    y = df["label_v1"].to_numpy() if variant == "v1" else label_fn(df[FEATURES].to_numpy(dtype=np.float32))
    # every variant sees the same inputs (incl. extra samples near thresholds); only labels / losses differ
    Xb = boundary_samples(len(X) // 2, rng).astype(np.float32)
    X = np.concatenate([X, Xb]); y = np.concatenate([y, label_fn(Xb)])
    use_property_loss = variant == "v3"
    margin, lam = 2.0, 4.0

    mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = torch.tensor((X - mean) / std); yt = torch.tensor(y)
    model = TriageNet(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xn); bs = 256
    regions = list(SPEC_REGIONS.values())
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xn[idx]), yt[idx])
            if use_property_loss:
                ploss = 0.0
                for region, kind in regions:
                    Xr = torch.tensor(((region_samples(bs // 2, rng, region, edge_fraction=0.5, slack=True) - mean) / std).astype(np.float32))
                    ploss = ploss + property_loss(model(Xr), kind, margin)
                loss = loss + lam * ploss
            loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xn).argmax(1) == yt).float().mean().item()
            print(f"[{variant}] epoch {ep:3d} loss {loss.item():.4f} train-acc {acc:.4f}")
    model.eval()
    print(f"[{variant}] empirical property violations on 40k fresh samples per region: {violations(model, mean, std, rng)}")

    # Counterexample-guided repair (v3 only): ask the *verifier* where the properties fail, add those points
    # (and a small neighbourhood, labelled by the clinical rule) to the training set, fine-tune, repeat.
    export_folded(model, mean, std, out)
    if cegis_rounds:
        for rnd in range(cegis_rounds):
            res = verify_hard_properties(out)
            summary = {k: v[0] for k, v in res.items()}
            print(f"[{variant}] repair round {rnd}: verifier says {summary}")
            ces = [ce for (st, ce) in res.values() if st == "falsified" and ce is not None]
            if not ces:
                print(f"[{variant}] all hard-safety properties verified after {rnd} repair round(s)")
                break
            Xc = [] if rnd == 0 else [Xc_all]
            for ce in ces:
                print(f"[{variant}]   counterexample: {np.round(ce, 3).tolist()}")
                jitter = rng.uniform(-0.4, 0.4, size=(2000, 7)); jitter[:, 4] *= 0.1; jitter[:, 5:] = 0
                Xc.append(np.clip(ce[None, :] + jitter, [VALID_RANGE[f][0] for f in FEATURES], [VALID_RANGE[f][1] for f in FEATURES]))
            Xc_all = np.concatenate(Xc).astype(np.float32); Xc = Xc_all; yc = label_fn(Xc)
            Xa = np.concatenate([X, Xc]); ya = np.concatenate([y, yc])
            Xan = torch.tensor((Xa - mean) / std); yat = torch.tensor(ya)
            Xcn = torch.tensor((Xc - mean) / std)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            model.train()
            ycn = torch.tensor(yc)
            for ep in range(15):
                perm = torch.randperm(len(Xan))
                for i in range(0, len(Xan), bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    sc = model(Xcn)
                    # hinge on the counterexample neighbourhoods: correct class must win by `margin` logits
                    other = sc.clone(); other[torch.arange(len(ycn)), ycn] = -1e9
                    ce_hinge = torch.relu(margin - (sc[torch.arange(len(ycn)), ycn] - other.max(1).values)).mean()
                    loss = loss_fn(model(Xan[idx]), yat[idx]) + 2.0 * loss_fn(sc, ycn) + 3.0 * ce_hinge
                    for region, kind in regions:
                        Xr = torch.tensor(((region_samples(bs // 2, rng, region, edge_fraction=0.5, slack=True) - mean) / std).astype(np.float32))
                        loss = loss + lam * property_loss(model(Xr), kind, margin)
                    loss.backward(); opt.step()
            model.eval()
            export_folded(model, mean, std, out)
        else:
            print(f"[{variant}] WARNING: repair budget exhausted; some properties may still fail")
    with torch.no_grad():
        acc_raw = (model(Xn).argmax(1) == yt).float().mean().item()
        correct = torch.tensor(df["label"].to_numpy())
        Xd = torch.tensor((df[FEATURES].to_numpy(dtype=np.float32) - mean) / std)
        acc_vs_correct = (model(Xd).argmax(1) == correct).float().mean().item()
    print(f"[{variant}] accuracy on its own training labels: {acc_raw:.4f}; on CORRECT NEWS2 bands: {acc_vs_correct:.4f}")
    m = onnx.load(str(out)); onnx.checker.check_model(m)
    print(f"[{variant}] ONNX ops: {[nd.op_type for nd in m.graph.node]}  input shape {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    xs = X[:1000]
    ort_out = np.stack([sess.run(None, {"patient": x[None]})[0][0] for x in xs])
    with torch.no_grad():
        pt_out = model(torch.tensor((xs - mean) / std)).numpy()
    print(f"[{variant}] max |onnxruntime - pytorch| on 1000 samples: {np.abs(ort_out - pt_out).max():.2e}")
    print(f"[{variant}] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--cegis", type=int, default=None, help="max counterexample-guided repair rounds (default: 8 for v3, 0 otherwise)")
    args = ap.parse_args()
    rounds = args.cegis if args.cegis is not None else (8 if args.variant == "v3" else 0)
    train(args.variant, args.seed, args.epochs, args.hidden, ROOT / "models" / f"triage-{args.variant}.onnx", rounds)
