"""Pick a small, deliberately chosen evaluation set of synthetic patients and write it as idx files.

We pick clear examples of each class plus a couple of *boundary* patients whose measurement-noise box
straddles a NEWS2 threshold. For those, a "falsified" robustness result is the specification being
honest about the clinical rule, not a network bug — the notebook flags them.
"""
from __future__ import annotations

from pathlib import Path

import idx2numpy
import numpy as np
import onnxruntime as ort
import pandas as pd

from news2 import FEATURES, aggregate, component_scores, triage_class

ROOT = Path(__file__).resolve().parents[1]

# rr, spo2, sbp, pulse, temp, o2, cvpu, note
HAND_PICKED = [
    (16, 98, 128, 72, 36.9, 0, 0, "clear low: textbook normal"),
    (14, 97, 135, 64, 37.1, 0, 0, "clear low"),
    (18, 99, 122, 80, 36.6, 0, 0, "clear low"),
    (17, 98, 118, 76, 37.4, 0, 0, "clear low"),
    (22, 94, 105, 98, 38.4, 0, 0, "clear medium: several 1-2 point parameters (aggregate 6)"),
    (16, 90, 130, 78, 37.0, 0, 0, "clear medium: single parameter scoring 3 (SpO2 90)"),
    (16, 97, 88, 84, 36.8, 0, 0, "clear medium: single parameter scoring 3 (SBP 88)"),
    (26, 89, 92, 118, 38.6, 1, 0, "clear high: aggregate well above 7"),
    (30, 85, 84, 135, 39.4, 1, 1, "clear high: deteriorating patient"),
    (24, 93, 98, 112, 35.0, 0, 0, "clear high: hypothermic, aggregate 9"),
    (16, 92, 128, 74, 37.0, 0, 0, "BOUNDARY: SpO2 92 is 2 points; one point lower and it is a 3 (medium)"),
    (16, 97, 128, 91, 37.0, 0, 0, "BOUNDARY: pulse 91 scores 1; one beat lower scores 0 (still low either way)"),
]


def main() -> None:
    X = np.array([p[:7] for p in HAND_PICKED], dtype=np.float32)
    notes = [p[7] for p in HAND_PICKED]
    y = triage_class(X)
    y_v1 = triage_class(X, single_parameter_rule=False)
    y_v3 = triage_class(X, uncertainty=1.0)  # the conservative labelling v3 is trained on
    comp = component_scores(X)
    agg = aggregate(X)

    preds = {}
    for v in ("v1", "v2", "v3"):
        sess = ort.InferenceSession(str(ROOT / "models" / f"triage-{v}.onnx"), providers=["CPUExecutionProvider"])
        preds[v] = np.array([int(np.argmax(sess.run(None, {"patient": x[None]})[0])) for x in X])

    df = pd.DataFrame(X, columns=FEATURES)
    df["news2_aggregate"] = agg
    df["max_single_parameter"] = comp.max(axis=1)
    df["label"] = y
    df["label_v1_rule"] = y_v1
    df["label_conservative"] = y_v3
    df["pred_v1"] = preds["v1"]
    df["pred_v2"] = preds["v2"]
    df["pred_v3"] = preds["v3"]
    df["boundary"] = [n.startswith("BOUNDARY") for n in notes]
    df["note"] = notes
    out = ROOT / "data"
    df.to_csv(out / "triage-eval-patients.csv", index=False)
    idx2numpy.convert_to_file(str(out / "triage-eval-patients.idx"), X)
    idx2numpy.convert_to_file(str(out / "triage-eval-labels.idx"), y.astype(np.uint8))
    print(df.to_string())
    print(f"\nwrote {len(df)} patients to {out}/triage-eval-patients.{{idx,csv}} and labels to triage-eval-labels.idx")
    print("v1 agrees with correct label on", int((preds['v1'] == y).sum()), "/", len(y))
    print("v2 agrees with correct label on", int((preds['v2'] == y).sum()), "/", len(y))
    print("v3 agrees with correct label on", int((preds['v3'] == y).sum()), "/", len(y), "and with its conservative label on", int((preds['v3'] == y_v3).sum()), "/", len(y))


if __name__ == "__main__":
    main()
