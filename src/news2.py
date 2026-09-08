"""NEWS2 (National Early Warning Score 2, Royal College of Physicians 2017) — Scale 1 only.

This is a plain re-statement of the published scoring thresholds so that we can generate a *synthetic*,
educational triage dataset. It is NOT a clinical tool. Scale 2 (patients with hypercapnic respiratory
failure), age and pregnancy adjustments are deliberately omitted.

Source: RCP, "National Early Warning Score (NEWS) 2", Charts 1 and 4,
https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/
"""
from __future__ import annotations

import numpy as np

# Column order used everywhere (data files, ONNX input, Vehicle specification).
FEATURES = ["rr", "spo2", "sbp", "pulse", "temp", "o2", "cvpu"]
UNITS = {
    "rr": "breaths/min",
    "spo2": "%",
    "sbp": "mmHg",
    "pulse": "beats/min",
    "temp": "°C",
    "o2": "1 = supplemental oxygen",
    "cvpu": "1 = new confusion / not alert",
}
# Physiologically plausible box used both for sampling and for the `validPatient` predicate in the spec.
VALID_RANGE = {
    "rr": (4.0, 60.0),
    "spo2": (50.0, 100.0),
    "sbp": (40.0, 250.0),
    "pulse": (20.0, 200.0),
    "temp": (30.0, 43.0),
    "o2": (0.0, 1.0),
    "cvpu": (0.0, 1.0),
}
CLASSES = ["low", "medium", "high"]


def score_rr(x):
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], default=3)


def score_spo2(x):
    return np.select([x <= 91, x <= 93, x <= 95], [3, 2, 1], default=0)


def score_sbp(x):
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], default=3)


def score_pulse(x):
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], default=3)


def score_temp(x):
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], default=2)


def truncate_to_recording_resolution(X: np.ndarray) -> np.ndarray:
    """NEWS2 is defined on *recorded* observations: whole numbers for respiration rate, SpO2, blood pressure
    and pulse, one decimal place for temperature, and yes/no flags. A verifier, however, reasons about real
    numbers. We therefore need a convention for the class of a real-valued input, and we adopt: **a reading is
    recorded by truncation** (90.7 mmHg is charted as 90). Consequence: the true class changes exactly at the
    integers (e.g. between 90.99 and 91.0), so a specification threshold such as `sbp <= 90` sits a full unit
    away from the nearest label change. That gap is what makes the properties learnable *and* provable: a
    network only has to place its decision boundary somewhere inside (90, 91), not at an infinitely sharp step.
    (Rounding instead of truncating would put the label change at 90.5 and leave only half a unit of slack.)
    """
    X = np.array(X, dtype=float, copy=True)
    X[:, :4] = np.floor(X[:, :4] + 1e-6)
    X[:, 4] = np.floor(X[:, 4] * 10 + 1e-6) / 10
    X[:, 5:] = np.round(X[:, 5:])
    return X


def component_scores(X: np.ndarray, uncertainty: float = 0.0) -> np.ndarray:
    """Per-parameter NEWS2 points for an (n, 7) array in FEATURES order. Returns (n, 7) ints.

    `uncertainty` > 0 gives a *conservative* scoring: each continuous parameter is scored at its recorded value
    and at +/- `uncertainty` units (+/- uncertainty/10 degrees for temperature) and the worst score is kept.
    This is how the property-driven model (triage-v3) is trained: it deliberately triggers one unit early, so
    that every specification threshold (e.g. `sbp <= 90`) sits two units away from the model's own decision
    boundary instead of one. The price is a little over-triage exactly at the thresholds.
    """
    if uncertainty > 0:
        deltas = np.array([uncertainty, uncertainty, uncertainty, uncertainty, uncertainty / 10, 0, 0])
        return np.maximum.reduce([_component_scores(X - deltas), _component_scores(X), _component_scores(X + deltas)])
    return _component_scores(X)


def _component_scores(X: np.ndarray) -> np.ndarray:
    X = truncate_to_recording_resolution(X)
    return np.stack(
        [
            score_rr(X[:, 0]),
            score_spo2(X[:, 1]),
            score_sbp(X[:, 2]),
            score_pulse(X[:, 3]),
            score_temp(X[:, 4]),
            np.where(X[:, 5] >= 0.5, 2, 0),  # supplemental oxygen
            np.where(X[:, 6] >= 0.5, 3, 0),  # consciousness: any of C, V, P, U scores 3
        ],
        axis=1,
    )


def aggregate(X: np.ndarray) -> np.ndarray:
    return component_scores(X).sum(axis=1)


def triage_class(X: np.ndarray, *, single_parameter_rule: bool = True, uncertainty: float = 0.0) -> np.ndarray:
    """Map NEWS2 to three urgency classes.

    0 = low    (aggregate 0-4)               -> routine ward monitoring
    1 = medium (aggregate 5-6, or any single parameter scoring 3) -> urgent clinical review
    2 = high   (aggregate >= 7)              -> emergency response

    `single_parameter_rule=False` reproduces a *deliberately wrong* labelling that ignores the
    "3 in any single parameter" trigger. It is used to train the flawed model `triage-v1`.
    """
    comp = component_scores(X, uncertainty)
    agg = comp.sum(axis=1)
    cls = np.where(agg >= 7, 2, np.where(agg >= 5, 1, 0))
    if single_parameter_rule:
        cls = np.where((cls == 0) & (comp.max(axis=1) >= 3), 1, cls)
    return cls.astype(np.int64)
