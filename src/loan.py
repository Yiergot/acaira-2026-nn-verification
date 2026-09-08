"""Stretch example: a loan-approval network and fairness as a *verifiable* property.

Builds (all synthetic, educational):
  data/loan-synthetic.csv            applicants with a historically biased approval label
  models/loan-v1.onnx                trained WITH the protected attribute as an input (learns the bias)
  models/loan-v2.onnx                same architecture, protected attribute removed from the model
  models/loan-twin-v1.onnx, -v2      "twin" self-composition: two weight-shared copies as ONE network,
                                     input [1,12] = applicant A ++ applicant B, output [1,4] = scores A ++ scores B
Output index 0 = deny, 1 = approve (training label 1 = approved).
  data/loan-eval-applicants.idx/.csv small evaluation set for local-robustness properties

Why the twin? Marabou-style verifiers evaluate a network once per query, so "f(x) == f(x')" cannot be stated
directly. Stacking two copies block-diagonally turns the hyperproperty into an ordinary property of one network
(see Athavale et al., CAV 2024, "Verifying Global Two-Safety Properties in Neural Networks with Confidence").
"""
from __future__ import annotations

from pathlib import Path

import idx2numpy
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["income", "debt_ratio", "history_years", "defaults", "age", "protected"]
UNITS = {"income": "k£ per year", "debt_ratio": "fraction of income", "history_years": "years", "defaults": "count", "age": "years", "protected": "0/1"}
LO = np.array([10, 0.0, 0, 0, 18, 0], dtype=np.float32)
HI = np.array([200, 1.0, 40, 5, 90, 1], dtype=np.float32)


def make_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    protected = (rng.random(n) < 0.4).astype(float)
    # proxy: the protected group has lower incomes on average (this is the *society*, not the model)
    income = np.clip(rng.lognormal(np.where(protected == 1, 3.35, 3.65), 0.45, n), 10, 200)
    debt_ratio = np.clip(rng.beta(2, 5, n), 0, 1)
    history = np.clip(rng.gamma(3, 3, n), 0, 40)
    defaults = rng.binomial(5, 0.08, n).astype(float)
    age = np.clip(rng.normal(42, 13, n), 18, 90)
    X = np.stack([income, debt_ratio, history, defaults, age, protected], axis=1)
    # 30 % uniform samples over the whole plausible box so the model has seen the regions a verifier explores
    n_uni = int(0.3 * n)
    U = rng.uniform(LO, HI, size=(n_uni, 6)); U[:, 3] = np.round(U[:, 3]); U[:, 5] = np.round(U[:, 5])
    X = np.concatenate([X, U]); protected = X[:, 5]; income = X[:, 0]; debt_ratio = X[:, 1]; history = X[:, 2]; defaults = X[:, 3]
    n = len(X)
    # creditworthiness (what a fair lender would use) ...
    merit = 0.035 * income - 4.0 * debt_ratio + 0.06 * history - 1.2 * defaults + rng.normal(0, 0.35, n)
    # ... plus a *historical* penalty applied to the protected group: the labels are biased
    thr = float(np.quantile(merit, 0.55))  # roughly 45 % of applicants are creditworthy
    biased = merit - 0.9 * protected
    y_biased = (biased > thr).astype(np.int64)
    y_fair = (merit > thr).astype(np.int64)
    df = pd.DataFrame(np.round(X, 2), columns=FEATURES)
    df["approved_historical"] = y_biased
    df["approved_merit_only"] = y_fair
    return df


class LoanNet(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def train(X: np.ndarray, y: np.ndarray, *, drop_protected: bool, seed=11, epochs=80) -> nn.Module:
    torch.manual_seed(seed)
    X = X.copy()
    if drop_protected:
        X[:, 5] = 0.0  # the model never sees the attribute
    mean, std = X.mean(0), X.std(0) + 1e-6
    if drop_protected:
        std[5] = 1.0; mean[5] = 0.0
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32); yt = torch.tensor(y)
    m = LoanNet(); opt = torch.optim.Adam(m.parameters(), 3e-3); lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), 256):
            idx = perm[i:i + 256]; opt.zero_grad(); lf(m(Xn[idx]), yt[idx]).backward(); opt.step()
    # fold normalisation into layer 1
    with torch.no_grad():
        l1 = m.net[0]; W = l1.weight.clone(); b = l1.bias.clone()
        l1.weight.copy_(W / torch.tensor(std)); l1.bias.copy_(b - (W * torch.tensor(mean / std)).sum(1))
        if drop_protected:
            l1.weight[:, 5] = 0.0  # provably independent of the protected column
        acc = (m(torch.tensor(X if not drop_protected else X, dtype=torch.float32)).argmax(1) == yt).float().mean().item()
    print(f"  train accuracy {acc:.4f} (drop_protected={drop_protected})")
    return m.eval()


def export(m: nn.Module, path: Path):
    torch.onnx.export(m, torch.zeros(1, 6), str(path), input_names=["applicant"], output_names=["scores"], opset_version=13, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


def layers(m: nn.Module):
    return [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32)) for l in m.net if isinstance(l, nn.Linear)]


def build_twin(m: nn.Module, path: Path):
    """Two weight-shared copies as one block-diagonal ONNX graph: Gemm/Relu only (Marabou-friendly)."""
    Ls = layers(m)
    nodes, inits = [], []
    prev = "pair"
    for k, (W, b) in enumerate(Ls):
        Wt = np.zeros((2 * W.shape[0], 2 * W.shape[1]), dtype=np.float32)
        Wt[: W.shape[0], : W.shape[1]] = W; Wt[W.shape[0]:, W.shape[1]:] = W
        bt = np.concatenate([b, b]).astype(np.float32)
        inits += [numpy_helper.from_array(Wt, f"W{k}"), numpy_helper.from_array(bt, f"b{k}")]
        out = "scores_pair" if k == len(Ls) - 1 else f"h{k}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{k}", f"b{k}"], [f"g{k}" if k < len(Ls) - 1 else out], transB=1))
        if k < len(Ls) - 1:
            nodes.append(helper.make_node("Relu", [f"g{k}"], [out]))
        prev = out
    g = helper.make_graph(nodes, "loan_twin", [helper.make_tensor_value_info("pair", TensorProto.FLOAT, [1, 12])],
                          [helper.make_tensor_value_info("scores_pair", TensorProto.FLOAT, [1, 4])], initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="acaira-2026-nn-verification")
    model.ir_version = 8
    onnx.checker.check_model(model); onnx.save(model, str(path))


def check_twin(single: Path, twin: Path, rng: np.random.Generator):
    s1 = ort.InferenceSession(str(single), providers=["CPUExecutionProvider"]); s2 = ort.InferenceSession(str(twin), providers=["CPUExecutionProvider"])
    X = rng.uniform(LO, HI, size=(1000, 6)).astype(np.float32); Xp = rng.uniform(LO, HI, size=(1000, 6)).astype(np.float32)
    err = 0.0
    for a, b in zip(X, Xp):
        fa = s1.run(None, {"applicant": a[None]})[0][0]; fb = s1.run(None, {"applicant": b[None]})[0][0]
        ft = s2.run(None, {"pair": np.concatenate([a, b])[None]})[0][0]
        err = max(err, float(np.abs(np.concatenate([fa, fb]) - ft).max()))
    print(f"  twin check max |twin(a++b) - (f(a)++f(b))| over 1000 pairs: {err:.2e}")


def main():
    rng = np.random.default_rng(2026)
    df = make_data(30000, rng)
    (ROOT / "data").mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "loan-synthetic.csv", index=False)
    X = df[FEATURES].to_numpy(np.float32); y = df["approved_historical"].to_numpy()
    print("approval rate in historical labels by group:", df.groupby("protected")["approved_historical"].mean().round(3).to_dict())
    print("approval rate in merit-only labels by group:", df.groupby("protected")["approved_merit_only"].mean().round(3).to_dict())

    print("training loan-v1 (uses protected attribute)")
    m1 = train(X, y, drop_protected=False); export(m1, ROOT / "models" / "loan-v1.onnx"); build_twin(m1, ROOT / "models" / "loan-twin-v1.onnx")
    check_twin(ROOT / "models" / "loan-v1.onnx", ROOT / "models" / "loan-twin-v1.onnx", rng)
    print("training loan-v2 (protected attribute removed)")
    m2 = train(X, y, drop_protected=True); export(m2, ROOT / "models" / "loan-v2.onnx"); build_twin(m2, ROOT / "models" / "loan-twin-v2.onnx")
    check_twin(ROOT / "models" / "loan-v2.onnx", ROOT / "models" / "loan-twin-v2.onnx", rng)

    # group approval rates of each model on the data (proxies survive removing the attribute)
    for name, m in (("v1", m1), ("v2", m2)):
        with torch.no_grad():
            pred = m(torch.tensor(X)).argmax(1).numpy()
        rates = pd.Series(pred).groupby(df["protected"]).mean().round(3).to_dict()
        print(f"  {name} approval rate by group on the data: {rates}")

    # small evaluation set: 8 applicants the model is confident about + 2 borderline ones (flagged)
    typical = X[: 30000]  # the realistic part of the data
    with torch.no_grad():
        logits = m2(torch.tensor(typical)).numpy()
    margin = logits[:, 1] - logits[:, 0]          # approve logit minus deny logit
    order = np.argsort(-np.abs(margin))
    approves = [i for i in order[:8000] if margin[i] > 0 and i % 5 == 0][:4]
    denies = [i for i in order[:8000] if margin[i] < 0 and i % 5 == 0][:4]
    confident = approves + denies
    borderline = list(np.argsort(np.abs(margin))[:2])
    pick = confident + borderline
    E = typical[pick]
    decision = (margin[pick] > 0).astype(np.uint8)          # True = approve
    ev = pd.DataFrame(E, columns=FEATURES)
    ev["model_v2_decision"] = np.where(decision == 1, "approve", "deny")
    ev["margin_logits"] = np.round(margin[pick], 3)
    ev["borderline"] = [False] * len(confident) + [True] * len(borderline)
    ev.to_csv(ROOT / "data" / "loan-eval-applicants.csv", index=False)
    idx2numpy.convert_to_file(str(ROOT / "data" / "loan-eval-applicants.idx"), E.astype(np.float32))
    idx2numpy.convert_to_file(str(ROOT / "data" / "loan-eval-decisions.idx"), decision.astype(np.uint8))  # class index 1 = approve, 0 = deny
    print(ev.to_string())
    print(f"wrote {len(E)} evaluation applicants")
    print("ONNX ops in loan-v1:", [n.op_type for n in onnx.load(str(ROOT / 'models' / 'loan-v1.onnx')).graph.node])


if __name__ == "__main__":
    main()
