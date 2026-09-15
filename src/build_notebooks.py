"""Generate the two Colab notebooks from Python (single source of truth). Run: python src/build_notebooks.py"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"; OUT.mkdir(exist_ok=True)

def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def code(s): return nbf.v4.new_code_cell(s.strip("\n"))

SETUP = r'''
# --- Setup (idempotent; safe to re-run) -------------------------------------------------------------
# Colab's kernel is Python 3.12 and the Marabou solver only ships wheels for Python 3.8-3.11 (x86_64),
# so the toolchain lives in a separate Python 3.11 environment created by environment/colab_bootstrap.sh.
# We never import the verifier into this kernel; we call the `vehicle` command-line tool.
import os, sys, subprocess, pathlib, shutil, time

REPO_URL = "https://github.com/Yiergot/acaira-2026-nn-verification"   # <- repository (placeholder until published)
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

here = pathlib.Path.cwd()
if (here / "specs" / "triage.vcl").exists():
    ROOT = here
elif (here.parent / "specs" / "triage.vcl").exists():
    ROOT = here.parent
else:
    if not pathlib.Path("acaira-2026-nn-verification").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    ROOT = pathlib.Path("acaira-2026-nn-verification").resolve()
os.chdir(ROOT)
print("working directory:", ROOT)

if shutil.which("vehicle") and shutil.which("Marabou"):
    print("toolchain already present:", shutil.which("vehicle"))
else:
    VENV = "/content/venv" if IN_COLAB else str(ROOT.parent / ".venv-nnv")
    t0 = time.time()
    subprocess.run(["bash", "environment/colab_bootstrap.sh"], env={**os.environ, "VENV": VENV}, check=True)
    os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
    print(f"bootstrap took {time.time() - t0:.0f} s")

# Python packages for THIS kernel (plain Python, no verifier): everything the notebooks import.
KERNEL_PACKAGES = ["numpy", "pandas", "matplotlib", "onnx", "onnxruntime", "idx2numpy"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *KERNEL_PACKAGES], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
import importlib.util
missing = [m for m in KERNEL_PACKAGES if importlib.util.find_spec(m) is None]
print("kernel packages:", ", ".join(f"{m} OK" for m in KERNEL_PACKAGES if m not in missing) + (f"  MISSING: {missing}" if missing else ""))
sys.path.insert(0, str(ROOT / "src"))
print("vehicle", subprocess.run(["vehicle", "--version"], capture_output=True, text=True).stdout.strip(),
      "| Marabou at", shutil.which("Marabou"))
print("OK - ready." if not missing else "NOT ready: re-run this cell, or `pip install` the missing packages.")
'''

HELPERS = r'''
# --- Helpers: run the verifier and read its answers ------------------------------------------------
import re, subprocess, time
import numpy as np, pandas as pd, onnxruntime as ort
import news2

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("Warning", "strict inequalit", "Unfortunately", "In order to provide", "not sound", "excluded middle",
         "issues/74", "floating point", "unexpected behaviour", "queries")   # progress bars end with "queries"

def run_vehicle(args, wall=None):
    """Run `vehicle <args>`, return (cleaned output lines, seconds). `wall` = hard wall-clock cap in seconds:
    the solver's own --timeout is not honoured on some degenerate queries, so we never wait for ever."""
    t0 = time.time()
    try:
        p = subprocess.run(["vehicle", *args], capture_output=True, text=True, timeout=wall)
        raw = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        raw = ((e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
              "\n    result: ? - wall-clock cap reached, solver killed\n"
    raw = ANSI.sub("", raw).replace("\r", "\n")
    lines = [l.rstrip() for l in raw.split("\n") if l.strip() and not any(n in l for n in NOISE)]
    return lines, time.time() - t0

def verify(spec, network, properties=(), datasets=None, parameters=None, network_name="triage", timeout=30, show=True):
    """Verify `properties` of `spec` for the ONNX file `network`. Returns a dict with statuses, counterexamples, summary."""
    args = ["verify", "-s", spec, "-n", f"{network_name}:{network}", "--solver", "Marabou", "-a", f"--timeout={timeout}"]
    for prop in properties: args += ["-y", prop]
    for k, v in (datasets or {}).items(): args += ["-d", f"{k}:{v}"]
    for k, v in (parameters or {}).items(): args += ["-p", f"{k}:{v}"]
    lines, secs = run_vehicle(args, wall=3 * timeout + 30)
    statuses, ces, summary, i = [], [], {}, 0
    while i < len(lines):
        l = lines[i]
        if "result:" in l:
            if "proved no counterexample" in l or "proved no witness" in l: statuses.append("verified"); ces.append(None)
            elif "found a counterexample" in l or "found a witness" in l:
                statuses.append("falsified")
                vec = None
                if i + 1 < len(lines) and re.match(r"\s*[a-zA-Z]+:\s*\[", lines[i + 1]):
                    txt = lines[i + 1]; j = i + 1
                    while "]" not in txt and j + 1 < len(lines): j += 1; txt += lines[j]
                    vec = [float(v) for v in re.search(r"\[(.*)\]", txt).group(1).split(",")]
                ces.append(vec)
            elif "timed out" in l or "wall-clock" in l: statuses.append("timeout"); ces.append(None)
            else: statuses.append("other"); ces.append(None)
        m = re.match(r"\s*(verified|falsified|timed-out|errored):\s*(\d+)/(\d+)", l)
        if m: summary[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        i += 1
    names = list(properties) if properties and not summary else [f"{properties[0] if properties else 'item'}!{k}" for k in range(len(statuses))]
    if show:
        glyph = {"verified": "VERIFIED  ", "falsified": "FALSIFIED ", "timeout": "TIMEOUT   ", "other": "?         "}
        for n, s, c in zip(names, statuses, ces):
            print(f"{glyph[s]} {n}" + (f"   counterexample: {np.round(c, 2).tolist()}" if c else ""))
        if summary: print("summary:", {k: f"{a}/{b}" for k, (a, b) in summary.items()})
        print(f"({secs:.1f} s)")
    return {"statuses": statuses, "counterexamples": ces, "summary": summary, "seconds": secs, "lines": lines}

_sessions = {}
def predict(model_path, x):
    """Class scores of an ONNX network for one input vector (list of floats)."""
    if model_path not in _sessions:
        _sessions[model_path] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    s = _sessions[model_path]; name = s.get_inputs()[0].name
    return s.run(None, {name: np.asarray(x, dtype=np.float32)[None]})[0][0]

def show_patient(x, models=("v1", "v2", "v3")):
    """Print a counterexample as a patient chart: values, units, NEWS2 points, and what each model advises."""
    x = np.asarray(x, dtype=float)
    pts = news2.component_scores(x[None])[0]
    rows = [(f, round(v, 2), news2.UNITS[f], int(p)) for f, v, p in zip(news2.FEATURES, x, pts)]
    print(pd.DataFrame(rows, columns=["parameter", "value", "unit", "NEWS2 points"]).to_string(index=False))
    agg = int(pts.sum()); rule = news2.CLASSES[int(news2.triage_class(x[None])[0])]
    print(f"\naggregate NEWS2 score {agg}; highest single parameter {int(pts.max())}  ->  the rule says: {rule.upper()}")
    for m in models:
        sc = predict(f"models/triage-{m}.onnx", x)
        top2 = np.sort(sc)[-2:]
        tie = "  <- TIE: the two highest scores are equal to within rounding; the decision boundary passes through this patient" if top2[1] - top2[0] < 1e-3 else ""
        print(f"triage-{m} advises: {news2.CLASSES[int(np.argmax(sc))].upper():7s}  (scores low/medium/high = {np.round(sc, 2).tolist()}){tie}")
print("helpers loaded")
'''

nb1 = nbf.v4.new_notebook()
nb1.cells = [
md(r'''
# Verifying a neural network: a triage model

**Neural network verification, hands-on.** ACM Europe Seasonal School on Responsible AI · Aston University · September 2026 · Luca Arnaboldi (University of Birmingham). *Notebook 1 of 2 · about 35 minutes · nothing to install.*

Testing a neural network tells you about the inputs you tried. **Verification** tells you about the inputs you did not: a verifier either *proves* that a property holds for **every** input in a region, or it hands you a concrete **counterexample**. In this notebook you will:

1. meet a small neural network that turns a patient's vital signs into an urgency class (low / medium / high);
2. state clinical hard limits as **properties** and ask a verifier whether they hold for *every* plausible patient;
3. watch a 96 %-accurate model fail, read the failing patient in clinical units, see a better model still fail at a corner, and see a third model prove every property;
4. probe robustness to measurement noise, edit a specification yourself, and export the property to the tool-neutral VNN-LIB format;
5. take the results to the debate: what should "robust enough" mean, and who decides?

> **Everything here is synthetic and educational.** The labelling rule is borrowed from the NHS National Early Warning Score 2 (Royal College of Physicians, 2017; Scale 1 only, no age or pregnancy adjustments) to make the properties feel real. The data are generated, the model is a toy, and nothing in this notebook is a clinical tool.

**Tools.** The specification language is [Vehicle](https://vehicle-lang.github.io/tutorial/) (you write properties in the problem's own units; it compiles them for the solver), the solver is [Marabou](https://github.com/NeuralNetworkVerification/Marabou), and the network format is ONNX. The same property compiles to [VNN-LIB](https://www.vnnlib.org/), the format every tool in the annual verification competition reads, so nothing here is tied to one tool.
'''),
md(r'''
## 0 · Setup (2–3 minutes)

Run the cell below once. It fetches the materials, creates a Python 3.11 environment with the verifier (Colab's own Python is too new for the solver's wheels), and prints `OK`. If Colab restarts the runtime, just run it again.

**Expected output:** the working directory, a few bootstrap lines, a `kernel packages: numpy OK, pandas OK, …` line, then `vehicle 0.27.1 | Marabou at /content/venv/bin/Marabou` and `OK - ready.`
'''),
code(SETUP),
md(r'''
## 1 · Two helpers

`verify(...)` runs the verifier from the command line and reads its answer; `show_patient(...)` prints a counterexample as a small observation chart, with the NEWS2 points the rule would give and what each model advises. Nothing to change here.
'''),
code(HELPERS),
md(r'''
## 2 · Meet the system

Seven inputs, in clinical units: respiration rate, SpO₂, systolic blood pressure, pulse, temperature, and two flags (on supplemental oxygen? not alert?). Three outputs: a score for *low*, *medium* and *high* urgency; the network "advises" the class with the highest score.

Three networks were trained on the same synthetic patients (`src/train.py`):

| model | training labels | what to expect |
|---|---|---|
| `triage-v1` | a rule that **forgot** one clause: "3 points in any single parameter is, on its own, a trigger for urgent review" | accurate on its own labels, wrong rule |
| `triage-v2` | the correct NEWS2 bands, ordinary training | good accuracy; is that enough? |
| `triage-v3` | the correct bands **plus the properties below as extra training losses** and a one-unit safety margin | we will see |

**Expected output:** class balance of the data, a table of accuracies (v1 ≈ 95 % on its own labels but ≈ 87 % on the correct ones; v2 ≈ 96 %; v3 ≈ 92 % — it deliberately over-triages at the thresholds), and the five ONNX nodes.
'''),
code(r'''
df = pd.read_csv("data/triage-synthetic.csv")
print(f"{len(df)} synthetic patients; class balance (correct labels):",
      dict(zip(news2.CLASSES, np.bincount(df["label"]).tolist())))
print(df[news2.FEATURES].describe().loc[["min", "50%", "max"]].round(1).to_string())

sample = df.sample(4000, random_state=0)
X = sample[news2.FEATURES].to_numpy(np.float32)
rows = []
for m in ("v1", "v2", "v3"):
    pred = np.array([int(np.argmax(predict(f"models/triage-{m}.onnx", x))) for x in X])
    rows.append((f"triage-{m}", f"{(pred == sample['label_v1'].to_numpy()).mean():.1%}", f"{(pred == sample['label'].to_numpy()).mean():.1%}"))
print("\n" + pd.DataFrame(rows, columns=["model", "accuracy vs the flawed labels", "accuracy vs the correct NEWS2 bands"]).to_string(index=False))

import onnx
print("\nONNX graph of triage-v1:", [n.op_type for n in onnx.load("models/triage-v1.onnx").graph.node],
      "- input shape", [d.dim_value for d in onnx.load("models/triage-v1.onnx").graph.input[0].type.tensor_type.shape.dim])
'''),
md(r'''
## 3 · The specification

Open `specs/triage.vcl` (printed below). Read it top to bottom once; it is written to be read.

- `validPatient` is the **region**: every physiologically plausible combination of the seven measurements. A verifier reasons about *all* of them, not a sample.
- `advises x c` says "class `c` has the strictly highest score".
- Each `@property` is a clinical hard limit: **for every valid patient with SpO₂ ≤ 91 %, the network must not advise low urgency** — because a single parameter scoring 3 points is a trigger for urgent review.
- The two flags are pinned with `==`: verifiers only know real numbers, and a "0.37 on oxygen" patient is not a thing (exercise 4 shows what happens otherwise).
- `noiseRobust` is different: it quantifies over a *perturbation* `d` around each of 12 real (synthetic) patients — two observation charts that differ by less than measurement error must not get different triage categories.
'''),
code(r'''
print(open("specs/triage.vcl").read())
'''),
md(r'''
## 4 · Ask the verifier: model v1

The property `hypoxiaNeverLow`, on the first model. The verifier negates the property and searches the whole region for a patient that violates it.

**Expected output:** `FALSIFIED hypoxiaNeverLow` with a counterexample vector in well under a second, followed by a patient chart: SpO₂ somewhere at or below 91 %, everything else unremarkable, the rule saying MEDIUM, and `triage-v1` advising LOW. That patient is the hole. Your test set would not have contained it.
'''),
code(r'''
r = verify("specs/triage.vcl", "models/triage-v1.onnx", ["hypoxiaNeverLow"])
ce = r["counterexamples"][0]
if ce:
    print("\nThe failing patient, as a chart:\n")
    show_patient(ce)
'''),
md(r'''
**Why does v1 fail?** Not because it is a bad network — it is 96 % accurate on the labels it was given. It fails because *the labels encoded the wrong rule*: they ignored the single-parameter trigger, so the data taught the model that "SpO₂ 88 %, everything else normal" is low urgency. Verification did not find a bug in the code. It found a wrong sentence in the requirement, and it found it without needing anyone to think of that patient.

Try the other two hard limits on v1 as well:
'''),
code(r'''
for prop in ("shockNeverLow", "notAlertNeverLow", "normalVitalsAlwaysLow"):
    r = verify("specs/triage.vcl", "models/triage-v1.onnx", [prop]); print()
'''),
md(r'''
## 5 · Fix the labels: model v2

`triage-v2` was trained on the correct NEWS2 bands with ordinary training (cross-entropy, 100 epochs). Accuracy against the strict rule is ≈ 96 %. Let us ask the same questions.

**Expected output:** `hypoxiaNeverLow` VERIFIED (this proof takes ~10 s: the solver has to exhaust the whole region). Then `shockNeverLow` FALSIFIED in about a second, with a patient whose systolic blood pressure is **90.0 mmHg exactly** — and, when you print the chart, the *low* and *medium* scores are equal to within rounding. The learned decision boundary passes right through the edge of the region: the rule puts the boundary at 90/91, the network put it at 90.0. "Advises medium" requires a *strict* winner, so a tie is a failure — a triage system that cannot decide is not safe, and with a whisper of noise this patient tips to *low*. No amount of testing on charted integer readings would have found a failure *at* 90.0; the verifier probes real numbers.
'''),
code(r'''
for prop in ("hypoxiaNeverLow", "shockNeverLow", "notAlertNeverLow"):
    r = verify("specs/triage.vcl", "models/triage-v2.onnx", [prop])
    if r["counterexamples"][0]:
        print(); show_patient(r["counterexamples"][0], models=("v2",))
    print()
'''),
md(r'''
## 6 · Put the property in the training loop: model v3

`triage-v3` was trained on the correct labels **and** on the properties themselves, as extra hinge losses that push the network to satisfy each property with a margin (the same idea as Vehicle's property-driven training, written out by hand in `src/train.py`). It also uses a deliberately *conservative* labelling: the worst NEWS2 score within ±1 unit of measurement uncertainty. That moves its own decision boundary two units away from every specification threshold.

**Expected output:** all five hard-safety properties VERIFIED, each in a few seconds. Then the price: v3 agrees with the strict rule on ≈ 92 % of patients, because it over-triages exactly at the thresholds (SpO₂ 92 %, SBP 91 mmHg, …).
'''),
code(r'''
HARD = ["hypoxiaNeverLow", "hypoxiaOnOxygenNeverLow", "shockNeverLow", "notAlertNeverLow", "normalVitalsAlwaysLow"]
results = {}
for m in ("v1", "v2", "v3"):
    for prop in HARD:
        r = verify("specs/triage.vcl", f"models/triage-{m}.onnx", [prop], show=False)
        results[(m, prop)] = (r["statuses"][0], r["seconds"])
table = pd.DataFrame({m: [f"{results[(m, p)][0]} ({results[(m, p)][1]:.1f} s)" for p in HARD] for m in ("v1", "v2", "v3")}, index=HARD)
print(table.to_string())
'''),
md(r'''
**Would you deploy v3?** It is the only one of the three with a *guarantee*. It is also the least accurate against the rule as written. That trade is not a technicality; it is the responsible-AI decision, and it belongs to people, not to the solver. Hold that thought for the debate.

**Where did the guarantee come from?** Look at `src/train.py`: the properties are sampled as regions during training and violated regions are penalised with a margin; the conservative labels widen the gap between the rule's boundary and the property's threshold. The verifier is what tells us the training *worked* — for every patient, not for a test set.
'''),
md(r'''
## 7 · Robustness to measurement noise

`noiseRobust` asks, for each of 12 evaluation patients: if every measurement is perturbed by less than ε (±1 breath/min, ±1 % SpO₂, ±3 mmHg, ±2 beats/min, ±0.1 °C), does the category stay the same? We scale ε by 1, 2 and 4 and count how many patients are provably stable.

Two of the twelve patients are **boundary** cases (flagged in `data/triage-eval-patients.csv`): SpO₂ 92 % is one point from a 3-point score. For them a falsification is the *rule* changing within ε, not a network bug. Read the table with that in mind.

**Expected output:** a per-patient table and a chart. Roughly: 9/12 stable at ε for both v2 and v3; at 4ε, v2 keeps ~7 and v3 only ~3 — v3's conservative boundaries sit closer to ordinary patients. A safety margin on one side is a robustness cost on the other.
'''),
code(r'''
import matplotlib.pyplot as plt
patients = pd.read_csv("data/triage-eval-patients.csv")
EPS1 = {"epsRR": 1, "epsSpo2": 1, "epsSbp": 3, "epsPulse": 2, "epsTemp": 0.1}
DATA = {"patients": "data/triage-eval-patients.idx", "labels": "data/triage-eval-labels.idx"}
per_patient, counts = {}, {}
for m in ("v2", "v3"):
    for scale in (1, 2, 4):
        eps = {k: v * scale for k, v in EPS1.items()}
        r = verify("specs/triage.vcl", f"models/triage-{m}.onnx", ["noiseRobust"], datasets=DATA, parameters=eps, show=False)
        per_patient[(m, scale)] = r["statuses"]; counts[(m, scale)] = r["summary"].get("verified", (0, 12))[0]
        print(f"{m} at {scale}x eps: {counts[(m, scale)]}/12 provably stable  ({r['seconds']:.1f} s)")

tab = patients[["spo2", "sbp", "pulse", "boundary", "note"]].copy()
for k, v in per_patient.items(): tab[f"{k[0]} {k[1]}x"] = ["ok" if s == "verified" else "CE" for s in v]
print("\n" + tab.to_string())

fig, ax = plt.subplots(figsize=(7, 3.2))
xs = np.arange(3); w = 0.36
for i, (m, hatch) in enumerate((("v2", ""), ("v3", "///"))):
    vals = [counts[(m, s)] for s in (1, 2, 4)]
    bars = ax.bar(xs + (i - 0.5) * w, vals, w, label=f"triage-{m}", color=("#14213D" if m == "v2" else "#F4EFE6"), edgecolor="#14213D", hatch=hatch)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}/12", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(["1x eps", "2x eps", "4x eps"]); ax.set_ylim(0, 13); ax.set_ylabel("patients provably stable")
ax.set_title("Robustness to measurement noise (12 evaluation patients; 2 are boundary cases)"); ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
'''),
md(r'''
## 8 · Edit the specification yourself

`specs/triage-exercises.vcl` is a copy of the specification with four exercises appended (search for `TODO`):

1. **Temperature extremes** — ≤ 35.0 °C scores 3 points: write the region so the property says something.
2. **Hypertensive crisis** — SBP ≥ 220 mmHg scores 3 points.
3. **Tighter (and looser) noise boxes** — complete `withinSpo2Noise`; then find the ε at which each model stops being provably stable.
4. **The relaxation trap** — run `hypoxiaNeverLowAnyOxygen` as written (`0 <= x ! o2 <= 1`) on v3 and read the counterexample.

Edit the file (Colab: double-click it in the file browser on the left, or use the cell below), typecheck, verify. `vehicle typecheck` catches most mistakes with a line number.

**Expected output for the cell as given:** the typecheck passes; `temperatureExtremesNeverLow` with the placeholder region is FALSIFIED immediately (the placeholder covers every patient, so of course some are low). After you fix the region to `x ! temp <= 35` it should be VERIFIED on v3.
'''),
code(r'''
# Edit specs/triage-exercises.vcl, then run this cell. Change PROP to the exercise you are working on.
PROP = "temperatureExtremesNeverLow"
lines, secs = run_vehicle(["typecheck", "-s", "specs/triage-exercises.vcl"])
print("typecheck:", "OK" if not lines else "\n".join(lines))
if not lines:
    r = verify("specs/triage-exercises.vcl", "models/triage-v3.onnx", [PROP])
    if r["counterexamples"][0]: print(); show_patient(r["counterexamples"][0], models=("v3",))
'''),
md(r'''
<details><summary><b>Solutions</b> (click to expand)</summary>

1. `x ! temp <= 35` (and, as a second property, `x ! rr <= 8` or `x ! pulse <= 40` — every "3-point" band is a candidate hard limit). All verified on v3; try v2 and watch the boundary ones fail.
2. `x ! sbp >= 220`. Verified on v3.
3. Replace `d ! spo2 == 0` by `-epsSpo2 <= d ! spo2 <= epsSpo2` and pass `-p epsSpo2:2` (etc.) together with the two `-d` datasets. Patient 10 (SpO₂ 92 %) is falsified at ε ≥ 1 for every model — the rule changes inside the box.
4. The verifier returns a patient with `o2 = 0.37` or similar: the relaxation allows a fractional flag, and the network extrapolates between the two flag values in ways nobody trained for. Pin the flag with `==` and split the property in two (`o2 == 0`, `o2 == 1`), which is what `triage.vcl` does.

</details>
'''),
md(r'''
## 9 · This is not tied to one tool

Vehicle compiles a specification to whatever the solver wants. The command below writes the queries for `hypoxiaNeverLow` in **VNN-LIB**, the property format used by every tool in the [International Verification of Neural Networks Competition](https://arxiv.org/abs/2512.19007) (VNN-COMP 2025: eight tools, including Marabou, α,β-CROWN, PyRAT, NeuralSAT, nnenum and NNV). Networks travel as ONNX. Any of those tools could take these two files.

**Expected output:** a folder with one query file per property (`hypoxiaNeverLow-query1.txt`, VNN-LIB 2.0 inside) and its first lines: the network declaration with 7 inputs and 3 outputs, the negated conclusion as `assert`s on the outputs, and the input bounds of the region in clinical units (normalisation is inside the network, so the query is readable).
'''),
code(r'''
import glob, shutil
shutil.rmtree("/tmp/vnnlib", ignore_errors=True)
lines, secs = run_vehicle(["compile", "queries", "--format", "VNNLibQueries", "-o", "/tmp/vnnlib",
                           "-s", "specs/triage.vcl", "-n", "triage:models/triage-v3.onnx", "-e", "hypoxiaNeverLow"])
files = sorted(glob.glob("/tmp/vnnlib/*query*.txt"))
print(f"{len(files)} query file(s):", [f.split('/')[-1] for f in files])
if files:
    print("\n" + "\n".join(open(files[0]).read().splitlines()[:32]))
else:
    print("\n".join(lines))
'''),
md(r'''
## 10 · Take it to the debate

You have seen three answers a verifier can give — a proof, a counterexample in your own units, a timeout — and one thing it can never give: the decision of what to demand. Pick a scenario card from `docs/debate-role-cards.pdf` (medical triage, authentication, autonomous control), follow the 20-minute protocol in `docs/debate-protocol.pdf`, and fill the decision sheet: *which region, which ε or margin, what happens on a counterexample, who signs.*

Then, for **your own project**, use `docs/spec-your-system-worksheet.pdf`: one requirement as a sentence about *every* input in a region, the same sentence in pseudo-specification, what a counterexample would look like, and what cannot be verified this way (and how you will address it instead).

### How these materials were built (for the curious)
`src/news2.py` re-states the NEWS2 thresholds and defines the class of a real-valued input as the class of its charted (truncated) reading, so every specification threshold sits a full unit away from where the labels change. `src/generate_data.py` samples 40 000 patients (a "typical ward" mixture plus uniform coverage of the whole box, because a verifier explores the whole box). `src/train.py` trains the three models with normalisation folded into the first layer so the ONNX takes clinical units and the solver sees only `Gemm` and `Relu`; v3 adds the property losses and can run a counterexample-guided repair loop against the real verifier. `src/smoke_test.sh` re-verifies everything. Two things that did *not* work are recorded in the field guide: rounding instead of truncating (an infinitely sharp step at 90.5), and repairing counterexamples one at a time (whack-a-mole).

### Further reading
Vehicle tutorial · Katz et al., *Reluplex* (CAV 2017) · VNN-COMP 2025 report (arXiv:2512.19007) · Cordeiro et al., *Neural Network Verification is a Programming Language Challenge* (ESOP 2025) · Sirman et al., *Vancomycert* (SAIV 2026) — full list in `docs/reading-list.pdf`.
'''),
]
nb1.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                "colab": {"name": "01-triage-verification.ipynb", "provenance": []}}
nbf.write(nb1, OUT / "01-triage-verification.ipynb")

# ---------------------------------------------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
md(r'''
# Fairness as a verifiable property (stretch)

*Notebook 2 of 2 · about 20 minutes · run notebook 1 first, or at least its setup cell.*

A loan-approval network takes six numbers about an applicant — income, debt-to-income ratio, years of credit history, past defaults, age, and a **protected attribute** (0/1) — and returns approve/deny scores. We would like to *prove*:

> for every pair of applicants that differ **only** in the protected attribute, the decision is the same.

That is a statement about two evaluations of the network at once. Solvers such as Marabou evaluate a network once per query, so the property cannot be written directly. The trick: build a **twin network** — two weight-shared copies side by side as *one* ONNX graph, taking a pair (A ++ B) as a single 12-number input and returning (scores A ++ scores B). Now the fairness property is an ordinary property of one network. (Idea: Athavale et al., *Verifying Global Two-Safety Properties in Neural Networks with Confidence*, CAV 2024.)

Everything is synthetic. The historical approval labels penalise the protected group on purpose — that is the point.
'''),
md("## 0 · Setup (skip if you ran notebook 1 in this session)\n\n**Expected output:** as in notebook 1, ending with `OK - ready.`"),
code(SETUP),
code(HELPERS.replace('print("helpers loaded")', 'print("helpers loaded")')),
md(r'''
## 1 · The data and the two models

`src/loan.py` generates 30 000 applicants. Creditworthiness is a simple function of the numbers; the **historical** approval label subtracts a penalty for the protected group. Two networks:

- `loan-v1` — trained on the historical labels with the protected attribute as an input;
- `loan-v2` — same, but the input weights for the attribute are zeroed after training: it *cannot* read the flag.

**Expected output:** approval rates by group in the labels (≈ 49 % vs ≈ 18 %) and for each model. Note that v2 still shows a gap (≈ 39 % vs ≈ 28 %): the protected group has lower incomes in this synthetic society, and income is a **proxy**.
'''),
code(r'''
loans = pd.read_csv("data/loan-synthetic.csv")
FEATS = ["income", "debt_ratio", "history_years", "defaults", "age", "protected"]
print("historical labels — approval rate by protected group:", loans.groupby("protected")["approved_historical"].mean().round(3).to_dict())
print("merit-only labels  — approval rate by protected group:", loans.groupby("protected")["approved_merit_only"].mean().round(3).to_dict())
sample = loans.sample(5000, random_state=1)
for m in ("v1", "v2"):
    pred = np.array([int(np.argmax(predict(f"models/loan-{m}.onnx", x))) for x in sample[FEATS].to_numpy(np.float32)])  # 1 = approve
    rates = pd.Series(pred).groupby(sample["protected"].to_numpy()).mean().round(3).to_dict()
    acc = (pred == sample["approved_historical"].to_numpy()).mean()
    print(f"loan-{m}: approval rate by group {rates}; accuracy vs historical labels {acc:.1%}")
'''),
md(r'''
## 2 · The twin specification

`specs/loan-twin.vcl` (printed below). Positions 0–5 are applicant A, 6–11 applicant B. `sameExceptProtected` ties every field of A to B and sets A's flag to 0, B's to 1. Two one-directional properties: `fairAtoB` (if A is *clearly* approved, B is approved) and `fairBtoA`.

Why "clearly", with a `margin`? Verifiers work with closed sets and relax `>` to `>=`, so a pair sitting exactly on the decision boundary would be reported as a spurious counterexample. Asking for a small margin on one side removes that artefact — and is itself a small specification decision.
'''),
code(r'''
print(open("specs/loan-twin.vcl").read())
'''),
md(r'''
## 3 · Ask the verifier (the biased model)

**Expected output.** For `loan-twin-v1`: `fairAtoB` **FALSIFIED** within a couple of seconds, with a pair — identical numbers, protected 0 approved, protected 1 denied — and `fairBtoA` VERIFIED (the model never favours the protected group).
'''),
code(r'''
LOAN_FEATS = ["income (k£/yr)", "debt ratio", "history (years)", "defaults", "age", "protected"]
r = verify("specs/loan-twin.vcl", "models/loan-twin-v1.onnx", ["fairAtoB", "fairBtoA"], parameters={"margin": 0.1},
           network_name="twin", timeout=60)
for prop, ce in zip(["fairAtoB", "fairBtoA"], r["counterexamples"]):
    if ce:
        A, B = ce[:6], ce[6:]
        tab = pd.DataFrame({"field": LOAN_FEATS, "applicant A": np.round(A, 3), "applicant B": np.round(B, 3)})
        print(f"\ncounterexample pair for {prop}:\n" + tab.to_string(index=False))
        for name, x in (("A", A), ("B", B)):
            sc = predict("models/loan-v1.onnx", x)
            print(f"  loan-v1 on {name}: {'APPROVE' if sc[1] > sc[0] else 'DENY'}  (deny/approve scores {np.round(sc, 2).tolist()})")
'''),
md(r'''
## 3b · The attribute-blind model: a proof without a solver

`loan-v2` had its first-layer weights on the protected column set to exactly zero after training. That is a proof of counterfactual fairness in one line: **the attribute has no path to the output**, so flipping it cannot change anything. The cell below checks the weights, and checks the claim numerically on 2,000 random pairs.

We deliberately do **not** run Marabou on `loan-twin-v2`. Two identical sub-networks side by side make the query degenerate: in our tests Marabou looped without terminating (not even honouring its own `--timeout`), and a variant with near-zero instead of zero weights returned a "counterexample" that, re-evaluated, did not violate the property. Complete solvers have numerical corners; knowing when *not* to reach for one is part of the craft. (If you want to see it, the `verify` helper has a wall-clock cap, so a call on `loan-twin-v2` ends with `timeout` after a few minutes rather than hanging.)

**Expected output:** `max |weight| on the protected column: 0.0` for v2 (and a clearly non-zero value for v1), then `0 of 2000 random pairs change decision` for v2.
'''),
code(r'''
import onnx
from onnx import numpy_helper
for m in ("v1", "v2"):
    g = onnx.load(f"models/loan-{m}.onnx").graph
    W1 = [numpy_helper.to_array(t) for t in g.initializer if t.dims and len(t.dims) == 2][0]   # first Gemm weight (out, in)
    print(f"loan-{m}: first-layer weight matrix {W1.shape}; max |weight| on the protected column: {np.abs(W1[:, 5]).max():.6f}")

rng = np.random.default_rng(0)
LO = np.array([10, 0.0, 0, 0, 18, 0]); HI = np.array([200, 1.0, 40, 5, 90, 1])
A = rng.uniform(LO, HI, size=(2000, 6)); A[:, 3] = np.round(A[:, 3]); A[:, 5] = 0; B = A.copy(); B[:, 5] = 1
for m in ("v1", "v2"):
    flips = sum(int(np.argmax(predict(f"models/loan-{m}.onnx", a)) != np.argmax(predict(f"models/loan-{m}.onnx", b))) for a, b in zip(A, B))
    print(f"loan-{m}: {flips} of 2000 random pairs change decision when only the protected attribute is flipped")
'''),
md(r'''
## 4 · So is v2 fair?

It is *individually* counterfactually fair, with a proof (a structural one). Its approval rates by group are still ≈ 39 % vs ≈ 28 %. Both statements are true. Verification answered exactly the question we asked — and no other. Whether the right question is "same decision for the same numbers" or "similar outcomes for the two groups" is a policy choice; the second is not a property of the network alone, it is a property of the network *and* the society that produced the data.

Legal anchors for the debate: the EU AI Act lists credit scoring of natural persons as **high-risk** (Annex III), so Article 15's "appropriate robustness" applies; in the UK, the Equality Act 2010 covers *indirect* discrimination, which is exactly what a proxy produces.
'''),
md(r'''
## 5 · Ordinary properties of the single network

`specs/loan.vcl` states two other kinds of property on `loan-v1/v2` directly: a **region** property ("a plainly strong applicant is approved, whatever the flag") and **local robustness** ("a small change in reported income never flips the decision") around ten evaluation applicants (eight confident, two borderline).

**Expected output:** in our runs both region properties are VERIFIED for v1 and for v2 (about a second each). Read what *your* run says: if a verifier ever returns a counterexample at a corner of the box (income 200 k£, age 90, …), that is a finding about the model's extrapolation, not a bug in the verifier — report it honestly. Income robustness: the eight confident applicants stay stable even at ±10 k£; the two borderline ones flip at the smallest ε (8/10 throughout).
'''),
code(r'''
for m in ("v1", "v2"):
    print(f"=== loan-{m}: strong applicants approved (protected = 0 / = 1) ===")
    r = verify("specs/loan.vcl", f"models/loan-{m}.onnx", ["strongApplicantsApproved0", "strongApplicantsApproved1"], network_name="loan", timeout=60)
    for prop, ce in zip(["strongApplicantsApproved0", "strongApplicantsApproved1"], r["counterexamples"]):
        if ce: print(f"  {prop} counterexample:", dict(zip(LOAN_FEATS, np.round(ce, 2).tolist())))
    print()
applicants = pd.read_csv("data/loan-eval-applicants.csv")
print(applicants.to_string()); print()
for eps in (0.5, 2, 10):
    r = verify("specs/loan.vcl", "models/loan-v2.onnx", ["incomeRobust"], network_name="loan",
               datasets={"applicants": "data/loan-eval-applicants.idx", "decisions": "data/loan-eval-decisions.idx"},
               parameters={"epsIncome": eps}, show=False)
    print(f"income +/- {eps} k£: {r['summary'].get('verified', (0, 10))[0]}/10 applicants provably stable   ({r['seconds']:.1f} s)")
'''),
md(r'''
## 6 · What you can now say in a design review

- "Counterfactual fairness with respect to attribute *p* is a *property*; here is the proof / here is the pair that violates it."
- "Removing the attribute made the property trivially true and changed group outcomes by this much; the rest is proxies."
- "The verifier reasons about one network application; hyperproperties need an encoding (self-composition), and that encoding is part of the specification you should review."

Back to the worksheet: which of your project's fairness or safety requirements is a *property*, and which is a *policy*?
'''),
]
nb2.metadata = nb1.metadata | {"colab": {"name": "02-fairness-stretch.ipynb", "provenance": []}}
nbf.write(nb2, OUT / "02-fairness-stretch.ipynb")
print("wrote", OUT / "01-triage-verification.ipynb", "and", OUT / "02-fairness-stretch.ipynb")
