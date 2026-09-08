# Verifying neural networks — hands-on materials

*ACM Europe Seasonal School on Responsible AI: Concepts and Applications · Aston University, Birmingham · 16 September 2026*
*Plenary and drop-in session by Dr Luca Arnaboldi (University of Birmingham).*

Testing a neural network tells you about the inputs you tried. **Verification** tells you about the inputs you did not:
it either *proves* that a property holds for **every** input in a region, or hands you a concrete **counterexample**.
These materials let you do that yourself, in a browser, in about an hour, and then argue about what "robust enough"
should mean for a real deployment.

The specification language used is [Vehicle](https://vehicle-lang.github.io/tutorial/) and the solver is
[Marabou](https://github.com/NeuralNetworkVerification/Marabou); the properties compile to the standard
[VNN-LIB](https://www.vnnlib.org/) format that every tool in the annual [VNN-COMP](https://arxiv.org/abs/2512.19007)
competition reads, so nothing here is tied to one tool.

> **Everything here is synthetic and educational.** The triage example borrows its labelling rule from the NHS
> National Early Warning Score 2 (Royal College of Physicians, 2017; Scale 1 only) to make the properties feel real.
> It is not a clinical tool and must not be used as one.

## Start here (nothing to install)

| | Open in Colab | What you do | Time |
|---|---|---|---|
| **01** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucaarnaboldi/acaira-2026-nn-verification/blob/main/notebooks/01-triage-verification.ipynb) | Verify a neural triage network against clinical hard limits. Watch a 96 %-accurate model fail, read the failing patient, repair the model, prove the properties, then probe robustness to measurement noise. | ~35 min |
| **02** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucaarnaboldi/acaira-2026-nn-verification/blob/main/notebooks/02-fairness-stretch.ipynb) | Stretch: state *fairness* as a verifiable property of a loan-approval network using a twin-network trick, and see why removing a protected attribute is not enough. | ~20 min |

No laptop? The **[field guide](docs/build/field-guide.pdf)** reproduces every step with its expected output, and the
**[role cards](docs/build/debate-role-cards.pdf)** + **[protocol](docs/build/debate-protocol.pdf)** run the debate
without a computer. Executed copies of both notebooks are in [`notebooks/executed/`](notebooks/executed/).

*(The Colab badges point at the repository URL placeholder `lucaarnaboldi/acaira-2026-nn-verification`; adjust if the
repository lives elsewhere.)*

## What is in the box

| Path | Contents |
|---|---|
| `notebooks/` | The two Colab notebooks (+ `executed/` copies that render on GitHub) |
| `specs/` | Vehicle specifications: `triage.vcl`, `triage-exercises.vcl` (TODOs), `loan.vcl`, `loan-twin.vcl` |
| `models/` | ONNX networks: `triage-v1/v2/v3.onnx`, `loan-v1/v2.onnx`, `loan-twin-v1/v2.onnx` |
| `data/` | Synthetic datasets (CSV) and the small evaluation sets in idx format used by the specs |
| `src/` | How everything was built: `news2.py` (labelling rule), `generate_data.py`, `train.py`, `make_idx.py`, `loan.py`, `smoke_test.sh` |
| `environment/` | `colab_bootstrap.sh` (Python 3.11 venv with `vehicle-lang` + `maraboupy`), `Dockerfile` mirroring Colab, `local-install.md` |
| `docs/` | LaTeX sources and PDFs: field guide, cheat sheet, debate role cards, debate protocol, "specify your system" worksheet, reading list |

### The three triage models, in one table

| Model | Trained on | Accuracy vs the strict rule | Hard-safety properties | Story |
|---|---|---|---|---|
| `triage-v1` | labels that *forgot* the "3 points in any single parameter" trigger | 88 % | 3 of 5 fail (counterexamples in < 1 s) | accurate on its own data, wrong rule |
| `triage-v2` | correct labels, ordinary training | 96 % | 3 of 5 pass; `shockNeverLow` fails at 90.0 mmHg | a learned boundary is never exactly where the rule puts it |
| `triage-v3` | correct labels + the properties as extra losses + a one-unit safety margin | 92 % | **5 of 5 proved** (each in seconds) | proofs bought with a margin; over-triage at the thresholds is the visible price |

## Run it locally

```bash
bash environment/colab_bootstrap.sh          # x86_64 Linux/macOS; creates ./venv unless VENV is set
export PATH="$PWD/venv/bin:$PATH"            # or the VENV you chose
bash src/smoke_test.sh                       # every property on every model, with timings
```
Marabou's wheels exist for Python 3.8–3.11 on x86_64 only. Apple Silicon: use Colab, or an x86_64 Python under
Rosetta, or the Docker image (`docker build --platform linux/amd64 -t nnv-colab environment/`). Details in
`environment/local-install.md` and the field guide's appendix.

## Rebuilding the models
```bash
python src/generate_data.py && python src/train.py --variant v1 && python src/train.py --variant v2
VEHICLE_BIN=vehicle MARABOU_BIN=Marabou python src/train.py --variant v3   # uses the verifier in the loop
python src/make_idx.py && python src/loan.py
```
(Needs PyTorch, ONNX, onnxruntime, pandas, idx2numpy; any Python ≥ 3.10.)

## Credits and licence
Code (`src/`, `environment/`, notebooks) — MIT. Documents, specifications, data and models — CC BY 4.0.
Built on the [Vehicle tutorial](https://vehicle-lang.github.io/tutorial/) by Daggitt, Kokke, Atkey, Arnaboldi,
Komendantskaya and colleagues (BSD-3-Clause); the ACAS Xu, MNIST and wind-controller examples referenced in the talk
are theirs. The twin-network encoding of fairness follows Athavale et al., "Verifying Global Two-Safety Properties
in Neural Networks with Confidence", CAV 2024. NEWS2 is © Royal College of Physicians 2017; only the published
thresholds are re-stated here.

Contact: see the speaker's institutional page at the University of Birmingham.
