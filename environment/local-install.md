# Installing the toolchain locally

The notebooks run in Google Colab with nothing to install. If you want the tools on your own machine:

| Platform | Works? | How |
|---|---|---|
| Linux x86_64, Python 3.10–3.11 | yes | `pip install vehicle-lang==0.27.1 maraboupy==2.0.0 "numpy<2"` |
| Linux x86_64, any Python | yes | `bash environment/colab_bootstrap.sh` (creates a Python 3.11 venv with `uv`) |
| macOS Intel | yes | as Linux |
| **macOS Apple Silicon** | not natively (Marabou ships no arm64 wheels) | `uv python install cpython-3.11-macos-x86_64-none`, then `uv venv --python <that python> venv-x86 && uv pip install --python venv-x86/bin/python vehicle-lang==0.27.1 maraboupy==2.0.0 "numpy<2"` (runs under Rosetta; do **not** wrap `uv` itself in `arch -x86_64`) — or Docker below |
| Windows | not for Marabou wheels | WSL2 (x86_64) or Docker |
| Any, with Docker | yes | `docker build --platform linux/amd64 -t nnv-colab environment/ && docker run --rm -it -v "$PWD:/work" -w /work nnv-colab` |

Check: `vehicle --version` prints `0.27.1`; `Marabou --help` prints Marabou's usage.

Gotchas: Vehicle writes a compiled `<spec>.vclo` next to the specification, so the folder must be writable; Vehicle
has no timeout of its own — pass `--solver-args "--timeout=30"`; strict inequalities are relaxed to non-strict for
Marabou (a warning you can ignore for these examples).
