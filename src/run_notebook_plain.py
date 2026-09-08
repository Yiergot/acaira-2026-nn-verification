"""Execute a notebook WITHOUT a Jupyter kernel and write an executed copy.

Why: the Jupyter kernel's ZeroMQ handshake hangs under CPU emulation (amd64 image on an Apple Silicon host), so
`nbconvert --execute` cannot be used to produce the executed copies there. This runner exec()s each code cell in one
shared namespace, captures stdout/stderr and matplotlib figures, and stores them as notebook outputs. The notebooks
use plain Python only (no IPython magics), so the semantics are identical.

    python src/run_notebook_plain.py notebooks/01-triage-verification.ipynb notebooks/executed/01-triage-verification.ipynb
"""
import base64, contextlib, io, sys, time, traceback
from pathlib import Path

import nbformat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
nb = nbformat.read(src_path, as_version=4)
ns = {"__name__": "__main__"}
figures = []

def _show(*_a, **_k):
    for num in plt.get_fignums():
        buf = io.BytesIO(); plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        figures.append(base64.b64encode(buf.getvalue()).decode()); plt.close(num)
plt.show = _show

failed = False; n = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    n += 1; buf = io.StringIO(); t0 = time.time(); figures.clear()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(cell.source, f"<cell {n}>", "exec"), ns)
        except Exception:
            traceback.print_exc(file=buf); failed = True
    outputs = []
    if buf.getvalue():
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
    for png in figures:
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": png}))
    cell.outputs = outputs; cell.execution_count = n
    status = "ERROR" if "Traceback" in buf.getvalue() else "ok"
    print(f"cell {n:2d}: {status} ({time.time() - t0:.1f} s)", flush=True)
    if status == "ERROR":
        print(buf.getvalue()[-1500:], flush=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out_path)
print("wrote", out_path, "FAILED" if failed else "OK")
sys.exit(1 if failed else 0)
