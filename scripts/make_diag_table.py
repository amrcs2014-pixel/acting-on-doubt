"""Write paper/tables/diag.tex: AoD-T trainer diagnostics (improvement loop), from results/raw only."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aod.exp import load  # noqa: E402

rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "raw", "_validation", "diag_escalation.jsonl"))]
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["seed"]] = r["cost"]
by["v1"] = {r["seed"]: r["cost"] for r in load("E1", current=False)
            if r["method"] == "AoD-T (ipcw)" and r.get("trainer", "v1") == "v1" and r["ds"] == "ciciot2023"
            and abs(r["cfg_rho"] - 0.005) < 1e-9}
names = [("v1", r"ES, $\omega=0.3$, 120 it., constant step (first trainer)"),
         ("oracle", "same, on oracle labels"),
         ("crn", "same, shared label draws"),
         ("long", "same, 240 iterations"),
         ("warm", "same, started at the oracle-tuned rule"),
         ("decay", "decaying step, average of last 40"),
         ("decay8", "decaying step, 8 windows per evaluation"),
         ("v3", r"\textbf{v3}: warm start + refinement + selection")]


def cell(a, s):
    return f"{by[a][s]:.2f}" if s in by.get(a, {}) else "--"


L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Trainer & seed 0 & seed 1 & seed 2\\", r"\midrule"]
L += [f"{nm} & {cell(a, 0)} & {cell(a, 1)} & {cell(a, 2)}" + r"\\" for a, nm in names]
L += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
open(os.path.join(ROOT, "paper", "tables", "diag.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
