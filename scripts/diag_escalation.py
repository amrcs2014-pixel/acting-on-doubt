"""Improvement-loop diagnostic: why does AoD-T escalate less than the oracle-tuned two-threshold rule
at generous budgets (rho >= 0.005)?  Label noise (IPCW) or the optimiser (init / iterations)?

Arms on CICIoT2023, rho = 0.005, seeds 0-1, evaluated on the usual 32 test networks:
  oracle   : ES on oracle labels (removes label noise)
  warm     : ES on IPCW labels, started from the oracle-tuned two-threshold solution
  long     : ES on IPCW labels, 240 iterations instead of 120
Writes results/raw/_validation/diag_escalation.jsonl
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aod.exp import pool_for, evaluate, ROOT          # noqa: E402
from aod.policies import ThresholdPolicy              # noqa: E402
from aod.sim import SimConfig                         # noqa: E402
from aod.train import train_es, train_aodt_v3        # noqa: E402

OUT = os.path.join(ROOT, "results", "raw", "_validation", "diag_escalation.jsonl")


def warm_theta(seed):
    for l in open(os.path.join(ROOT, "results", "raw", "E1b.jsonl"), encoding="utf8"):
        r = json.loads(l)
        if r["ds"] == "ciciot2023" and abs(r["cfg_rho"] - 0.005) < 1e-9 and r["seed"] == seed:
            p = r["params"]
            return [p["bE"], 0, 0, float(np.log(np.expm1(p["gap"]))), 0, 0]
    raise KeyError(seed)


def arm(name, seed):
    cfg = SimConfig(rho=0.005)
    pool = pool_for("ciciot2023")[0]
    kw = dict(label_mode="ipcw", iters=120, seed=seed)
    if name == "oracle":
        kw["label_mode"] = "oracle"
    elif name == "warm":
        kw["theta0"] = warm_theta(seed)
    elif name == "long":
        kw["iters"] = 240
    elif name == "crn":                      # IPCW labels with common random numbers for the label channel
        cfg = SimConfig(rho=0.005, label_crn=True)
    elif name == "decay":                    # decaying step size, longer Polyak average
        kw.update(lr_end=0.006, avg_last=40)
    elif name == "decay8":                   # ... and twice the windows per ES evaluation
        kw.update(lr_end=0.006, avg_last=40, n_env=8)
    elif name == "decay_oracle":
        kw.update(lr_end=0.006, avg_last=40, label_mode="oracle")
    t0 = time.time()
    info = {}
    if name == "v3":
        th, lam, hist, info = train_aodt_v3(pool, cfg, seed=seed)
        # the warm-start (grid) policy on its own, evaluated on the same test networks
        info["test_warm"] = evaluate(ThresholdPolicy(info["theta_warm"], lam=0.0), pool, cfg)["cost"]
    else:
        th, lam, hist = train_es(pool, cfg, **kw)
    rec = dict(arm=name, seed=seed, theta=list(map(float, th)), lam=lam, train_s=time.time() - t0, info=info,
               **evaluate(ThresholdPolicy(th, lam=lam), pool, cfg))
    with open(OUT, "a", encoding="utf8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = {(json.loads(l)["arm"], json.loads(l)["seed"]) for l in open(OUT, encoding="utf8")} if os.path.exists(OUT) else set()
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["oracle", "warm", "long"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]
    jobs = [(a, s) for s in seeds for a in arms if (a, s) not in done]
    with ProcessPoolExecutor(min(3, len(jobs))) as ex:
        for r in ex.map(arm, *zip(*jobs)):
            print(r["arm"], r["seed"], round(r["cost"], 3), round(r["esc_rate"], 4), flush=True)
