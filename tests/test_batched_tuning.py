"""Batched grid tuning must reproduce the one-point-at-a-time results exactly.

  python -m tests.test_batched_tuning
"""
import itertools
import time
from dataclasses import replace

import numpy as np

from aod.exp import pool_for, make_baselines, theta_batch, BATCH
from aod.policies import ThresholdPolicy
from aod.sim import SimConfig, BatchSim
from aod.train import tune_grid


def sequential(make, grid, pool, cfg, n_env, seed):
    seeds = list(range(seed * 1000, seed * 1000 + n_env))
    c2 = replace(cfg, pool_split="cal")
    rows = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        pol = make(**kw)
        if pol is None:
            continue
        m = BatchSim(c2, pool, seeds).run(pol).metrics
        rows.append((float(m["cost"].mean()), float(m["esc_rate"].mean())))
    return rows


def main():
    pool = pool_for("ciciot2023")[0]
    cfg = SimConfig(rho=0.005)
    small = {"Static threshold": {"tau": [1.0, 3.0]},
             "Window NP": {"w": [5, 20], "tau": [2, 6]},
             "Belief stopping": {"bC": [-2.0, 1.0]},
             "Three-zone deferral": {"l": [0.5, 2.5], "u": [2.0, 6.0], "v": [0.0, 2.0]}}
    b2 = lambda bE, gap: ThresholdPolicy([bE, 0, 0, np.log(np.expm1(gap)), 0, 0], congestion=False)
    cases = [(n, make_baselines(cfg)[n][0], small[n], BATCH.get(n)) for n in small]
    cases += [(n + " (stacked)", make_baselines(cfg)[n][0], small[n], None) for n in ("Static threshold", "Three-zone deferral")]
    cases.append(("Two-threshold", b2, {"bE": [-3.0, -1.0], "gap": [0.5, 3.0]},
                  theta_batch(lambda bE, gap: [bE, 0, 0, np.log(np.expm1(gap)), 0, 0])))
    for name, make, grid, mb in cases:
        t = time.time(); ref = sequential(make, grid, pool, cfg, 4, 1001); t1 = time.time() - t
        t = time.time(); _, rows = tune_grid(make, grid, pool, cfg, n_env=4, seed=1001, make_batch=mb, chunk=3)
        t2 = time.time() - t
        got = [(r["cost"], r["rate"]) for r in rows]
        assert len(got) == len(ref), (name, len(got), len(ref))
        err = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(got, ref))
        print(f"{name:20s} points={len(ref):2d} max|diff|={err:.2e}  sequential {t1:.1f}s  batched {t2:.1f}s")
        assert err < 1e-9, name
    print("batched tuning: OK")


if __name__ == "__main__":
    main()
