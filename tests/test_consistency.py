"""Consistency checks between the simulator, the label channel and the paper's equations.

Run:  python -m tests.test_consistency
"""
from dataclasses import replace

import numpy as np

from aod.exp import pool_for
from aod.sim import SimConfig, BatchSim, dwell_estimates, G_true
from aod.policies import ThresholdPolicy, StaticThreshold


def check_dwell_matches_episodes(pool):
    cfg = SimConfig(T=600, rho=0.01)
    for pol in [ThresholdPolicy(np.tile([-2.0, 0.5, -0.5, 1.0, 0, 0], (8, 1)), lam=np.zeros(8)),
                StaticThreshold(2.0)]:
        res = BatchSim(cfg, pool, list(range(8))).run(pol, record_dwell=True)
        est = dwell_estimates(res, cfg, 8)
        assert np.allclose(est["oracle"], res.metrics["dwell"]), (est["oracle"], res.metrics["dwell"])
        assert np.allclose(res.dwell.sum((1, 2)), res.metrics["dwell"])
    print("ok  dwell counted by the simulator == sum of episode lengths (Eq. 1 vs Eq. 4)")


def check_all_revealed(pool):
    cfg = SimConfig(T=600, p_never=0.0, delay_median=1.0, delay_sigma=0.01, grace=10_000)
    res = BatchSim(cfg, pool, list(range(6))).run(StaticThreshold(2.0))
    est = dwell_estimates(res, cfg, 6)
    assert np.allclose(est["naive"], est["oracle"])
    assert np.allclose(est["ipcw"], est["oracle"], rtol=1e-6)
    print("ok  all labels revealed: naive == IPCW == oracle")


def check_ipcw_unbiased(pool, reps=400):
    cfg = SimConfig(T=600, delay_median=300.0, p_never=0.3, grace=300)
    res = BatchSim(cfg, pool, [11]).run(StaticThreshold(3.0))
    H = cfg.T + cfg.grace
    rng = np.random.default_rng(0)
    eps = res.episodes
    oracle = sum(e[3] - e[2] for e in eps)
    vals_ipcw, vals_naive = [], []
    for _ in range(reps):
        tot_i = tot_n = 0.0
        for (bi, di, on, end, cont, rev, d) in eps:
            never = rng.random() < cfg.p_never
            dd = np.inf if never else np.exp(np.log(cfg.delay_median) + cfg.delay_sigma * rng.standard_normal())
            if on + dd <= H:
                g = float(G_true(cfg, np.array([H - on]))[0])
                tot_i += (end - on) / max(g, 0.05)
                tot_n += (end - on)
        vals_ipcw.append(tot_i); vals_naive.append(tot_n)
    mi, se = np.mean(vals_ipcw), np.std(vals_ipcw) / np.sqrt(reps)
    assert abs(mi - oracle) < 4 * se + 1e-9, (mi, oracle, se)
    print(f"ok  IPCW unbiased over resampled delays: mean {mi:.1f} vs oracle {oracle} (se {se:.1f}); "
          f"naive mean {np.mean(vals_naive):.1f}")


def check_bucket(pool):
    cfg = SimConfig(T=500, rho=0.004, bucket=3.0)
    always_E = ThresholdPolicy(np.tile([-30.0, 0, 0, 30.0, 0, 0], (4, 1)), lam=np.zeros(4), congestion=False)
    res = BatchSim(cfg, pool, list(range(4))).run(always_E)
    per_dev_max = res.metrics["esc_req"].max() / cfg.N
    bound = np.floor(cfg.bucket + cfg.rho * (cfg.T - 1))
    # every online device requests every tick, so each device saturates its bucket
    assert res.metrics["esc_req"].max() <= cfg.N * bound + 1e-9
    print(f"ok  token bucket: <= {bound:.0f} escalations per device per window (mean {per_dev_max:.1f})")


if __name__ == "__main__":
    pool = pool_for("ciciot2023")[0]
    check_dwell_matches_episodes(pool)
    check_all_revealed(pool)
    check_ipcw_unbiased(pool)
    check_bucket(pool)
    print("all consistency checks passed")
