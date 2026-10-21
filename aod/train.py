"""Training and tuning.

* train_es: AoD-T. Primal-dual policy search over the 6 threshold parameters.
  The primal step is antithetic evolution strategies with common random numbers on the
  Lagrangian  J(theta) + lam * (escalation rate - rho), where J uses ONLY the training
  labels available to a deployment (naive / IPCW-corrected / oracle for the B10 arm).
  The dual step is projected ascent on lam.
* tune_grid: every baseline is tuned on the calibration split with FULL ground truth
  (an advantage the baselines get and AoD does not).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np

from .sim import SimConfig, BatchSim, dwell_estimates
from .policies import Policy, ThresholdPolicy


def lagrangian_terms(res, cfg, B, label_mode, **lab_kw):
    est = dwell_estimates(res, cfg, B, **lab_kw)
    dwell = est["oracle"] if label_mode == "oracle" else est[label_mode]
    m = res.metrics
    cost_hat = (cfg.c_fn * dwell + cfg.c_s * m["iso"]) / (cfg.N * cfg.T) * 1000
    return cost_hat, m["esc_rate"]


def train_es(pool, cfg: SimConfig, label_mode="ipcw", iters=80, pairs=8, n_env=4, sigma=0.3, lr=0.06,
             theta0=None, lam0=0.5, dual_lr=0.3, seed=0, no_escalate=False, congestion=True,
             verbose=False, lab_kw=None, avg_last=20, lr_end=None):
    """lr_end: if set, the primal step size decays linearly from lr to lr_end (stochastic-approximation
    schedule; with a constant step, Adam keeps taking full-size steps on pure noise near the optimum)."""
    lab_kw = lab_kw or {}
    rng = np.random.default_rng(seed)
    cfg = replace(cfg, pool_split="cal")
    th = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0] if theta0 is None else theta0, float)
    dim = len(th)
    lam = float(lam0)
    m_, v_ = np.zeros(dim), np.zeros(dim)
    hist = []
    for it in range(iters):
        eps = rng.standard_normal((pairs, dim))
        if no_escalate:
            eps[:, [0, 1, 2] + ([6] if dim > 6 else [])] = 0.0
        thetas = np.concatenate([th + sigma * eps, th - sigma * eps])          # (2P,6)
        base = int(rng.integers(0, 2**31 - 1 - pairs * n_env))
        seeds = []
        for k in range(pairs):
            seeds += [base + k * n_env + e for e in range(n_env)]
        seeds = seeds + seeds                                                    # CRN for +/-
        theta_env = np.repeat(thetas, n_env, axis=0)
        pol = ThresholdPolicy(theta_env, lam=np.full(len(seeds), lam), congestion=congestion,
                              no_escalate=no_escalate)
        sim = BatchSim(cfg, pool, seeds)
        res = sim.run(pol)
        cost_hat, rate = lagrangian_terms(res, cfg, len(seeds), label_mode, **lab_kw)
        L = cost_hat + lam * (rate - cfg.rho) * 1000
        Lk = L.reshape(2 * pairs, n_env).mean(1)
        Lp, Lm = Lk[:pairs], Lk[pairs:]
        g = ((Lp - Lm)[:, None] * eps).sum(0) / (2 * sigma * pairs)
        # Adam (minimise)
        m_ = 0.9 * m_ + 0.1 * g; v_ = 0.999 * v_ + 0.001 * g * g
        mh, vh = m_ / (1 - 0.9 ** (it + 1)), v_ / (1 - 0.999 ** (it + 1))
        lr_t = lr if lr_end is None else lr + (lr_end - lr) * it / max(iters - 1, 1)
        th = th - lr_t * mh / (np.sqrt(vh) + 1e-8)
        r_mean = float(rate.mean())
        if not no_escalate:
            lam = max(0.0, lam + dual_lr * (r_mean / cfg.rho - 1.0))
        hist.append(dict(it=it, L=float(L.mean()), cost_hat=float(cost_hat.mean()),
                         cost_true=float(res.metrics["cost"].mean()), rate=r_mean, lam=lam, theta=th.tolist()))
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  it {it:3d} L {L.mean():7.3f} cost_hat {cost_hat.mean():6.3f} true {res.metrics['cost'].mean():6.3f} "
                  f"rate {r_mean:.4f} lam {lam:.3f} th {np.round(th, 2)}", flush=True)
    # Polyak averaging over the last iterations: the returned policy does not hinge on one noisy step
    k = min(avg_last, len(hist))
    th_avg = np.mean([h["theta"] for h in hist[-k:]], axis=0)
    lam_avg = float(np.mean([h["lam"] for h in hist[-k:]]))
    return th_avg, lam_avg, hist


WARM_GRID = {"bE": list(np.arange(-6.0, 2.01, 0.5)), "gap": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]}


def _two_thr(bE, gap):
    return [bE, 0.0, 0.0, float(np.log(np.expm1(gap))), 0.0, 0.0]


def corrected_cost(theta, lam, pool, cfg, label_mode, seeds, lab_kw=None, congestion=True):
    """Mean training-label (e.g. IPCW-corrected) cost and escalation rate of one policy on calibration windows."""
    res = BatchSim(replace(cfg, pool_split="cal"), pool, seeds).run(ThresholdPolicy(theta, lam=lam, congestion=congestion))
    c, rate = lagrangian_terms(res, cfg, len(seeds), label_mode, **(lab_kw or {}))
    return float(c.mean()), float(rate.mean())


def train_aodt_v3(pool, cfg: SimConfig, label_mode="ipcw", seed=0, lab_kw=None, congestion=True, iters=40,
                  n_env=4, sigma=0.15, lr=0.02, lr_end=0.002, n_sel=32, n_env_grid=8):
    """AoD-T trainer v3 (improvement loop), using only the training labels throughout:
    1. warm start: grid over constant (beta_E, beta_C - beta_E), scored with the training-label cost on common
       calibration windows (common random numbers across grid points), budget-feasible points only;
    2. refinement: primal-dual ES on all six parameters from the warm start (decaying step, 8 windows per eval);
    3. selection: keep the refinement only if its training-label cost on fresh calibration windows is lower
       and it meets the budget; otherwise keep the warm start."""
    lab_kw = lab_kw or {}
    mk = lambda kws, n: ThresholdPolicy(np.repeat(np.array([_two_thr(**kw) for kw in kws]), n, axis=0),
                                        lam=np.zeros(len(kws) * n), congestion=congestion)
    best, rows = tune_grid(lambda **kw: True, WARM_GRID, pool, cfg, n_env=n_env_grid, seed=5000 + seed, budget=True,
                           make_batch=mk, label_mode=label_mode, lab_kw=lab_kw)
    th0 = np.array(_two_thr(**best[0]) if best else [-1.0, 0, 0, 1.0, 0, 0])
    # local refinement: small exploration noise (sigma=0.3 biased ES towards the flat side of the cost surface)
    th_es, lam_es, hist = train_es(pool, cfg, label_mode=label_mode, iters=iters, seed=seed, lab_kw=lab_kw,
                                   congestion=congestion, theta0=th0, lam0=0.0, n_env=n_env, sigma=sigma, lr=lr,
                                   lr_end=lr_end)
    sel = list(range(7_000_000 + 1000 * seed, 7_000_000 + 1000 * seed + n_sel))
    c0, r0 = corrected_cost(th0, 0.0, pool, cfg, label_mode, sel, lab_kw, congestion)
    c1, r1 = corrected_cost(th_es, lam_es, pool, cfg, label_mode, sel, lab_kw, congestion)
    use_es = (c1 < c0) and (r1 <= cfg.rho * 1.05)
    info = dict(warm=best[0] if best else None, sel_cost_warm=c0, sel_rate_warm=r0, sel_cost_es=c1, sel_rate_es=r1,
                use_es=bool(use_es), theta_warm=th0.tolist(), theta_es=th_es.tolist(), lam_es=lam_es)
    return (th_es, lam_es, hist, info) if use_es else (th0, 0.0, hist, info)


def evaluate(policy, pool, cfg: SimConfig, seeds, split="test"):
    sim = BatchSim(replace(cfg, pool_split=split), pool, seeds)
    res = sim.run(policy)
    return res


class Stacked(Policy):
    """Several policies side by side, each on its own block of n_env environments.

    Environments in a BatchSim never interact and each draws from its own seeded generator, so a
    policy's results on its block are identical to running it alone on those seeds."""

    def __init__(self, pols, n_env):
        self.pols, self.n = pols, n_env
        self.uses_congestion = pols[0].uses_congestion
        assert all(p.uses_congestion == self.uses_congestion for p in pols)

    def reset(self, B, N, cfg):
        super().reset(B, N, cfg)
        self.sl = [slice(k * self.n, (k + 1) * self.n) for k in range(len(self.pols))]
        self.lam = np.zeros((B, 1))
        for p, s in zip(self.pols, self.sl):
            p.reset(self.n, N, cfg)
            self.lam[s] = np.broadcast_to(np.asarray(p.lam, float).reshape(-1, 1), (self.n, 1))

    def act(self, obs):
        out = np.empty(obs["lo"].shape, np.int64)
        for p, s in zip(self.pols, self.sl):
            out[s] = p.act({k: (v[s] if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        return out

    def observe_verdict(self, served, llr_v):
        for p, s in zip(self.pols, self.sl):
            p.observe_verdict(served[s], llr_v[s])


def tune_grid(make, grid: dict, pool, cfg: SimConfig, n_env=16, seed=123, budget=True, make_batch=None, chunk=64,
              label_mode="oracle", lab_kw=None, refine_top=0, refine_env=64):
    """Exhaustive grid on the calibration split; respects the energy budget.

    Baselines are scored with the true (oracle) cost. label_mode="ipcw"/"naive" scores with the cost a
    deployment could compute from its delayed labels instead (used for AoD-T's warm start).
    Grid points are simulated side by side in batches of `chunk` (same seeds for every point).
    make_batch(list_of_kw, n_env), if given, builds one vectorised policy for a whole batch;
    otherwise the points' policies are stacked."""
    keys = list(grid)
    best, rows = None, []
    seeds = list(range(seed * 1000, seed * 1000 + n_env))
    c2 = replace(cfg, pool_split="cal")
    pts = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        pol = make(**kw)
        if pol is not None:
            pts.append((kw, pol))

    def score(kws, sds, n):
        out = []
        for i in range(0, len(kws), chunk):
            part = kws[i:i + chunk]
            pol = make_batch(part, n) if make_batch else Stacked([make(**kw) for kw in part], n)
            res = BatchSim(c2, pool, sds * len(part)).run(pol)
            m = res.metrics
            c_all = m["cost"] if label_mode == "oracle" else lagrangian_terms(res, c2, len(m["cost"]), label_mode,
                                                                               **(lab_kw or {}))[0]
            for k in range(len(part)):
                blk = slice(k * n, (k + 1) * n)
                out.append((float(c_all[blk].mean()), float(m["esc_rate"][blk].mean())))
        return out

    for kw, (cost, rate) in zip([kw for kw, _ in pts], score([kw for kw, _ in pts], seeds, n_env)):
        feasible = (rate <= cfg.rho * 1.05) or not budget
        rows.append(dict(**kw, cost=cost, rate=rate, feasible=feasible))
        if feasible and (best is None or cost < best[1]):
            best = (kw, cost)
    if refine_top and best is not None:
        # re-score the best screened points on fresh windows: with many near-equivalent points and few
        # screening windows, the screened winner is often just the luckiest (winner's curse)
        keys_ = list(grid)
        top = sorted([r for r in rows if r["feasible"]], key=lambda r: r["cost"])[:refine_top]
        kws = [{k: r[k] for k in keys_} for r in top]
        rseeds = list(range(seed * 1000 + 500_000, seed * 1000 + 500_000 + refine_env))
        best = None
        for kw, r, (cost, rate) in zip(kws, top, score(kws, rseeds, refine_env)):
            r["refined_cost"], r["refined_rate"] = cost, rate
            if ((rate <= cfg.rho * 1.05) or not budget) and (best is None or cost < best[1]):
                best = (kw, cost)
        if best is None:
            best = (kws[0], top[0]["cost"])
    return best, rows
