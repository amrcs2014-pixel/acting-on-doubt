"""Exact dynamic programming for one device (the T3 setting).

Belief MDP on a log-odds grid. Per tick, at decision time the device holds log-odds x
(after its local observation). Actions:
  W: pay c_fn*sigma(x); next tick predict with hazard h and observe a local bin.
  E: as W plus price pi; the gateway verdict (sum of W per-flow strong LLRs) moves x first.
  C: isolate D ticks (c_s each), reflash, restart from b0.
Observations are LLR-valued, so the monotone-likelihood-ratio condition (A2) holds by
construction.
"""
from __future__ import annotations

import numpy as np

from .detectors import ScorePool
from .sim import llr_tables

XG = np.linspace(-16.0, 16.0, 641)          # log-odds grid


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def verdict_dist(pool: ScorePool, q: float, W: int, res: float = 0.05):
    """Support and probabilities (healthy, compromised) of the W-flow verdict LLR sum."""
    _, ls = llr_tables(pool, q)
    g0 = pool.h_strong_ben
    lo, hi = W * ls.min() - 1, W * ls.max() + 1
    n = int(np.ceil((hi - lo) / res)) + 1
    one = np.zeros(n)
    np.add.at(one, np.round((ls - lo / W) / res).astype(int), g0)   # per-flow, offset lo/W
    d = one.copy()
    for _ in range(W - 1):
        d = np.convolve(d, one)[:n]
    s = lo + np.arange(n) * res
    keep = d > 1e-12
    s, p0 = s[keep], d[keep] / d[keep].sum()
    p1 = p0 * np.exp(s)                      # exact LR identity, renormalised for rounding
    p1 = p1 / p1.sum()
    return s, p0, p1


def solve(pool: ScorePool, q: float, h: float, price: float, W: int = 5, c_fn: float = 1.0,
          c_s: float = 0.2, D: int = 30, b0: float = 1e-3, gamma: float = 0.99,
          iters: int = 3000, tol: float = 1e-7, allow_E: bool = True, verdict=None, V0=None):
    lw, _ = llr_tables(pool, q)
    f0 = pool.h_weak_ben
    f1 = q * pool.h_weak_att + (1 - q) * pool.h_weak_ben
    s, p0v, p1v = verdict if verdict is not None else verdict_dist(pool, q, W)
    x = XG
    px = sig(x)
    ppred = px + (1 - px) * h
    xpred = logit(ppred)
    Pbin = ppred[:, None] * f1[None, :] + (1 - ppred[:, None]) * f0[None, :]       # (X, bins)
    Xnext = np.clip(xpred[:, None] + lw[None, :], x[0], x[-1])                    # (X, bins)
    Ps = px[:, None] * p1v[None, :] + (1 - px[:, None]) * p0v[None, :]           # (X, S)
    Xs = np.clip(x[:, None] + s[None, :], x[0], x[-1])                           # (X, S)
    xb0 = logit(b0)
    iso_cost = c_s * (1 - gamma ** D) / (1 - gamma)
    V = np.zeros_like(x) if V0 is None else V0.copy()
    for it in range(iters):
        Vn = np.interp(Xnext, x, V)                       # V at next decision, per bin
        Vc = (Pbin * Vn).sum(1)                           # continuation value from x
        QW = c_fn * px + gamma * Vc
        Vc_s = np.interp(Xs, x, Vc)
        QE = c_fn * px + price + gamma * (Ps * Vc_s).sum(1) if allow_E else np.full_like(x, np.inf)
        QC = iso_cost + gamma ** D * np.interp(xb0, x, Vc) * np.ones_like(x)
        Q = np.stack([QW, QE, QC])
        Vnew = Q.min(0)
        if np.max(np.abs(Vnew - V)) < tol:
            V = Vnew
            break
        V = Vnew
    # tie-break within solver tolerance: prefer W, then C; escalate only if strictly better
    Qmin = Q.min(0)
    tie = 1e-6 * (1.0 + np.abs(Qmin))
    act = np.where(Q[0] <= Qmin + tie, 0, np.where(Q[2] <= Qmin + tie, 2, 1))
    return dict(x=x, V=V, Q=Q, act=act, iters=it + 1)


def evaluate(pool: ScorePool, q: float, h: float, price: float, act: np.ndarray, W: int = 5, c_fn: float = 1.0,
             c_s: float = 0.2, D: int = 30, b0: float = 1e-3, gamma: float = 0.99, iters: int = 4000,
             tol: float = 1e-8, verdict=None):
    """Exact value of a fixed deterministic grid policy `act` (0=W, 1=E, 2=C)."""
    lw, _ = llr_tables(pool, q)
    f0 = pool.h_weak_ben
    f1 = q * pool.h_weak_att + (1 - q) * pool.h_weak_ben
    s, p0v, p1v = verdict if verdict is not None else verdict_dist(pool, q, W)
    x = XG; px = sig(x)
    ppred = px + (1 - px) * h
    Pbin = ppred[:, None] * f1[None, :] + (1 - ppred[:, None]) * f0[None, :]
    Xnext = np.clip(logit(ppred)[:, None] + lw[None, :], x[0], x[-1])
    Ps = px[:, None] * p1v[None, :] + (1 - px[:, None]) * p0v[None, :]
    Xs = np.clip(x[:, None] + s[None, :], x[0], x[-1])
    xb0 = logit(b0)
    iso_cost = c_s * (1 - gamma ** D) / (1 - gamma)
    V = np.zeros_like(x)
    for _ in range(iters):
        Vc = (Pbin * np.interp(Xnext, x, V)).sum(1)
        QW = c_fn * px + gamma * Vc
        QE = c_fn * px + price + gamma * (Ps * np.interp(Xs, x, Vc)).sum(1)
        QC = iso_cost + gamma ** D * np.interp(xb0, x, Vc)
        Vn = np.where(act == 0, QW, np.where(act == 1, QE, QC))
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    Vc = (Pbin * np.interp(Xnext, x, V)).sum(1)
    return float(np.interp(xb0, x, Vc)), V          # value from a fresh (just reflashed) device


def two_threshold_gap(pool, q, h, price, W=5, c_s=0.2, **kw):
    """Relative value lost by the best of a few two-threshold repairs of the optimal policy."""
    r = solve(pool, q, h, price, W=W, c_s=c_s, **kw)
    a = r["act"]; x = XG
    ver = verdict_dist(pool, q, W)
    v_opt, _ = evaluate(pool, q, h, price, a, W=W, c_s=c_s, verdict=ver)
    has_C = (a == 2).any()
    iC = int(np.argmax(a == 2)) if has_C else len(a)
    iE = int(np.argmax(a == 1)) if (a == 1).any() else iC
    cands = []
    c1 = np.where(np.arange(len(a)) < iE, 0, np.where(np.arange(len(a)) < iC, 1, 2)); cands.append(c1)   # fill gaps with E
    c2 = np.where(np.arange(len(a)) < iC, 0, 2); cands.append(c2)                                           # no E at all
    last_E = int(np.max(np.where(a == 1)[0])) if (a == 1).any() else iE
    c3 = np.where(np.arange(len(a)) < iE, 0, np.where(np.arange(len(a)) <= last_E, 1, 2)); cands.append(c3)  # contain right after E band
    evs = [evaluate(pool, q, h, price, c, W=W, c_s=c_s, verdict=ver) for c in cands]
    vals = [e[0] for e in evs]
    k = int(np.argmin(vals)); best = vals[k]
    _, V_opt_grid = evaluate(pool, q, h, price, a, W=W, c_s=c_s, verdict=ver)
    sup = float(np.max((evs[k][1] - V_opt_grid) / np.maximum(np.abs(V_opt_grid), 1e-9)))
    return dict(v_opt=v_opt, v_two=best, rel_gap=(best - v_opt) / abs(v_opt) if v_opt else 0.0, sup_rel_gap=sup)


def structure(act: np.ndarray, x: np.ndarray = XG):
    """Check the W* E* C* ordering. Returns (ok, beta_E, beta_C, n_switches)."""
    sw = int((np.diff(act) != 0).sum())
    ok = bool(np.all(np.diff(act) >= 0))
    has_E = (act == 1).any()
    has_C = (act == 2).any()
    bC = float(x[np.argmax(act == 2)]) if has_C else np.inf
    bE = float(x[np.argmax(act >= 1)]) if (has_E or has_C) else np.inf
    return ok, bE, bC, sw, bool(has_E)


def table(pool: ScorePool, q: float, logh_grid, price_grid, **kw):
    """Threshold table for TablePolicy over (log10 h, price)."""
    ver = verdict_dist(pool, q, kw.get("W", 5))
    nh, npr = len(logh_grid), len(price_grid)
    bE = np.zeros((nh, npr)); bC = np.zeros((nh, npr)); ok = np.zeros((nh, npr), bool); hasE = np.zeros((nh, npr), bool)
    for i, lh in enumerate(logh_grid):
        V0 = None
        for j, pr in enumerate(price_grid):
            r = solve(pool, q, 10 ** lh, pr, verdict=ver, V0=V0, **kw)
            V0 = r["V"]
            o, e, c, _, he = structure(r["act"])
            bE[i, j], bC[i, j], ok[i, j], hasE[i, j] = min(e, 40.0), min(c, 40.0), o, he
    # token-aware fallback: with the bucket empty, escalation is unavailable (optimal stopping)
    bC0 = np.zeros(nh)
    for i, lh in enumerate(logh_grid):
        r = solve(pool, q, 10 ** lh, 0.0, verdict=ver, allow_E=False, **kw)
        _, _, c, _, _ = structure(r["act"])
        bC0[i] = min(c, 40.0)
    return dict(logh=np.asarray(logh_grid, float), price=np.asarray(price_grid, float), bE=bE, bC=bC, ok=ok, hasE=hasE,
                bC_noE=bC0)
