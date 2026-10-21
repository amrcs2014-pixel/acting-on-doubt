"""Experiment runner (resumable, multi-process).

  python -m aod.exp E0 --workers 11      # T3 structural verification by exact DP
  python -m aod.exp E1 --workers 11      # main comparison across datasets and energy budgets
  python -m aod.exp E2 --workers 11      # delayed / censored labels
  python -m aod.exp E4 --workers 11      # robustness (evaluation-time shifts, unseen families)
  python -m aod.exp E5 --workers 11      # scale and coordinator ablations
  python -m aod.exp E3 --workers 3       # neural policy (PPO) variants and RL baselines

Every record is one (experiment, dataset, method, seed, setting) evaluated on the SAME 32
held-out test networks driven by test-split flows. Training/tuning uses calibration-split
flows only.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def ignore_console_ctrl():
    """Unattended runs: ignore console Ctrl+C / Ctrl+Break (Windows delivered one to every console process
    after the laptop woke from a battery-triggered hibernation, killing all runs). Stop runs from Task Manager."""
    import signal
    for s in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if s is not None:
            signal.signal(s, signal.SIG_IGN)
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)     # inherited by child processes
    except Exception:
        pass


if os.environ.get("AOD_IGNORE_CTRL") == "1":      # also runs in pool workers, which import this module
    ignore_console_ctrl()

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict

import numpy as np

from .detectors import build_pool, profile_view, ROOT
from .sim import SimConfig, BatchSim
from .policies import ThresholdPolicy, TablePolicy, StaticThreshold, WindowNP, ThreeZone, Never
from .train import train_es, tune_grid, train_aodt_v3
from . import dp as dpm

RAW = os.path.join(ROOT, "results", "raw")
CACHE = os.path.join(ROOT, "results", "cache")
TEST_SEEDS = list(range(1_000_000, 1_000_032))
DATASETS = ["ciciot2023", "edgeiiot", "nfton_v3"]
BUDGETS = {"ciciot2023": [0.001, 0.002, 0.005, 0.01, 0.02], "edgeiiot": [0.002, 0.005, 0.01],
           "nfton_v3": [0.002, 0.005, 0.01]}
WEB = ["commandinjection", "sqlinjection", "xss", "browserhijacking", "uploading_attack", "backdoor_malware"]
METRICS = ["cost", "cdh", "wi", "esc_rate", "energy_x", "esc_blocked", "esc_dropped", "infections",
           "peak_inf", "ttm", "frac_contained", "censor_rate", "contain", "contain_benign"]

_POOLS = {}


def pool_for(ds, excl=()):
    key = (ds, tuple(excl))
    if key not in _POOLS:
        base = build_pool(ds, exclude_families=tuple(excl))
        fams = None
        if excl:  # evaluate only on the unseen families
            fams = list(excl)
        train = profile_view(base)
        # unseen families: sample their test flows, but keep the likelihood tables the policies were tuned on
        _POOLS[key] = (train, profile_view(base, fams, hist_from=train) if fams else None)
    return _POOLS[key]


def summarize(res):
    m = res.metrics
    return {k: float(np.nanmean(m[k])) for k in METRICS if k in m}


def evaluate(policy, pool, cfg, seeds=TEST_SEEDS):
    return summarize(BatchSim(replace(cfg, pool_split="test"), pool, seeds).run(policy))


# ------------------------------------------------------------------ DP tables
LOGH = np.arange(-4.5, -0.49, 0.5)
PRICE = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])


def dp_table(ds, cfg: SimConfig, pool=None, tag=""):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"dp_{ds}_q{cfg.q}_W{cfg.W}_cs{cfg.c_s}_D{cfg.D_iso}{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    pool = pool or pool_for(ds)[0]
    tab = dpm.table(pool, cfg.q, LOGH, PRICE, W=cfg.W, c_fn=cfg.c_fn, c_s=cfg.c_s, D=cfg.D_iso, b0=cfg.b0)
    np.savez(path, **tab)
    return tab


# ------------------------------------------------------------------ methods
GRID = "v3"
REFINE = {"v1": 0, "v2": 0, "v3": 10}   # v3: re-score the 10 best screened points on 64 fresh windows


def _gr():
    """Task kwargs selecting the baseline grid; v1 adds none, so v1 task keys (and .done files) are unchanged."""
    return {} if GRID == "v1" else {"grid": GRID}


def llr_values(pool, cfg):
    """Every distinct positive per-flow log-likelihood ratio the device detector can produce (32 score bins).
    Thresholds placed just below these values enumerate every distinct per-flow rule, on any dataset."""
    from .sim import llr_tables
    lw, _ = llr_tables(pool, cfg.model()[0])
    used = (pool.h_weak_ben > 0) | (pool.h_weak_att > 0)
    return sorted({float(x) - 1e-9 for x in cfg.temp * lw[used] if x > 0})


def make_baselines(cfg, pool=None, grid="v1"):
    """grid v1: fixed absolute grids (as first run). v2: per-flow thresholds enumerate every distinct detector
    LLR value (v1's fixed grid missed all useful thresholds on NF-ToN-IoT-v3, whose largest LLR is 1.48),
    and wider grids for the window and belief rules."""
    if grid in ("v2", "v3"):
        vals = llr_values(pool, cfg)
        return {
            "Static threshold": (lambda tau: StaticThreshold(tau), {"tau": vals}, False),
            "Window NP": (lambda w, tau: WindowNP(tau, w),
                          {"w": [5, 10, 20], "tau": [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12]}, False),
            "Belief stopping": (lambda bC: ThresholdPolicy([bC, 0, 0, -30, 0, 0], no_escalate=True, congestion=False),
                                {"bC": list(np.arange(-6.0, 6.01, 0.25))}, False),
            "Three-zone deferral": (lambda l, u, v: ThreeZone(l, u, v) if l < u else None,
                                    {"l": vals, "u": vals + [99.0], "v": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]}, True),
        }
    return {
        "Static threshold": (lambda tau: StaticThreshold(tau),
                             {"tau": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]}, False),
        "Window NP": (lambda w, tau: WindowNP(tau, w),
                      {"w": [5, 10, 20], "tau": [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]}, False),
        # theta=[bC,0,0,-30,0,0] makes the contain threshold exactly bC (softplus gap ~ 0)
        "Belief stopping": (lambda bC: ThresholdPolicy([bC, 0, 0, -30, 0, 0], no_escalate=True, congestion=False),
                            {"bC": list(np.arange(-6.0, 6.01, 0.5))}, False),
        "Three-zone deferral": (lambda l, u, v: ThreeZone(l, u, v) if l < u else None,
                                {"l": [0.25, 0.5, 1.0, 1.5, 2.5], "u": [1.5, 2.0, 3.0, 4.0, 6.0, 99.0],
                                 "v": [0.0, 1.0, 2.0, 4.0]}, True),
    }


def rebuild(method, params, cfg, tab=None):
    if method == "Static threshold":
        return StaticThreshold(params["tau"])
    if method == "Window NP":
        return WindowNP(params["tau"], params["w"])
    if method == "Belief stopping":
        return ThresholdPolicy([params["bC"], 0, 0, -30, 0, 0], no_escalate=True, congestion=False)
    if method == "Three-zone deferral":
        return ThreeZone(params["l"], params["u"], params["v"])
    if method.startswith("AoD-T"):
        return ThresholdPolicy(params["theta"], lam=params["lam"], congestion=params.get("congestion", True))
    if method.startswith("AoD-DP"):
        return TablePolicy(tab, lam=params["lam"], congestion=params.get("congestion", True))
    if method.startswith("Belief two-threshold"):
        th = [params["bE"], 0, 0, np.log(np.expm1(params["gap"])), 0, 0] + ([params["fb"]] if "fb" in params else [])
        return ThresholdPolicy(th, congestion=False)
    if method == "No response":
        return Never()
    raise KeyError(method)


def theta_batch(theta_of, no_escalate=False):
    """Vectorised ThresholdPolicy for tune_grid: one parameter vector per block of n_env environments."""
    def mk(kws, n_env):
        th = np.repeat(np.array([theta_of(**kw) for kw in kws], float), n_env, axis=0)
        return ThresholdPolicy(th, lam=np.zeros(len(th)), congestion=False, no_escalate=no_escalate)
    return mk


def _col(kws, key, n_env):
    """One parameter per block of n_env environments, shaped (B, 1) to broadcast over devices."""
    return np.repeat(np.array([kw[key] for kw in kws], float), n_env)[:, None]


BATCH = {"Belief stopping": theta_batch(lambda bC: [bC, 0, 0, -30, 0, 0], no_escalate=True),
         # the per-flow rules are elementwise in their thresholds, so (B,1) parameter arrays vectorise them
         "Static threshold": lambda kws, n: StaticThreshold(_col(kws, "tau", n)),
         "Three-zone deferral": lambda kws, n: ThreeZone(_col(kws, "l", n), _col(kws, "u", n), _col(kws, "v", n))}


def task_baselines(ds, seed, cfg_kw, exp, excl=(), grid="v1"):
    cfg = SimConfig(**cfg_kw)
    pool, evalpool = pool_for(ds, excl)
    out = []
    for name, (make, grd, esc) in make_baselines(cfg, pool, grid).items():
        n_env = 8
        best, rows = tune_grid(make, grd, pool, cfg, n_env=n_env, seed=1000 + seed, budget=esc,
                               make_batch=BATCH.get(name), refine_top=REFINE[grid])
        if best is None:
            continue
        pol = make(**best[0])
        rec = dict(exp=exp, ds=ds, method=name, seed=seed, params={k: float(v) for k, v in best[0].items()},
                   **{"cfg_" + k: v for k, v in cfg_kw.items()}, excl=list(excl), grid=grid,
                   grid_size=sum(1 for r in rows), n_feasible=sum(1 for r in rows if r["feasible"]))
        rec.update(evaluate(pol, evalpool or pool, cfg))
        out.append(rec)
    if seed == 0:
        rec = dict(exp=exp, ds=ds, method="No response", seed=0, params={}, grid=grid,
                   **{"cfg_" + k: v for k, v in cfg_kw.items()}, excl=list(excl))
        rec.update(evaluate(Never(), evalpool or pool, cfg)); out.append(rec)
    return out


def task_belief2(ds, seed, cfg_kw, exp, excl=(), grid="v1"):
    """AoD's own two-threshold belief rule, grid-tuned with FULL ground truth (constant thresholds)."""
    cfg = SimConfig(**cfg_kw)
    pool, evalpool = pool_for(ds, excl)
    make = lambda bE, gap: ThresholdPolicy([bE, 0, 0, np.log(np.expm1(gap)), 0, 0], congestion=False)
    grd = {"bE": list(np.arange(-6.0, 2.01, 0.5)), "gap": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]}
    best, rows = tune_grid(make, grd, pool, cfg, n_env=8, seed=1000 + seed, budget=True,
                           make_batch=theta_batch(lambda bE, gap: [bE, 0, 0, np.log(np.expm1(gap)), 0, 0]),
                           refine_top=REFINE[grid])
    rec = dict(exp=exp, ds=ds, method="Belief two-threshold (oracle-tuned)", seed=seed, grid=grid,
               params={k: float(v) for k, v in best[0].items()},
               **{"cfg_" + k: v for k, v in cfg_kw.items()}, excl=list(excl))
    rec.update(evaluate(make(**best[0]), evalpool or pool, cfg))
    return [rec]


def tasks_E1b(seeds=range(10)):
    return [(task_belief2, dict(**_gr(), ds=ds, seed=s, cfg_kw=dict(rho=rho), exp="E1b"))
            for ds in DATASETS for rho in BUDGETS[ds] for s in seeds]


TRAINERS = {"v1": dict(),                                   # constant Adam step, average of last 20 iterates
            "v2": dict(lr_end=0.006, avg_last=40),          # decaying step, average of last 40 (improvement loop)
            "v3": None}                                     # train.train_aodt_v3: warm-start grid + ES + selection
TRAINER = "v3"


def _tr():
    """Task kwargs selecting the trainer; v1 adds none, so v1 task keys (and .done files) are unchanged."""
    return {} if TRAINER == "v1" else {"trainer": TRAINER}


def task_aodt(ds, seed, cfg_kw, exp, label_mode="ipcw", lab_kw=None, iters=120, excl=(), method=None,
              congestion=True, trainer="v1"):
    cfg = SimConfig(**cfg_kw)
    pool, evalpool = pool_for(ds, excl)
    t0 = time.time()
    info = {}
    if trainer == "v3":
        th, lam, hist, info = train_aodt_v3(pool, cfg, label_mode=label_mode, seed=seed, lab_kw=lab_kw or {},
                                            congestion=congestion)
    else:
        th, lam, hist = train_es(pool, cfg, label_mode=label_mode, iters=iters, seed=seed, lab_kw=lab_kw or {},
                                 congestion=congestion, **TRAINERS[trainer])
    th = np.asarray(th, float)
    pol = ThresholdPolicy(th, lam=lam, congestion=congestion)
    rec = dict(exp=exp, ds=ds, method=method or f"AoD-T ({label_mode})", seed=seed, label_mode=label_mode, trainer=trainer,
               params=dict(theta=th.tolist(), lam=lam, congestion=congestion), lab_kw=lab_kw or {},
               **{"cfg_" + k: v for k, v in cfg_kw.items()}, excl=list(excl), train_s=time.time() - t0,
               curve=[(h["it"], h["cost_true"], h["rate"], h["lam"]) for h in hist],
               train_info=json.loads(json.dumps(info, default=float)))
    rec.update(evaluate(pol, evalpool or pool, cfg))
    return [rec]


def task_aoddp(ds, seed, cfg_kw, exp, excl=()):
    cfg = SimConfig(**cfg_kw)
    pool, evalpool = pool_for(ds, excl)
    tab = dp_table(ds, cfg, pool, tag=("_x" + "-".join(excl)) if excl else "")
    tab = {k: v for k, v in tab.items() if k != "bC_noE"}      # same policy class as AoD-T (no fallback)
    c2 = replace(cfg, pool_split="cal")
    seeds = list(range(2000 * (seed + 1), 2000 * (seed + 1) + 8))
    lo, hi = 0.0, 5.0
    r0 = float(BatchSim(c2, pool, seeds).run(TablePolicy(tab, lam=0.0)).metrics["esc_rate"].mean())
    lam = 0.0
    if r0 > cfg.rho:
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            r = float(BatchSim(c2, pool, seeds).run(TablePolicy(tab, lam=mid)).metrics["esc_rate"].mean())
            if r > cfg.rho:
                lo = mid
            else:
                hi = mid
        lam = hi
    rec = dict(exp=exp, ds=ds, method="AoD-DP (model-based)", seed=seed, params=dict(lam=lam),
               **{"cfg_" + k: v for k, v in cfg_kw.items()}, excl=list(excl))
    rec.update(evaluate(TablePolicy(tab, lam=lam), evalpool or pool, cfg))
    return [rec]


def task_dp_point(ds, q, h, price, c_s, W):
    pool = pool_for(ds)[0]
    r = dpm.solve(pool, q, h, price, W=W, c_s=c_s)
    ok, bE, bC, sw, hasE = dpm.structure(r["act"])
    return [dict(exp="E0", ds=ds, q=q, h=h, price=price, c_s=c_s, W=W, ok=ok, bE=bE, bC=bC, switches=sw,
                 hasE=hasE, iters=r["iters"])]


def task_eval_shift(rec_src, shift_kw, exp, tag, tab_needed=False, per_network=False):
    """Re-evaluate a trained/tuned policy under an evaluation-time shift."""
    ds = rec_src["ds"]
    cfg_kw = {k[4:]: v for k, v in rec_src.items() if k.startswith("cfg_")}
    cfg = replace(SimConfig(**cfg_kw), **shift_kw)
    pool, _ = pool_for(ds)
    tab = dp_table(ds, SimConfig(**cfg_kw), pool) if rec_src["method"].startswith("AoD-DP") else None
    if tab is not None:
        tab = {k: v for k, v in tab.items() if k != "bC_noE"}
    params = dict(rec_src["params"])
    if "congestion_off" in shift_kw:
        pass
    pol = rebuild(rec_src["method"], params, cfg, tab)
    rec = dict(exp=exp, ds=ds, method=rec_src["method"], seed=rec_src["seed"], shift=tag,
               shift_kw=shift_kw, params=params, **{"cfg_" + k: v for k, v in cfg_kw.items()},
               **{k: rec_src[k] for k in ("trainer", "grid") if k in rec_src})
    if per_network:   # keep each test network's cost, for a bootstrap over networks as well as seeds
        res = BatchSim(replace(cfg, pool_split="test"), pool, TEST_SEEDS).run(pol)
        rec.update(summarize(res), net_cost=[float(x) for x in res.metrics["cost"]])
    else:
        rec.update(evaluate(pol, pool, cfg))
    return [rec]


def tasks_E1net():
    """Main-table policies (rho=0.005, all datasets) re-evaluated network by network."""
    src = [r for r in load("E1") + load("E1aod") + load("E1b") if abs(r.get("cfg_rho", 0) - 0.005) < 1e-9
           and r["method"] != "No response"]
    return [(task_eval_shift, dict(rec_src=r, shift_kw={}, exp="E1net", tag="none", per_network=True)) for r in src]


def task_ppo(ds, seed, cfg_kw, exp, variant, iters=100, n_env=6):
    """Neural variants. variant in {"AoD-N", "AoD-N (no shaping)", "PPO no escalation", "PPO fixed penalty"}."""
    import torch
    torch.set_num_threads(1)
    from .ppo import train_ppo, NeuralPolicy
    cfg = SimConfig(**cfg_kw)
    pool, _ = pool_for(ds)
    kw = dict(label_mode="ipcw", shaping=True, allow_E=True, constraint="lagrangian")
    if variant == "AoD-N (no shaping)":
        kw.update(shaping=False)
    elif variant == "PPO no escalation":
        kw.update(label_mode="oracle", allow_E=False)
    elif variant == "PPO fixed penalty":
        kw.update(label_mode="oracle", constraint="penalty", lam_fix=0.5)
    t0 = time.time()
    net, lam, hist = train_ppo(pool, cfg, iters=iters, n_env=n_env, seed=seed, **kw)
    os.makedirs(os.path.join(ROOT, "results", "models"), exist_ok=True)
    torch.save(net.state_dict(), os.path.join(ROOT, "results", "models", f"{variant.replace(' ', '_')}_{ds}_s{seed}.pt"))
    pol = NeuralPolicy(net, lam=lam, allow_E=kw["allow_E"], greedy=True, record=True)
    res = BatchSim(replace(cfg, pool_split="test"), pool, TEST_SEEDS).run(pol)
    rec = dict(exp=exp, ds=ds, method=variant, seed=seed, params=dict(lam=lam), train_s=time.time() - t0,
               **{"cfg_" + k: v for k, v in cfg_kw.items()},
               curve=[(h["it"], h["cost"], h["rate"], h["lam"]) for h in hist])
    rec.update(summarize(res))
    # structure: how well does a two-threshold rule on the belief reproduce the greedy actions?
    A = np.stack(pol.buf["a"], 1).reshape(-1); ON = np.stack(pol.buf["online"], 1).reshape(-1)
    LO = np.stack(pol.buf["lo"], 1).reshape(-1)
    A, LO = A[ON], LO[ON]
    sub = np.random.default_rng(0).choice(len(A), min(len(A), 400000), replace=False)
    A, LO = A[sub], LO[sub]
    grid = np.arange(-12, 12.01, 0.25)
    best = (0.0, None, None)
    order = np.argsort(LO); LOs, As = LO[order], A[order]
    n = len(As)
    cW = np.concatenate([[0], np.cumsum(As == 0)]); cE = np.concatenate([[0], np.cumsum(As == 1)])
    cC = np.concatenate([[0], np.cumsum(As == 2)])
    idx = np.searchsorted(LOs, grid)
    for i, e in enumerate(idx):
        for c in idx[i:]:
            agree = cW[e] + (cE[c] - cE[e]) + (cC[n] - cC[c])
            if agree > best[0]:
                best = (agree, float(LOs[min(e, n - 1)]), float(LOs[min(c, n - 1)]))
    rec.update(struct_agree=float(best[0] / n), struct_bE=best[1], struct_bC=best[2],
               frac_W=float((A == 0).mean()), frac_E=float((A == 1).mean()), frac_C=float((A == 2).mean()))
    return [rec]


def tasks_E3(seeds=range(5)):
    t = []
    for v in ["AoD-N", "AoD-N (no shaping)", "PPO no escalation", "PPO fixed penalty"]:
        for s in seeds:
            t.append((task_ppo, dict(ds="ciciot2023", seed=s, cfg_kw=dict(rho=0.005), exp="E3", variant=v)))
    return t


# ------------------------------------------------------------------ plumbing
def key_of(t):
    fn, kw = t
    return json.dumps([fn.__name__, kw], sort_keys=True, default=str)


def run_tasks(exp, tasks, workers):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{exp}.jsonl")
    done_path = os.path.join(RAW, f"{exp}.done")
    done = set(open(done_path, encoding="utf8").read().splitlines()) if os.path.exists(done_path) else set()
    todo = [t for t in tasks if key_of(t) not in done]
    print(f"{exp}: {len(tasks)} tasks, {len(todo)} to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(t[0], **t[1]): t for t in todo}
        for i, f in enumerate(as_completed(futs)):
            t = futs[f]
            try:
                recs = f.result()
            except Exception:
                print("FAILED", key_of(t), traceback.format_exc(), flush=True)
                continue
            with open(path, "a", encoding="utf8") as fo:
                for r in recs:
                    fo.write(json.dumps(r, default=float) + "\n")
            with open(done_path, "a", encoding="utf8") as fo:
                fo.write(key_of(t) + "\n")
            print(f"  [{i + 1}/{len(todo)}] {t[0].__name__} {t[1].get('ds', '')} seed={t[1].get('seed', '')} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)


BASELINES = ("Static threshold", "Window NP", "Belief stopping", "Three-zone deferral", "No response",
             "Belief two-threshold (oracle-tuned)")


def is_current(r):
    """False for records superseded by the improvement loop (older AoD-T trainer or baseline grid)."""
    if r.get("method", "").startswith("AoD-T"):
        return r.get("trainer", "v1") == TRAINER
    if r.get("method") in BASELINES:
        return r.get("grid", "v1") == GRID
    return True


def load(exp, current=True):
    path = os.path.join(RAW, f"{exp}.jsonl")
    if not os.path.exists(path):
        return []
    recs = [json.loads(l) for l in open(path, encoding="utf8")]
    return [r for r in recs if is_current(r)] if current else recs


def tasks_E0():
    t = []
    for ds in DATASETS:
        for q in [0.1, 0.3, 0.5]:
            for h in [1e-4, 1e-3, 1e-2]:
                for price in [0.0, 0.25, 0.5, 1.0, 2.0]:
                    for c_s in [0.1, 0.2, 0.5]:
                        for W in [3, 5]:
                            t.append((task_dp_point, dict(ds=ds, q=q, h=h, price=price, c_s=c_s, W=W)))
    return t


def tasks_E1(seeds=range(10)):
    t = []
    for ds in DATASETS:
        for rho in BUDGETS[ds]:
            kw = dict(rho=rho)
            for s in seeds:
                t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=kw, exp="E1")))
                t.append((task_baselines, dict(**_gr(), ds=ds, seed=s, cfg_kw=kw, exp="E1")))
                t.append((task_aoddp, dict(ds=ds, seed=s, cfg_kw=kw, exp="E1")))
    # cheap tasks last so long ES runs start first
    return sorted(t, key=lambda x: 0 if x[0] is task_aodt else 1)


def tasks_E2(seeds=range(5)):
    t = []
    ds, kw0 = "ciciot2023", dict(rho=0.005)
    for med in [100.0, 300.0, 1000.0]:
        for pn in [0.0, 0.2, 0.4]:
            kw = dict(kw0, delay_median=med, p_never=pn)
            for mode in ["naive", "ipcw"]:
                for s in seeds:
                    t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=kw, exp="E2", label_mode=mode)))
    for s in seeds:
        t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=dict(kw0), exp="E2", label_mode="oracle")))
    # misspecified label model (defender's G-hat wrong) and A4 violation (stealth-dependent delays)
    for name, lab_kw, extra in [("pnever-0.1", dict(p_never_hat=0.1), {}), ("pnever+0.2", dict(p_never_hat=0.4), {}),
                                ("median x0.5", dict(median_hat=150.0), {}), ("median x2", dict(median_hat=600.0), {}),
                                ("A4 violated", {}, dict(stealth_delay=True, q=0.15))]:
        for s in seeds:
            kw = dict(kw0, **extra)
            t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=kw, exp="E2", label_mode="ipcw", lab_kw=lab_kw,
                                      method=f"AoD-T (ipcw, {name})")))
            if extra:
                t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=kw, exp="E2", label_mode="naive",
                                          method=f"AoD-T (naive, {name})")))
                t.append((task_aodt, dict(**_tr(), ds=ds, seed=s, cfg_kw=kw, exp="E2", label_mode="oracle",
                                          method=f"AoD-T (oracle, {name})")))
    return t


def tasks_E4(seeds=range(5)):
    """Evaluation-time shifts of the E1 policies (CICIoT2023, rho=0.005) + unseen families."""
    src = [r for r in load("E1") + load("E1aod") + load("E1b") if r["ds"] == "ciciot2023" and abs(r.get("cfg_rho", 0) - 0.005) < 1e-9
           and r["method"] != "No response"]
    shifts = {"none": dict(), "temp x0.5": dict(temp=0.5), "temp x2": dict(temp=2.0), "stealth q=0.15": dict(q=0.15, q_hat=0.3),
              "aggressive q=0.5": dict(q=0.5, q_hat=0.3), "drain 25%": dict(drain_frac=0.25),
              "drain 50%": dict(drain_frac=0.5), "faster spread p1x3": dict(p1=0.03, p1_hat=0.01)}
    t = []
    for r in src:
        for tag, kw in shifts.items():
            t.append((task_eval_shift, dict(rec_src=r, shift_kw=kw, exp="E4", tag=tag)))
    for s in seeds:  # unseen web-exploit families: detectors and policies never saw them
        kw = dict(rho=0.005)
        t.append((task_aodt, dict(**_tr(), ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4", excl=tuple(WEB),
                                  method="AoD-T (ipcw)")))
        t.append((task_baselines, dict(**_gr(), ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4", excl=tuple(WEB))))
        t.append((task_aoddp, dict(ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4", excl=tuple(WEB))))
    return t


def tasks_E4u(seeds=range(5)):
    """Unseen families, rerun after fixing the evaluation view (it rebuilt the likelihood tables from the
    removed families' calibration flows; the policies now keep the tables they were tuned on)."""
    t = []
    kw = dict(rho=0.005)
    for s in seeds:
        t.append((task_aodt, dict(**_tr(), ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4u", excl=tuple(WEB),
                                  method="AoD-T (ipcw)")))
        t.append((task_baselines, dict(**_gr(), ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4u", excl=tuple(WEB))))
        t.append((task_aoddp, dict(ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4u", excl=tuple(WEB))))
        t.append((task_belief2, dict(**_gr(), ds="ciciot2023", seed=s, cfg_kw=kw, exp="E4u", excl=tuple(WEB))))
    return t


def tasks_E4lat():
    """Verdict latency: the gateway verdict arrives d ticks after the escalation is served."""
    src = [r for r in load("E1") + load("E1aod") + load("E1b") if r["ds"] == "ciciot2023" and abs(r.get("cfg_rho", 0) - 0.005) < 1e-9
           and r["method"] != "No response"]
    return [(task_eval_shift, dict(rec_src=r, shift_kw=dict(verdict_delay=d), exp="E4lat", tag=f"verdict latency {d}"))
            for r in src for d in (5, 10)]


def tasks_E5():
    src = [r for r in load("E1") + load("E1aod") + load("E1b") if r["ds"] == "ciciot2023" and abs(r.get("cfg_rho", 0) - 0.005) < 1e-9
           and r["method"] not in ("No response",)]
    t = []
    for r in src:
        t.append((task_eval_shift, dict(rec_src=r, shift_kw=dict(), exp="E5", tag="N=64 (none)")))
        for N in [16, 256, 512]:
            t.append((task_eval_shift, dict(rec_src=r, shift_kw=dict(N=N, G=N // 8), exp="E5", tag=f"N={N}")))
        if r["method"].startswith("AoD"):
            t.append((task_eval_shift, dict(rec_src=r, shift_kw=dict(broadcast_hazard=False), exp="E5",
                                            tag="no hazard broadcast")))
            t.append((task_eval_shift, dict(rec_src=r, shift_kw=dict(congestion=False), exp="E5",
                                            tag="no congestion price")))
            t.append((task_eval_shift, dict(rec_src=r, shift_kw=dict(bucket=1e9), exp="E5", tag="no token bucket")))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--skip", default="", help="comma-separated task function names to leave for later")
    ap.add_argument("--out", default=None, help="write results/raw/<OUT>.jsonl|.done instead of <EXP> (lets two runners of one experiment run side by side without sharing files)")
    a = ap.parse_args()
    tasks = {"E0": tasks_E0, "E1": tasks_E1, "E1b": tasks_E1b, "E2": tasks_E2, "E3": tasks_E3, "E4": tasks_E4, "E4u": tasks_E4u, "E4lat": tasks_E4lat, "E1net": tasks_E1net,
             "E5": tasks_E5}[a.exp]()
    skip = set(filter(None, a.skip.split(",")))
    tasks = [t for t in tasks if t[0].__name__ not in skip]
    run_tasks(a.out or a.exp, tasks, a.workers)
