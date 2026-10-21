"""Turn results/raw/*.jsonl into the paper's figures (figures/*.pdf) and tables (paper/tables/*.tex).

Every number in the paper's results section is produced here.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .detectors import ROOT
from .stats import iqm, boot_ci, paired, holm, unpaired, boot_diff_two_level

RAW = os.path.join(ROOT, "results", "raw")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

DS_NAME = {"ciciot2023": "CICIoT2023", "edgeiiot": "Edge-IIoTset", "nfton_v3": "NF-ToN-IoT-v3"}
ORDER = ["AoD-T (ipcw)", "AoD-DP (model-based)", "Belief two-threshold (oracle-tuned)", "Three-zone deferral",
         "Belief stopping", "Window NP", "Static threshold"]
SHORT = {"AoD-T (ipcw)": "AoD-T", "AoD-DP (model-based)": "AoD-DP",
         "Belief two-threshold (oracle-tuned)": "Two-thr. (oracle)", "Three-zone deferral": "Three-zone",
         "Belief stopping": "Belief stop", "Window NP": "Window NP", "Static threshold": "Static",
         "No response": "No response"}
COL = {"AoD-T (ipcw)": "#1f4e86", "AoD-DP (model-based)": "#5b8fd1", "Belief two-threshold (oracle-tuned)": "#17becf", "Three-zone deferral": "#9a5b00",
       "Belief stopping": "#06704a", "Window NP": "#8e44ad", "Static threshold": "#7f7f7f"}
MARK = {"AoD-T (ipcw)": "o", "AoD-DP (model-based)": "s", "Belief two-threshold (oracle-tuned)": "P", "Three-zone deferral": "^", "Belief stopping": "D",
        "Window NP": "v", "Static threshold": "x"}


def load(exp):
    """Current records only: results superseded in the improvement loop (an older AoD-T trainer or baseline
    grid) stay in results/raw for the record but are never reported."""
    from .exp import is_current
    p = os.path.join(RAW, f"{exp}.jsonl")
    return [r for r in (json.loads(l) for l in open(p, encoding="utf8")) if is_current(r)] if os.path.exists(p) else []


def by(recs, *keys):
    d = defaultdict(list)
    for r in recs:
        d[tuple(r.get(k) for k in keys)].append(r)
    return d


def fmt(x, ci=None, nd=2):
    if ci is None:
        return f"{x:.{nd}f}"
    return f"{x:.{nd}f} \\scriptsize[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


# --------------------------------------------------------------------------- E0
def e0():
    R = load("E0")
    if not R:
        return {}
    ok = np.mean([r["ok"] for r in R]); hasE = np.mean([r["hasE"] for r in R])
    out = dict(n=len(R), ok=float(ok), hasE=float(hasE), max_switch=int(max(r["switches"] for r in R)))
    # monotonicity of beta_E in price and of beta_C in c_s, holding other factors fixed
    g = by(R, "ds", "q", "h", "c_s", "W")
    mono_E = []; band_shrink = []
    for k, rs in g.items():
        rs = sorted(rs, key=lambda r: r["price"])
        bE = [min(r["bE"], 40) for r in rs]; bC = [min(r["bC"], 40) for r in rs]
        mono_E.append(all(np.diff(bE) >= -1e-9))
        band = [c - e for e, c in zip(bE, bC)]
        band_shrink.append(all(np.diff(band) <= 1e-9))
    g2 = by(R, "ds", "q", "h", "price", "W")
    mono_C = []
    for k, rs in g2.items():
        rs = sorted(rs, key=lambda r: r["c_s"])
        bC = [min(r["bC"], 40) for r in rs]
        mono_C.append(all(np.diff(bC) >= -1e-9))
    out.update(mono_E=float(np.mean(mono_E)), band_shrink=float(np.mean(band_shrink)), mono_C=float(np.mean(mono_C)),
               n_price_series=len(mono_E), n_cs_series=len(mono_C), n_ok=int(sum(r["ok"] for r in R)))
    # configurations without the W*E*C* ordering: re-solved with the tie-break, and the value lost by the
    # best two-threshold repair (scripts/refine_check.py; also confirmed on a 2x finer grid)
    p = os.path.join(RAW, "E0_recheck.json")
    if os.path.exists(p):
        rc = json.load(open(p))
        ties = [r for r in rc if r["ok_tiebreak"]]
        gen = [r for r in rc if not r["ok_tiebreak"]]
        out.update(n_tie=len(ties), n_nonmono=len(gen), max_rel_gap=float(max(r["rel_gap"] for r in gen)) if gen else 0.0,
                   nonmono_cs=sorted({r["c_s"] for r in gen}), nonmono_price_pos=all(r["price"] > 0 for r in gen))
    json.dump(out, open(os.path.join(TAB, "e0_summary.json"), "w"), indent=1)
    return out


def fig_policy_map(ds="ciciot2023"):
    """Optimal action over (belief log-odds, price) from the cached DP table grid (fixed h)."""
    from . import dp as dpm
    from .exp import pool_for
    from .sim import SimConfig
    cfg = SimConfig()
    pool = pool_for(ds)[0]
    prices = np.linspace(0, 2.5, 26)
    ver = dpm.verdict_dist(pool, cfg.q, cfg.W)
    acts, vois = [], None
    V0 = None
    for pr in prices:
        r = dpm.solve(pool, cfg.q, 1e-3, pr, W=cfg.W, verdict=ver, V0=V0)
        V0 = r["V"]; acts.append(r["act"])
        if abs(pr - 0.5) < 1e-9:
            Q = r["Q"]; vois = (Q[1] - Q[0])  # Q_E - Q_W = price - gamma*VOI
            vois = (pr - vois) / 0.99
    A = np.array(acts)
    x = dpm.XG
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#e8eef6", "#f2c46b", "#c0392b"])
    sel = (x >= -8) & (x <= 4)
    ax[0].pcolormesh(x[sel], prices, A[:, sel], cmap=cm, vmin=0, vmax=2, shading="nearest", rasterized=True)
    ax[0].set_xlabel("belief (log-odds)"); ax[0].set_ylabel("escalation price $\\pi$")
    for lab, xx, yy, c in [("wait", -6.5, 1.8, "#1f2d3d"), ("escalate", -2.2, 0.25, "#1f2d3d"), ("contain", 1.8, 1.8, "white")]:
        ax[0].text(xx, yy, lab, color=c, fontsize=8, ha="center", fontweight="bold")
    ax[0].set_title(f"(a) Optimal device action, {DS_NAME[ds]}, $h=10^{{-3}}$")
    xs = x[sel]
    ax[1].plot(xs, 0.99 * vois[sel], color="#1f4e86", lw=1.2, label="$\\gamma\\,$VOI$(b)$")
    ax[1].axhline(0.5, color="#9a5b00", lw=1, ls="--", label="price $\\pi=0.5$")
    ax[1].set_xlabel("belief (log-odds)"); ax[1].set_ylabel("value")
    ax[1].legend(frameon=False); ax[1].set_title("(b) Escalate where $\\gamma\\,$VOI $\\geq\\pi$ (Thm. 2)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "policy_map.pdf")); plt.close(fig)


# --------------------------------------------------------------------------- E1
def e1_tables_figs():
    R = load("E1") + load("E1aod") + load("E1b")
    if not R:
        return
    seeds_by = by(R, "ds", "cfg_rho", "method")
    # Pareto-style figure: cost vs budget
    dss = [d for d in ["ciciot2023", "edgeiiot", "nfton_v3"] if any(k[0] == d for k in seeds_by)]
    fig, axs = plt.subplots(1, len(dss), figsize=(7.16, 2.5), squeeze=False)
    handles = {}
    for j, ds in enumerate(dss):
        ax = axs[0, j]
        rhos = sorted({k[1] for k in seeds_by if k[0] == ds})
        curves = {}
        for m in ORDER:
            xs, ys, lo, hi = [], [], [], []
            for rho in rhos:
                rs = seeds_by.get((ds, rho, m), [])
                if len(rs) < 2:
                    continue
                v = [r["cost"] for r in rs]
                c = boot_ci(v)
                xs.append(rho); ys.append(iqm(v)); lo.append(c[0]); hi.append(c[1])
            if xs:
                curves[m] = (xs, ys, lo, hi)
        # methods an order of magnitude worse than the rest would flatten every other curve: note them instead
        best = min(min(c[1]) for c in curves.values())
        off = [m for m, c in curves.items() if min(c[1]) > 3 * best]
        for m, (xs, ys, lo, hi) in curves.items():
            if m in off:
                continue
            handles[m], = ax.plot(xs, ys, marker=MARK[m], ms=3, lw=1, color=COL[m], label=SHORT[m])
            ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.15, lw=0)
        if off:
            ax.text(0.03, 0.03, "off scale: " + ", ".join(f"{SHORT[m]} ({min(curves[m][1]):.0f})" for m in off),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=6)
        ax.set_xscale("log"); ax.set_xticks(rhos); ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.minorticks_off(); ax.set_xlabel("energy budget $\\rho$")
        ax.set_title(DS_NAME[ds])
        if j == 0:
            ax.set_ylabel("total cost / device / 1000 ticks")
    fig.legend([handles[m] for m in ORDER if m in handles], [SHORT[m] for m in ORDER if m in handles],
               loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, "e1_budget.pdf")); plt.close(fig)

    # Main table at rho = 0.005. Point estimate: IQM over seeds. CI: two-level bootstrap over seeds AND the 32
    # test networks (per-network costs from E1net). Test: two-sided Mann-Whitney U over seed-level costs
    # (AoD-T and the baselines share no training randomness, so pairing by seed index would be meaningless),
    # Holm-corrected over all comparisons in the table.
    net = by([r for r in load("E1net") if "net_cost" in r], "ds", "method")
    def net_matrix(ds, m):
        rs = sorted(net.get((ds, m), []), key=lambda r: r["seed"])
        return np.array([r["net_cost"] for r in rs]) if rs else None
    def iqm_ci_two_level(M, n=3000, seed=0):
        rng = np.random.default_rng(seed); S, K = M.shape
        bs = [iqm(M[rng.integers(0, S, S)][:, rng.integers(0, K, K)].mean(1)) for _ in range(n)]
        return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    lines = []
    tests = []
    for ds in dss:
        aod = [r["cost"] for r in seeds_by.get((ds, 0.005, "AoD-T (ipcw)"), [])]
        A = net_matrix(ds, "AoD-T (ipcw)")
        for m in ORDER:
            rs = seeds_by.get((ds, 0.005, m), [])
            if not rs:
                continue
            row = {"ds": ds, "m": m}
            for k in ["cost", "cdh", "wi", "ttm", "energy_x"]:
                v = [r[k] for r in rs if r[k] == r[k]]
                row[k] = (iqm(v), boot_ci(v))
            M = net_matrix(ds, m)
            if M is not None:
                row["cost"] = (row["cost"][0], iqm_ci_two_level(M))
                row["ci_kind"] = "seeds x networks"
            if m != "AoD-T (ipcw)" and aod:
                pr = unpaired([r["cost"] for r in rs], aod)
                if M is not None and A is not None:
                    d, lo, hi = boot_diff_two_level(M, A)
                    pr.update(diff=d, diff_lo=lo, diff_hi=hi)
                row["test"] = pr; tests.append((ds, m, pr["p"]))
            lines.append(row)
    if tests:
        adj = holm([t[2] for t in tests])
        padj = {(t[0], t[1]): a for t, a in zip(tests, adj)}
    else:
        padj = {}
    aod_iqm = {row["ds"]: row["cost"][0] for row in lines if row["m"] == "AoD-T (ipcw)"}
    out = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
           "Dataset & Method & Total cost & $\\Delta$ vs AoD-T [95\\% CI] & Compromised dev.-time & Wrongful isolation & TTM (ticks) & Energy ($\\times e_0$) \\\\",
           "\\midrule"]
    last = None
    for row in lines:
        dsn = DS_NAME[row["ds"]] if row["ds"] != last else ""
        if row["ds"] != last and last is not None:
            out.append("\\midrule")
        last = row["ds"]
        st = ""
        if row["m"] != "AoD-T (ipcw)" and "test" in row:
            better = row["test"].get("diff", row["cost"][0] - aod_iqm[row["ds"]]) > 0   # method costs more
            s = stars(padj.get((row["ds"], row["m"]), 1.0))
            st = s if better else s.replace("*", "\\dagger")
        name = ("\\textbf{" + SHORT[row["m"]] + "}") if row["m"] == "AoD-T (ipcw)" else SHORT[row["m"]]
        t = row.get("test", {})
        delta = (f"{t['diff']:+.2f}{st} {{\\scriptsize[{t['diff_lo']:+.2f}, {t['diff_hi']:+.2f}]}}" if "diff" in t
                 else ("--" if row["m"] == "AoD-T (ipcw)" else st))
        out.append(f"{dsn} & {name} & {row['cost'][0]:.2f} & {delta} & "
                   f"{row['cdh'][0]:.2f} & {row['wi'][0]:.2f} & {row['ttm'][0]:.1f} & {row['energy_x'][0]:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e1_main.tex"), "w").write("\n".join(out))
    json.dump([dict(ds=r["ds"], m=r["m"], cost=r["cost"][0], cost_ci=r["cost"][1], cdh=r["cdh"][0], wi=r["wi"][0],
                    ttm=r["ttm"][0], energy=r["energy_x"][0], test=r.get("test"), ci_kind=r.get("ci_kind", "seeds"),
                    p_holm=padj.get((r["ds"], r["m"]))) for r in lines],
              open(os.path.join(TAB, "e1_main.json"), "w"), indent=1, default=float)


# --------------------------------------------------------------------------- E2
def e2():
    R = load("E2")
    if not R:
        return
    def cost_of(method, **cfg):
        rs = [r for r in R if r["method"] == method and all(abs(r.get("cfg_" + k, d) - v) < 1e-9
                                                             for k, (v, d) in cfg.items())]
        return {r["seed"]: r for r in rs}
    oracle = {r["seed"]: r for r in R if r["method"] == "AoD-T (oracle)" and "cfg_delay_median" not in r
              and "cfg_stealth_delay" not in r}
    meds, pns = [100.0, 300.0, 1000.0], [0.0, 0.2, 0.4]
    grid = {m: np.full((3, 3), np.nan) for m in ["naive", "ipcw"]}
    cens = np.full((3, 3), np.nan)
    rows = []
    for i, md in enumerate(meds):
        for j, pn in enumerate(pns):
            for mode in ["naive", "ipcw"]:
                rs = [r for r in R if r["method"] == f"AoD-T ({mode})" and abs(r.get("cfg_delay_median", -1) - md) < 1e-9
                      and abs(r.get("cfg_p_never", -1) - pn) < 1e-9]
                if not rs or not oracle:
                    continue
                s = [r["seed"] for r in rs if r["seed"] in oracle]
                rel = [100 * (next(r for r in rs if r["seed"] == q)["cost"] / oracle[q]["cost"] - 1) for q in s]
                grid[mode][i, j] = iqm(rel)
                if mode == "ipcw":
                    cens[i, j] = np.mean([r["censor_rate"] for r in rs])
                rows.append(dict(median=md, p_never=pn, mode=mode, excess=iqm(rel), ci=boot_ci(rel),
                                 cost=iqm([r["cost"] for r in rs]), cdh=iqm([r["cdh"] for r in rs]),
                                 wi=iqm([r["wi"] for r in rs]), censor=float(np.mean([r["censor_rate"] for r in rs]))))
    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
        vmax = np.nanmax([np.nanmax(np.abs(g)) for g in grid.values()])
        for ax, mode, title in zip(axs, ["naive", "ipcw"], ["(a) Naive delayed labels", "(b) IPCW-corrected (ours)"]):
            im = ax.imshow(grid[mode], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            for i in range(3):
                for j in range(3):
                    if not np.isnan(grid[mode][i, j]):
                        ax.text(j, i, f"{grid[mode][i, j]:+.0f}%\n(cens. {cens[i, j]*100:.0f}%)", ha="center",
                                va="center", fontsize=6.5)
            ax.set_xticks(range(3)); ax.set_xticklabels([f"{p:.1f}" for p in pns])
            ax.set_yticks(range(3)); ax.set_yticklabels([f"{m:.0f}" for m in meds])
            ax.set_xlabel("never-discovered share $p_{\\mathrm{nd}}$"); ax.set_ylabel("median discovery delay (ticks)")
            ax.set_title(title)
        fig.colorbar(im, ax=axs, shrink=0.8, label="excess cost vs. oracle labels (%)")
        fig.savefig(os.path.join(FIG, "e2_labels.pdf"), bbox_inches="tight"); plt.close(fig)
    # misspecification / A4 rows
    mis = []
    for name in ["pnever-0.1", "pnever+0.2", "median x0.5", "median x2", "A4 violated"]:
        for mode in ["ipcw", "naive", "oracle"]:
            rs = [r for r in R if r["method"] == f"AoD-T ({mode}, {name})"]
            if rs:
                mis.append(dict(setting=name, mode=mode, cost=iqm([r["cost"] for r in rs]),
                                ci=boot_ci([r["cost"] for r in rs]), n=len(rs)))
    json.dump(dict(grid=rows, misspec=mis, oracle_cost=iqm([r["cost"] for r in oracle.values()]) if oracle else None),
              open(os.path.join(TAB, "e2.json"), "w"), indent=1, default=float)


# --------------------------------------------------------------------------- E3
def e3():
    R = load("E3")
    if not R:
        return
    out = []
    for m in ["AoD-N", "AoD-N (no shaping)", "PPO no escalation", "PPO fixed penalty"]:
        rs = [r for r in R if r["method"] == m]
        if not rs:
            continue
        row = dict(m=m, n=len(rs))
        for k in ["cost", "cdh", "wi", "energy_x", "esc_rate", "struct_agree", "frac_E"]:
            v = [r[k] for r in rs if k in r and r[k] == r[k]]
            row[k] = iqm(v) if v else float("nan")
            row[k + "_ci"] = boot_ci(v) if len(v) > 1 else None
        out.append(row)
    json.dump(out, open(os.path.join(TAB, "e3.json"), "w"), indent=1, default=float)


# --------------------------------------------------------------------------- E4 / E5
def shift_table(exp, base_exp="E1"):
    R = load(exp) + (load("E4lat") if exp == "E4" else [])
    if not R:
        return
    base = {(r["method"], r["seed"]): r for r in R if r.get("shift") in ("none", "N=64 (none)")}
    if not base:
        base = {(r["method"], r["seed"]): r for r in load(base_exp) + load("E1aod") + load("E1b") if r["ds"] == "ciciot2023"
                and abs(r.get("cfg_rho", 0) - 0.005) < 1e-9}
    g = by([r for r in R if "shift" in r], "shift", "method")
    out = []
    for (sh, m), rs in sorted(g.items()):
        rel = [100 * (r["cost"] / base[(m, r["seed"])]["cost"] - 1) for r in rs if (m, r["seed"]) in base]
        out.append(dict(shift=sh, m=m, cost=iqm([r["cost"] for r in rs]), ci=boot_ci([r["cost"] for r in rs]),
                        rel=iqm(rel) if rel else None, energy=iqm([r["energy_x"] for r in rs]),
                        wi=iqm([r["wi"] for r in rs]), cdh=iqm([r["cdh"] for r in rs]),
                        blocked=iqm([r["esc_blocked"] for r in rs])))
    # unseen-family block (trained on excluded pool)
    # (E4u reruns these after the evaluation-view fix; the original E4 unseen-family records are superseded)
    fam = [r for r in load("E4u") if r.get("excl")] if exp == "E4" else [r for r in R if r.get("excl")]
    for m, rs in by(fam, "method").items():
        out.append(dict(shift="unseen families", m=m[0], cost=iqm([r["cost"] for r in rs]),
                        ci=boot_ci([r["cost"] for r in rs]), cdh=iqm([r["cdh"] for r in rs]),
                        wi=iqm([r["wi"] for r in rs]), energy=iqm([r["energy_x"] for r in rs])))
    json.dump(out, open(os.path.join(TAB, f"{exp.lower()}.json"), "w"), indent=1, default=float)
    return out




# --------------------------------------------------------------------------- LaTeX writers (E2-E5)
def _ci(c, nd=2):
    return f"\\scriptsize[{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if c else ""


def write_e2_tex():
    p = os.path.join(TAB, "e2.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    names = {"pnever-0.1": "$\\hat p_{\\mathrm{nd}}$ too low (0.1 vs 0.2)",
             "pnever+0.2": "$\\hat p_{\\mathrm{nd}}$ too high (0.4 vs 0.2)",
             "median x0.5": "$\\hat m$ half the true median", "median x2": "$\\hat m$ twice the true median",
             "A4 violated": "Stealthy attacks found later (A.~\\ref{as:label}(i) violated)"}
    by_set = defaultdict(dict)
    for r in d["misspec"]:
        by_set[r["setting"]][r["mode"]] = r
    L = ["\\begin{tabular}{lrrr}", "\\toprule", "Label-model condition & IPCW & Naive & Oracle\\\\", "\\midrule"]
    if d.get("oracle_cost") is not None:
        L.append(f"Correct label model (reference) & -- & -- & {d['oracle_cost']:.2f}\\\\")
    for k in ["pnever-0.1", "pnever+0.2", "median x0.5", "median x2", "A4 violated"]:
        if k not in by_set:
            continue
        row = by_set[k]
        cell = lambda m: f"{row[m]['cost']:.2f}" if m in row else "--"
        L.append(f"{names[k]} & {cell('ipcw')} & {cell('naive')} & {cell('oracle')}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e2_misspec.tex"), "w").write("\n".join(L))


def write_e3_tex():
    p = os.path.join(TAB, "e3.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    st = os.path.join(TAB, "e3_structure.json")
    srec = json.load(open(st)) if os.path.exists(st) else []
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Method & Cost & Compr. & Wrong. iso. & Energy & Esc.\\,(\\textperthousand)\\\\", "\\midrule"]
    for r in rows:
        if r["m"] == "PPO fixed penalty":      # one untuned penalty; not a fair baseline (see text)
            continue
        L.append(f"{r['m']} & {r['cost']:.2f} & {r['cdh']:.2f} & {r['wi']:.2f} & {r['energy_x']:.2f}$\\times$ & "
                 f"{r['esc_rate']*1000:.2f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e3.tex"), "w").write("\n".join(L))
    if srec:
        b = [s["balanced_recall"] for s in srec]; e = [s["recall_E"] for s in srec]; c = [s["recall_C"] for s in srec]
        json.dump(dict(balanced=float(np.mean(b)), balanced_min=float(np.min(b)), recall_E=float(np.mean(e)),
                       recall_C=float(np.mean(c)), n=len(srec)), open(os.path.join(TAB, "e3_structure_summary.json"), "w"))


SHIFT_ORDER = ["temp x0.5", "temp x2", "stealth q=0.15", "aggressive q=0.5", "faster spread p1x3",
               "verdict latency 5", "verdict latency 10", "drain 25%", "drain 50%", "unseen families"]
SHIFT_NAME = {"temp x0.5": "Under-confident detector (LLR $\\times0.5$)", "temp x2": "Over-confident detector (LLR $\\times2$)",
              "stealth q=0.15": "Stealthier attacker ($q{=}0.15$, model 0.3)", "aggressive q=0.5": "More active attacker ($q{=}0.5$)",
              "faster spread p1x3": "Faster lateral spread ($3\\times p_1$)", "drain 25%": "Escalation drain, 25\\% of devices",
              "drain 50%": "Escalation drain, 50\\% of devices", "unseen families": "Unseen attack families (web exploits)",
              "verdict latency 5": "Verdict arrives 5 ticks late", "verdict latency 10": "Verdict arrives 10 ticks late"}
MAIN = ["AoD-T (ipcw)", "AoD-DP (model-based)", "Belief two-threshold (oracle-tuned)", "Three-zone deferral", "Belief stopping", "Window NP",
        "Static threshold"]


def write_e4_tex():
    p = os.path.join(TAB, "e4.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    cell = {(r["shift"], r["m"]): r for r in rows}
    ms = [m for m in MAIN if any(k[1] == m for k in cell)]
    L = ["\\begin{tabular}{l" + "r" * len(ms) + "}", "\\toprule",
         "Shift & " + " & ".join(SHORT.get(m, m) for m in ms) + "\\\\", "\\midrule"]
    base = [cell.get(("none", m)) for m in ms]
    L.append("No shift & " + " & ".join(f"{b['cost']:.2f}" if b else "--" for b in base) + "\\\\")
    for sh in SHIFT_ORDER:
        vals = [cell.get((sh, m)) for m in ms]
        if not any(vals):
            continue
        best = min(v["cost"] for v in vals if v)
        cells = []
        for v in vals:
            if not v:
                cells.append("--"); continue
            s = f"{v['cost']:.2f}"
            if abs(v["cost"] - best) < 1e-9:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        L.append(SHIFT_NAME[sh] + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e4.tex"), "w").write("\n".join(L))


def write_e5_tex():
    p = os.path.join(TAB, "e5.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    cell = {(r["shift"], r["m"]): r for r in rows}
    ms = [m for m in ["AoD-T (ipcw)", "AoD-DP (model-based)", "Belief two-threshold (oracle-tuned)", "Belief stopping",
                      "Three-zone deferral"] if any(k[1] == m for k in cell)]
    L = ["\\begin{tabular}{l" + "r" * len(ms) + "}", "\\toprule",
         "Setting & " + " & ".join(SHORT.get(m, m) for m in ms) + "\\\\", "\\midrule"]
    for sh, nm in [("N=16", "$N=16$"), ("N=64 (none)", "$N=64$ (default)"), ("N=256", "$N=256$"), ("N=512", "$N=512$"),
                   ("no hazard broadcast", "No hazard broadcast"), ("no congestion price", "No congestion price"),
                   ("no token bucket", "No token bucket")]:
        vals = [cell.get((sh, m)) for m in ms]
        if not any(vals):
            continue
        L.append(nm + " & " + " & ".join(f"{v['cost']:.2f}" + (f" ({v['energy']:.2f}$\\times$)" if sh == "no token bucket" else "")
                                         if v else "--" for v in vals) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e5.tex"), "w").write("\n".join(L))


def fig_decomp(ds="ciciot2023", rhos=(0.001, 0.005)):
    """Where the cost goes: harm from compromised devices, necessary isolation, wrongful isolation."""
    R = load("E1") + load("E1aod") + load("E1b")
    g = by([r for r in R if r["ds"] == ds], "cfg_rho", "method")
    ms = [m for m in ORDER if any(k[1] == m for k in g)] 
    fig, axs = plt.subplots(1, len(rhos), figsize=(7.0, 2.3), sharey=True)
    for ax, rho in zip(np.atleast_1d(axs), rhos):
        xs, harm, need, wrong = [], [], [], []
        for m in ms:
            rs = g.get((rho, m), [])
            if not rs:
                continue
            cdh = iqm([r["cdh"] for r in rs]); wi = iqm([r["wi"] for r in rs]); cost = iqm([r["cost"] for r in rs])
            xs.append(SHORT[m]); harm.append(cdh); wrong.append(0.2 * wi); need.append(max(cost - cdh - 0.2 * wi, 0))
        x = np.arange(len(xs))
        ax.bar(x, harm, color="#c0392b", label="compromised, uncontained")
        ax.bar(x, need, bottom=harm, color="#f2c46b", label="isolation of compromised devices")
        ax.bar(x, wrong, bottom=np.array(harm) + np.array(need), color="#7f7f7f", label="wrongful isolation")
        ax.set_xticks(x); ax.set_xticklabels(xs, rotation=35, ha="right", fontsize=6.5)
        ax.set_title(f"{DS_NAME[ds]}, $\\rho={rho}$")
    np.atleast_1d(axs)[0].set_ylabel("cost per device per 1000 ticks")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e1_decomp.pdf")); plt.close(fig)


def thresholds_summary(ds="ciciot2023", rho=0.005, logh=-3.6):
    """Thresholds AoD-T learned on the network vs the single-device DP optimum, at the typical
    operating hazard (the broadcast hazard stays within 10^-3.7..10^-3.1 in 99.9% of device-ticks)."""
    from .policies import ThresholdPolicy
    R = [r for r in load("E1") + load("E1aod") if r["ds"] == ds and abs(r["cfg_rho"] - rho) < 1e-9]
    aod = [r for r in R if r["method"] == "AoD-T (ipcw)"]
    dpr = [r for r in R if r["method"] == "AoD-DP (model-based)"]
    p = os.path.join(ROOT, "results", "cache", f"dp_{ds}_q0.3_W5_cs0.2_D30.npz")
    if not aod or not os.path.exists(p):
        return
    z = np.load(p)
    h = np.array([[10.0 ** logh]])
    th = [ThresholdPolicy(r["params"]["theta"]).thresholds(h, np.array([[r["params"]["lam"]]])) for r in aod]
    lam_dp = float(np.median([r["params"]["lam"] for r in dpr])) if dpr else float("nan")
    out = dict(logh=logh, lam_aod=float(np.median([r["params"]["lam"] for r in aod])), lam_dp=lam_dp,
               aod_bE=float(np.median([t[0].item() for t in th])), aod_bC=float(np.median([t[1].item() for t in th])),
               aod_bC_iqr=[float(np.quantile([t[1].item() for t in th], q)) for q in (0.25, 0.75)])
    if lam_dp == lam_dp:
        i = np.interp(logh, z["logh"], np.arange(len(z["logh"])))
        j = np.interp(lam_dp, z["price"], np.arange(len(z["price"])))
        bil = lambda M: float(_bilinear(M, i, j))
        out.update(dp_bE=bil(z["bE"]), dp_bC=bil(z["bC"]))
    json.dump(out, open(os.path.join(TAB, "thresholds.json"), "w"), indent=1)
    return out


def _bilinear(M, i, j):
    i0, j0 = int(min(np.floor(i), M.shape[0] - 2)), int(min(np.floor(j), M.shape[1] - 2))
    fi, fj = i - i0, j - j0
    return ((1 - fi) * (1 - fj) * M[i0, j0] + fi * (1 - fj) * M[i0 + 1, j0] + (1 - fi) * fj * M[i0, j0 + 1]
            + fi * fj * M[i0 + 1, j0 + 1])


def run_all():
    print("E0", e0())
    thresholds_summary()
    e1_tables_figs()
    fig_decomp()
    e2(); e3(); shift_table("E4"); shift_table("E5")
    write_e2_tex(); write_e3_tex(); write_e4_tex(); write_e5_tex()


if __name__ == "__main__":
    run_all()
