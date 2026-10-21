"""RQ3: does the unstructured AoD-N network learn the wait/escalate/contain structure?

Two pieces of evidence, both robust to the ~99.6% share of 'wait' decisions:
 (1) the network's action probabilities on a belief grid, at fixed hazard / price / tokens / local LLR;
 (2) balanced (per-action) recall of the best two-threshold rule fitted to its greedy test decisions.
"""
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aod.detectors import ROOT
from aod.exp import pool_for, TEST_SEEDS
from aod.ppo import Net, NeuralPolicy, feats, DEV
from aod.sim import SimConfig, BatchSim

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
FIG = os.path.join(ROOT, "figures"); TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def load_net(path):
    net = Net().to(DEV); net.load_state_dict(torch.load(path, map_location=DEV)); net.eval(); return net


def probs_on_grid(net, lo, h, price, tokens=5.0, llr=0.0, bucket=5.0):
    obs = dict(lo=lo, hhat=np.full_like(lo, h), price=np.full_like(lo, price), tokens=np.full_like(lo, tokens),
               llr_loc=np.full_like(lo, llr))
    x = torch.from_numpy(feats(obs, bucket)).to(DEV)
    with torch.no_grad():
        logits, _ = net(x)
    return torch.softmax(logits, -1).cpu().numpy()


def balanced_fit(lo, a):
    order = np.argsort(lo); los, as_ = lo[order], a[order]
    n = len(as_)
    cnt = [np.concatenate([[0], np.cumsum(as_ == k)]) for k in range(3)]
    tot = [c[-1] for c in cnt]
    grid = np.quantile(los, np.linspace(0, 1, 400))
    idx = np.searchsorted(los, grid)
    best = (-1, None, None, None)
    for i, e in enumerate(idx):
        for c in idx[i:]:
            rec = [cnt[0][e] / max(tot[0], 1), (cnt[1][c] - cnt[1][e]) / max(tot[1], 1),
                   (cnt[2][n] - cnt[2][c]) / max(tot[2], 1)]
            present = [k for k in range(3) if tot[k] > 0]
            bal = float(np.mean([rec[k] for k in present]))
            if bal > best[0]:
                best = (bal, float(los[min(e, n - 1)]), float(los[min(c, n - 1)]), rec)
    return best


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "results", "models", "AoD-N_ciciot2023_s*.pt")))
    pool = pool_for("ciciot2023")[0]
    cfg = SimConfig(rho=0.005, pool_split="test")
    lo = np.linspace(-8, 4, 241)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    out = []
    for si, p in enumerate(paths):
        net = load_net(p)
        rec = json.loads([l for l in open(os.path.join(ROOT, "results", "raw", "E3.jsonl"))
                          if json.loads(l)["method"] == "AoD-N" and f"_s{json.loads(l)['seed']}.pt" in p][0])
        lam = rec["params"]["lam"]
        for j, h in enumerate([2e-4, 2e-3, 2e-2]):
            P = probs_on_grid(net, lo, h, lam)
            if si == 0:
                axs[j].plot(lo, P[:, 0], color="#7f7f7f", lw=1, label="wait")
                axs[j].plot(lo, P[:, 1], color="#c8902a", lw=1.4, label="escalate")
                axs[j].plot(lo, P[:, 2], color="#c0392b", lw=1.4, label="contain")
            else:
                axs[j].plot(lo, P[:, 1], color="#c8902a", lw=0.6, alpha=0.5)
                axs[j].plot(lo, P[:, 2], color="#c0392b", lw=0.6, alpha=0.5)
            axs[j].set_title(f"hazard $\\hat h={h:g}$"); axs[j].set_xlabel("belief (log-odds)")
        # balanced two-threshold fit on greedy test decisions (8 test networks)
        pol = NeuralPolicy(net, lam=lam, greedy=True, record=True)
        BatchSim(cfg, pool, TEST_SEEDS[:8]).run(pol)
        A = np.stack(pol.buf["a"], 1).reshape(-1); ON = np.stack(pol.buf["online"], 1).reshape(-1)
        LO = np.stack(pol.buf["lo"], 1).reshape(-1)
        A, LO = A[ON], LO[ON]
        bal, bE, bC, rec3 = balanced_fit(LO, A)
        out.append(dict(seed=si, balanced_recall=bal, recall_W=rec3[0], recall_E=rec3[1], recall_C=rec3[2],
                        bE=bE, bC=bC, n_E=int((A == 1).sum()), n_C=int((A == 2).sum()), n=int(len(A))))
        print(out[-1], flush=True)
    axs[0].set_ylabel("action probability"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_structure.pdf")); plt.close(fig)
    json.dump(out, open(os.path.join(TAB, "e3_structure.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
