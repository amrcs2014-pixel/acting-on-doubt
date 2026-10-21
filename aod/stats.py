"""Statistics used in the paper.

* IQM (interquartile mean) and a 95% bootstrap CI over seeds (Agarwal et al., NeurIPS 2021).
* Paired Wilcoxon signed-rank test (method vs AoD, paired by seed), Holm-corrected.
* Effect size: paired rank-biserial correlation, and mean paired difference with bootstrap CI.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < 4:
        return float(np.mean(x))
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(np.mean(x[lo:hi]))


def boot_ci(x, stat=iqm, n=5000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired(a, b, n=5000, seed=0):
    """a, b paired by seed. Returns mean diff (a-b), its CI, Wilcoxon p, rank-biserial r."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    try:
        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    nz = d[d != 0]
    if len(nz):
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1
        rpos, rneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        r = float((rpos - rneg) / ranks.sum())
    else:
        r = 0.0
    return dict(diff=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), p=p, r=r)


def unpaired(a, b):
    """a vs b from independent training/tuning seeds (no shared randomness): two-sided Mann-Whitney U
    and the common-language effect size P(a < b) + P(a = b)/2."""
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return dict(p=1.0, cl=0.5)
    u = mannwhitneyu(a, b, alternative="two-sided")
    return dict(p=float(u.pvalue), cl=float(1 - u.statistic / (len(a) * len(b))))


def boot_two_level(M, n=4000, seed=0, stat=np.mean):
    """95% CI of the mean cost over (seeds x test networks), resampling both seeds and networks.
    M: (seeds, networks) array of per-network costs."""
    M = np.asarray(M, float)
    rng = np.random.default_rng(seed)
    S, K = M.shape
    bs = [stat(M[rng.integers(0, S, S)][:, rng.integers(0, K, K)]) for _ in range(n)]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def boot_diff_two_level(A, B, n=4000, seed=0):
    """95% CI of mean(A) - mean(B); A, B: (seeds, networks) on the SAME test networks (networks resampled
    jointly, seeds independently)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    rng = np.random.default_rng(seed)
    K = A.shape[1]
    out = []
    for _ in range(n):
        k = rng.integers(0, K, K)
        a = A[rng.integers(0, A.shape[0], A.shape[0])][:, k].mean()
        b = B[rng.integers(0, B.shape[0], B.shape[0])][:, k].mean()
        out.append(a - b)
    return float(np.mean(A) - np.mean(B)), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj
