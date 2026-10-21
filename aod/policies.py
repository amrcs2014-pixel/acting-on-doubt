"""Device policies: AoD (threshold-structured), and the non-learning baselines.

Every policy maps the per-device observation to W (wait), E (escalate) or C (contain).
Beliefs are handled in log-odds `lo`.
"""
from __future__ import annotations

import numpy as np

from .sim import W_, E_, C_


class Policy:
    name = "policy"
    lam = 0.0                 # energy price used by price-aware policies
    uses_congestion = False   # whether the coordinator runs the online congestion price

    def reset(self, B, N, cfg):
        self.B, self.N, self.cfg = B, N, cfg

    def act(self, obs) -> np.ndarray:
        raise NotImplementedError

    def observe_verdict(self, served, llr_v):
        pass


# ----------------------------------------------------------------- AoD family
class ThresholdPolicy(Policy):
    """Two belief thresholds in log-odds, each affine in log10(hazard) and price.

    theta = [eE0, eEh, eEp, gC0, gCh, gCp]:
        beta_E = eE0 + eEh*(log10 h + 3) + eEp*price
        beta_C = beta_E + softplus(gC0 + gCh*(log10 h + 3) + gCp*price)
    With no_escalate=True it is a belief-stopping rule (contain iff lo >= beta_C), i.e. the
    optimal-stopping baseline.

    Optional 7th parameter (token-aware fallback): when the device's token bucket is empty it
    cannot escalate, the device problem reduces to optimal stopping (Thm 2 with E removed), and
    it contains iff lo >= beta_0 = min(beta_E + theta_7, beta_C).
    """

    def __init__(self, theta, lam=0.0, congestion=True, no_escalate=False, name="AoD-T"):
        # theta: (6|7,) shared, or (B,6|7) one parameter vector per environment (for ES batches)
        self.theta = np.asarray(theta, float)
        lam = np.asarray(lam, float)
        self.lam = lam[:, None] if lam.ndim == 1 else lam
        self.uses_congestion, self.no_escalate, self.name = congestion, no_escalate, name

    def _c(self):
        th = self.theta if self.theta.ndim == 2 else self.theta[None, :]
        return [th[:, k][:, None] for k in range(th.shape[1])]

    def thresholds(self, hhat, price):
        c = self._c()
        x = np.log10(np.clip(hhat, 1e-8, 1)) + 3.0
        bE = c[0] + c[1] * x + c[2] * price
        gap = np.logaddexp(0.0, c[3] + c[4] * x + c[5] * price)
        return bE, bE + gap

    def act(self, obs):
        bE, bC = self.thresholds(obs["hhat"], obs["price"])
        lo = obs["lo"]
        if self.no_escalate:
            return np.where(lo >= bC, C_, W_)
        a = np.where(lo >= bC, C_, np.where(lo >= bE, E_, W_))
        c = self._c()
        if len(c) >= 7:                                  # token-aware fallback when E is unavailable
            b0 = np.minimum(bE + c[6], bC)
            empty = obs["tokens"] < 1.0
            a = np.where(empty, np.where(lo >= b0, C_, W_), a)
        return a


class TablePolicy(Policy):
    """AoD-DP: thresholds read from the exact single-device DP table on (log10 h, price)."""

    def __init__(self, table, lam=0.0, congestion=True, name="AoD-DP"):
        self.tab = table              # dict: logh grid, price grid, bE[h,p], bC[h,p] (log-odds)
        self.lam, self.uses_congestion, self.name = lam, congestion, name

    def _interp(self, M, hhat, price):
        lh = np.log10(np.clip(hhat, 1e-8, 1))
        gh, gp = self.tab["logh"], self.tab["price"]
        ih = np.clip((lh - gh[0]) / (gh[1] - gh[0]), 0, len(gh) - 1.000001)
        ip = np.clip((price - gp[0]) / (gp[1] - gp[0]), 0, len(gp) - 1.000001)
        h0, p0 = ih.astype(int), ip.astype(int)
        fh, fp = ih - h0, ip - p0
        return ((1 - fh) * (1 - fp) * M[h0, p0] + fh * (1 - fp) * M[h0 + 1, p0]
                + (1 - fh) * fp * M[h0, p0 + 1] + fh * fp * M[h0 + 1, p0 + 1])

    def act(self, obs):
        bE = self._interp(self.tab["bE"], obs["hhat"], obs["price"])
        bC = self._interp(self.tab["bC"], obs["hhat"], obs["price"])
        lo = obs["lo"]
        a = np.where(lo >= bC, C_, np.where(lo >= bE, E_, W_))
        if "bC_noE" in self.tab:                        # token-aware fallback (optimal stopping)
            lh = np.log10(np.clip(obs["hhat"], 1e-8, 1))
            b0 = np.interp(lh, self.tab["logh"], self.tab["bC_noE"])
            a = np.where(obs["tokens"] < 1.0, np.where(lo >= b0, C_, W_), a)
        return a


# ------------------------------------------------------------------ baselines
class StaticThreshold(Policy):
    """B1 detect-then-respond: contain when one flow's local LLR exceeds tau."""

    def __init__(self, tau, name="Static threshold"):
        self.tau, self.name = tau, name

    def act(self, obs):
        return np.where(obs["llr_loc"] >= self.tau, C_, W_)


class WindowNP(Policy):
    """B3 ARGUS-style: sliding-window evidence (sum of last w local LLRs) vs a
    Neyman-Pearson threshold set on benign calibration traffic; no escalation."""

    def __init__(self, tau, w=10, name="Window NP"):
        self.tau, self.w, self.name = tau, int(w), name      # w arrives as a float when rebuilt from a record

    def reset(self, B, N, cfg):
        super().reset(B, N, cfg)
        self.buf = np.zeros((B, N, self.w)); self.k = 0

    def act(self, obs):
        self.buf[:, :, self.k % self.w] = np.where(obs["online"], obs["llr_loc"], 0.0)
        self.k += 1
        s = self.buf.sum(2)
        a = np.where(s >= self.tau, C_, W_)
        self.buf[a == C_] = 0.0
        return a


class ThreeZone(Policy):
    """B2/B4 cascade / dual-threshold deferral: per-flow zones.
    llr >= u: contain;  l <= llr < u: escalate;  after a verdict, contain if llr+verdict >= v."""

    def __init__(self, l, u, v, name="Three-zone deferral"):
        self.l, self.u, self.v, self.name = l, u, v, name

    def reset(self, B, N, cfg):
        super().reset(B, N, cfg)
        self.pending = np.zeros((B, N), bool); self.last = np.zeros((B, N))

    def act(self, obs):
        x = obs["llr_loc"]
        self.last = x
        a = np.where(self.pending | (x >= self.u), C_, np.where(x >= self.l, E_, W_))
        self.pending[:] = False
        return a

    def observe_verdict(self, served, llr_v):
        self.pending |= served & ((self.last + llr_v) >= self.v)


class Never(Policy):
    name = "No response"

    def act(self, obs):
        return np.zeros(obs["lo"].shape, np.int64)
