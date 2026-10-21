"""Vectorised trace-driven simulator of an IoT network under attack.

B independent environments x N devices. Each tick every online device emits one
real flow drawn from a dataset's score pool (benign flow if healthy; attack flow
with probability q if compromised). Its weak (device) detector bin is observed
locally. An escalation captures W fresh flows that the strong (gateway) detector
scores. Compromise spreads laterally along a random subnet graph.

Timing within tick t:
  1 emission + belief predict/update   2 policy acts (W/E/C)
  3 escalations served (gateway cap)   4 containment
  5 infection (onset at t+1)           6 costs, isolation clock, tokens
  7 coordinator (every K ticks): hazard broadcast + congestion price
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from .detectors import ScorePool, N_BINS

W_, E_, C_ = 0, 1, 2


@dataclass
class SimConfig:
    N: int = 64
    G: int = 8
    T: int = 1000
    p0: float = 2e-4          # external infection probability per device-tick
    p1: float = 0.01          # lateral infection per active compromised neighbour-tick
    q: float = 0.3            # attack activity of a compromised device
    D_iso: int = 30           # isolation + reflash duration (ticks)
    c_fn: float = 1.0         # harm per compromised, uncontained device-tick
    c_s: float = 0.2          # service loss per isolated device-tick
    W: int = 5                # flows captured per escalation
    verdict_delay: int = 0    # ticks between a served escalation and its verdict (capture + inspection latency)
    cap: float = 0.05         # gateway deep-inspection service rate (escalations per tick)
    cap_depth: float = 3.0    # gateway service burst depth
    K: int = 10               # coordinator period (ticks)
    rho: float = 0.01         # energy budget: token refill = escalations per device-tick
    bucket: float = 5.0       # token-bucket depth (hard cap on bursts)
    e_ratio: float = 100.0    # energy of one escalation / energy of one local inference
    deg_in: float = 4.0       # mean intra-subnet degree
    cross: float = 1.0        # mean cross-subnet degree
    b0: float = 1e-3          # belief after reflash / at start
    mu_eta: float = 0.5       # congestion-price step
    u_target: float = 0.9     # target gateway utilisation
    broadcast_hazard: bool = True   # coordinator sends mean-field hazard (else devices assume p0)
    congestion: bool = True         # coordinator runs the online congestion price
    # the defender's model (may be misspecified)
    q_hat: float | None = None
    p0_hat: float | None = None
    p1_hat: float | None = None
    temp: float = 1.0         # scales the local log-likelihood ratio (miscalibration)
    # label channel
    delay_median: float = 300.0
    delay_sigma: float = 1.0
    p_never: float = 0.2
    grace: int = 500
    stealth_delay: bool = False   # A4 violation: discovery slower for stealthier attacks
    label_crn: bool = False       # draw each episode's discovery delay from a generator seeded by (env seed,
                                  # device, onset): policies compared on the same seeds then share label noise
    # attacker variants
    drain_frac: float = 0.0   # share of benign devices targeted by escalation-drain traffic
    drain_q: float = 0.3      # per-tick probability of a crafted flow at a drain target
    pool_split: str = "test"  # "cal" for training/tuning, "test" for evaluation

    def model(self):
        return (self.q if self.q_hat is None else self.q_hat,
                self.p0 if self.p0_hat is None else self.p0_hat,
                self.p1 if self.p1_hat is None else self.p1_hat)


def llr_tables(pool: ScorePool, q_hat: float):
    """Per-bin log-likelihood ratio of a compromised (activity q_hat) vs healthy device."""
    fw = q_hat * pool.h_weak_att / pool.h_weak_ben + (1 - q_hat)
    fs = q_hat * pool.h_strong_att / pool.h_strong_ben + (1 - q_hat)
    return np.log(fw), np.log(fs)


def make_topology(cfg: SimConfig, rng: np.random.Generator) -> sp.csr_matrix:
    N, G = cfg.N, cfg.G
    per = N // G
    rows, cols = [], []
    for g in range(G):
        base = g * per
        m = int(round(cfg.deg_in * per / 2))
        for _ in range(m):
            a, b = rng.integers(0, per, 2)
            if a != b:
                rows += [base + a, base + b]; cols += [base + b, base + a]
    m = int(round(cfg.cross * N / 2))
    for _ in range(m):
        a, b = rng.integers(0, N, 2)
        if a // per != b // per:
            rows += [a, b]; cols += [b, a]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A.data[:] = 1.0
    return A


@dataclass
class EpisodeResult:
    metrics: dict                      # per-env arrays
    episodes: list                     # infection episodes: (env, dev, onset, end, contained, revealed, delay)
    dwell: np.ndarray | None = None    # (B, T, N) bool, uncontained compromise indicator
    extra: dict = field(default_factory=dict)


class BatchSim:
    def __init__(self, cfg: SimConfig, pool: ScorePool, seeds: list[int]):
        self.cfg, self.pool, self.seeds = cfg, pool, list(seeds)
        self.B = len(seeds)
        q_hat, self.p0h, self.p1h = cfg.model()
        self.q_hat = q_hat
        self.llr_w, self.llr_s = llr_tables(pool, q_hat)
        if cfg.pool_split == "cal":
            self.wb, self.wa, self.sb, self.sa = pool.cweak_ben, pool.cweak_att, pool.cstrong_ben, pool.cstrong_att
        else:
            self.wb, self.wa, self.sb, self.sa = pool.weak_ben, pool.weak_att, pool.strong_ben, pool.strong_att
        # crafted drain flows: attack-looking to the weak detector (top half of attack weak bins)
        thr = np.median(self.llr_w[self.wa])
        self.w_drain = self.wa[self.llr_w[self.wa] >= thr]
        self.gw = np.arange(cfg.N) // (cfg.N // cfg.G)

    # ------------------------------------------------------------------ run
    def run(self, policy, record_dwell: bool = False, record_trace: bool = False) -> EpisodeResult:
        cfg, B, N, T = self.cfg, self.B, self.cfg.N, self.cfg.T
        rngs = [np.random.default_rng(s) for s in self.seeds]
        As = [make_topology(cfg, r) for r in rngs]
        A = sp.block_diag(As, format="csr")                       # (B*N, B*N)
        Alog_nb = A                                                # for mean-field hazard
        drain_target = np.stack([r.random(N) < cfg.drain_frac for r in rngs])

        z = np.zeros((B, N), bool)
        iso = np.zeros((B, N), np.int32)
        lo = np.full((B, N), np.log(cfg.b0 / (1 - cfg.b0)))
        hhat = np.full((B, N), self.p0h)
        tokens = np.full((B, N), cfg.bucket)
        mu = np.zeros((B, cfg.G))
        load = np.zeros((B, cfg.G))
        gtok = np.full((B, cfg.G), cfg.cap_depth)
        onset = np.full((B, N), -1, np.int64)
        benign_cont = np.zeros((B, N), bool)
        episodes = []
        dwell_rec = np.zeros((B, T, N), bool) if record_dwell else None
        m = {k: np.zeros(B) for k in ["dwell", "iso", "wrong_iso", "esc_req", "esc_served", "esc_blocked",
                                      "esc_dropped", "contain", "contain_benign", "infections", "peak_inf"]}
        trace = {"b": [], "act": [], "z": [], "price": [], "hhat": []} if record_trace else None
        policy.reset(B, N, cfg)
        gw = self.gw
        dly = int(cfg.verdict_delay)
        if dly > 0:                                   # ring buffer of verdicts in flight
            v_llr = np.zeros((dly + 1, B, N)); v_due = np.zeros((dly + 1, B, N), bool)
        for t in range(T):
            # columns: 0 activity, 1 local flow, 2 drain, 3 gateway priority, 4 infection, 5.. verdict
            U = np.stack([r.random((N, 5 + 2 * cfg.W)) for r in rngs])  # (B,N,k)
            online = iso == 0
            active = z & online & (U[:, :, 0] < cfg.q)
            # local observation
            idx_b = (U[:, :, 1] * len(self.wb)).astype(int)
            idx_a = (U[:, :, 1] * len(self.wa)).astype(int)
            wbin = np.where(active, self.wa[idx_a], self.wb[idx_b])
            if cfg.drain_frac > 0:
                crafted = drain_target & ~z & online & (U[:, :, 2] < cfg.drain_q)
                idx_d = (U[:, :, 1] * len(self.w_drain)).astype(int)
                wbin = np.where(crafted, self.w_drain[idx_d], wbin)
            llr_loc = cfg.temp * self.llr_w[wbin]
            # belief predict + update (online devices only)
            p = 1.0 / (1.0 + np.exp(-lo))
            p_pred = p + (1 - p) * hhat
            lo_pred = np.log(np.clip(p_pred, 1e-12, 1) / np.clip(1 - p_pred, 1e-12, 1))
            lo = np.where(online, np.clip(lo_pred + llr_loc, -30, 30), lo)
            if dly > 0:                               # verdicts requested dly ticks ago arrive now
                slot = t % (dly + 1)
                arr = v_due[slot] & online
                if arr.any():
                    lo = np.where(arr, np.clip(lo + v_llr[slot], -30, 30), lo)
                    policy.observe_verdict(arr, v_llr[slot])
                v_llr[slot] = 0.0; v_due[slot] = False
            price = policy.lam + mu[:, gw]
            obs = dict(lo=lo, hhat=hhat, price=price, tokens=tokens, online=online, llr_loc=llr_loc,
                       t=t, wbin=wbin)
            act = policy.act(obs)
            act = np.where(online, act, W_)
            # escalation
            want = act == E_
            blocked = want & (tokens < 1.0)
            req = want & ~blocked
            m["esc_blocked"] += blocked.sum(1)
            m["esc_req"] += req.sum(1)
            tokens = tokens - req
            served = np.zeros_like(req)
            if req.any():
                # random priority within gateway; serve up to floor(gateway tokens) per gateway
                per = N // cfg.G
                pri = np.where(req, U[:, :, 3], 2.0).reshape(B, cfg.G, per)
                rank = np.argsort(np.argsort(pri, axis=2, kind="stable"), axis=2, kind="stable")
                reqg = req.reshape(B, cfg.G, per)
                servg = reqg & (rank < np.floor(gtok)[:, :, None])
                served = servg.reshape(B, N)
                gtok -= servg.sum(2)
                load += reqg.sum(2)
                m["esc_served"] += served.sum(1)
                m["esc_dropped"] += (req & ~served).sum(1)
                Uv = U[:, :, 5:]
                act_v = z[:, :, None] & (Uv[:, :, :cfg.W] < cfg.q)
                ib = (Uv[:, :, cfg.W:] * len(self.sb)).astype(int)
                ia = (Uv[:, :, cfg.W:] * len(self.sa)).astype(int)
                sbin = np.where(act_v, self.sa[ia], self.sb[ib])
                llr_v = self.llr_s[sbin].sum(2)
                if dly > 0:
                    slot = (t + dly) % (dly + 1)
                    v_llr[slot] += np.where(served, llr_v, 0.0); v_due[slot] |= served
                else:
                    lo = np.where(served, np.clip(lo + llr_v, -30, 30), lo)
                    policy.observe_verdict(served, llr_v)
            # containment
            cont = act == C_
            if cont.any():
                m["contain"] += cont.sum(1)
                cb = cont & ~z
                m["contain_benign"] += cb.sum(1)
                benign_cont = np.where(cont, ~z, benign_cont)
                ce = np.argwhere(cont & z)
                for (bi, di) in ce:
                    episodes.append((bi, di, onset[bi, di], t, True))
                    onset[bi, di] = -1
                iso = np.where(cont, cfg.D_iso, iso)
            # costs at tick t: state z_t after this tick's actions, BEFORE new infections
            # (new infections have onset t+1, matching the episode records and Eq. (4))
            isoed = iso > 0
            dw = z & ~isoed
            if record_dwell:
                dwell_rec[:, t, :] = dw
            m["dwell"] += dw.sum(1)
            m["iso"] += isoed.sum(1)
            m["wrong_iso"] += (isoed & benign_cont).sum(1)
            # infection (onset t+1)
            n_act = (A @ active.reshape(-1).astype(float)).reshape(B, N)
            haz = 1 - (1 - cfg.p0) * (1 - cfg.p1) ** n_act
            healthy = ~z & (iso == 0)
            new = healthy & (U[:, :, 4] < haz)
            if new.any():
                z = z | new
                onset = np.where(new, t + 1, onset)
                m["infections"] += new.sum(1)
            m["peak_inf"] = np.maximum(m["peak_inf"], z.sum(1) / N)
            # isolation clock / release
            rel = iso == 1
            iso = np.maximum(iso - 1, 0)
            if rel.any():
                z = np.where(rel, False, z)
                benign_cont = np.where(rel, False, benign_cont)
                lo = np.where(rel, np.log(cfg.b0 / (1 - cfg.b0)), lo)
            tokens = np.minimum(cfg.bucket, tokens + cfg.rho)
            gtok = np.minimum(cfg.cap_depth, gtok + cfg.cap)
            # coordinator
            if (t + 1) % cfg.K == 0:
                pb = 1.0 / (1.0 + np.exp(-lo))
                risk = np.where(iso == 0, self.p1h * self.q_hat * pb, 0.0)
                s = (A @ np.log1p(-np.clip(risk, 0, 0.999)).reshape(-1)).reshape(B, N)
                if cfg.broadcast_hazard:
                    hhat = 1 - (1 - self.p0h) * np.exp(s)
                cap_K = cfg.cap * cfg.K * cfg.u_target
                if policy.uses_congestion and cfg.congestion:
                    mu = np.maximum(0.0, mu + cfg.mu_eta * (load - cap_K) / cap_K)
                load[:] = 0
            if record_trace:
                trace["b"].append(1 / (1 + np.exp(-lo[0]))); trace["act"].append(act[0].copy())
                trace["z"].append(z[0].copy()); trace["price"].append(price[0].copy()); trace["hhat"].append(hhat[0].copy())
        # close open infection episodes
        for (bi, di) in np.argwhere(onset >= 0):
            episodes.append((bi, di, onset[bi, di], T, False))
        # label channel: discovery delays
        eps = []
        lab_rngs = [np.random.default_rng(10_000 + s) for s in self.seeds]
        for (bi, di, on, end, cont) in episodes:
            r = np.random.default_rng([10_000 + self.seeds[bi], int(di), int(on)]) if cfg.label_crn else lab_rngs[bi]
            never = r.random() < cfg.p_never
            med = cfg.delay_median * ((0.3 / cfg.q) if cfg.stealth_delay else 1.0)
            d = np.inf if never else float(np.exp(np.log(med) + cfg.delay_sigma * r.standard_normal()))
            revealed = (on + d) <= (T + cfg.grace)
            eps.append((int(bi), int(di), int(on), int(end), bool(cont), bool(revealed), d))
        ttm_sum, n_c, n_all = np.zeros(B), np.zeros(B), np.zeros(B)
        for (bi, di, on, end, cont, rev, d) in eps:
            n_all[bi] += 1
            if cont:
                ttm_sum[bi] += end - on; n_c[bi] += 1
        m["ttm"] = np.where(n_c > 0, ttm_sum / np.maximum(n_c, 1), np.nan)
        m["frac_contained"] = np.where(n_all > 0, n_c / np.maximum(n_all, 1), np.nan)
        m["censor_rate"] = np.array([np.mean([not e[5] for e in eps if e[0] == bi]) if n_all[bi] else np.nan
                                     for bi in range(B)])
        m["esc_rate"] = m["esc_req"] / (N * T)
        m["energy_x"] = 1.0 + cfg.e_ratio * m["esc_rate"]
        m["cost"] = (cfg.c_fn * m["dwell"] + cfg.c_s * m["iso"]) / (N * T) * 1000
        m["cdh"] = m["dwell"] / (N * T) * 1000
        m["wi"] = m["wrong_iso"] / (N * T) * 1000
        res = EpisodeResult(metrics=m, episodes=eps, dwell=dwell_rec)
        if record_trace:
            res.extra["trace"] = {k: np.array(v) for k, v in trace.items()}
        return res


# ---------------------------------------------------------------------- labels
def G_true(cfg: SimConfig, d: np.ndarray, q_eff: float | None = None) -> np.ndarray:
    """P(discovered within d ticks) under the label model (never-discovered mass included)."""
    from scipy.stats import norm
    med = cfg.delay_median * ((0.3 / (q_eff or cfg.q)) if cfg.stealth_delay else 1.0)
    d = np.maximum(d, 1e-9)
    return (1 - cfg.p_never) * norm.cdf((np.log(d) - np.log(med)) / cfg.delay_sigma)


def dwell_estimates(res: EpisodeResult, cfg: SimConfig, B: int, G_hat=None, g_min: float = 0.05,
                    p_never_hat: float | None = None, median_hat: float | None = None) -> dict:
    """Per-env dwell totals under oracle, naive and IPCW-corrected labels."""
    H = cfg.T + cfg.grace
    out = {k: np.zeros(B) for k in ["oracle", "naive", "ipcw"]}
    for (bi, di, on, end, cont, rev, d) in res.episodes:
        L = end - on
        out["oracle"][bi] += L
        if rev:
            out["naive"][bi] += L
            if G_hat is None:
                c2 = replace(cfg, p_never=cfg.p_never if p_never_hat is None else p_never_hat,
                             delay_median=cfg.delay_median if median_hat is None else median_hat,
                             stealth_delay=False)
                g = float(G_true(c2, np.array([H - on]))[0])
            else:
                g = float(G_hat(H - on))
            out["ipcw"][bi] += L / max(g, g_min)
    return out


def weight_tensor(res: EpisodeResult, cfg: SimConfig, B: int, mode: str, g_min: float = 0.05,
                  p_never_hat: float | None = None, median_hat: float | None = None) -> np.ndarray:
    """(B,T,N) per device-tick label weight replacing 1(z=1, uncontained) in the training loss."""
    T, N = cfg.T, cfg.N
    H = T + cfg.grace
    w = np.zeros((B, T, N), np.float32)
    c2 = replace(cfg, p_never=cfg.p_never if p_never_hat is None else p_never_hat,
                 delay_median=cfg.delay_median if median_hat is None else median_hat, stealth_delay=False)
    for (bi, di, on, end, cont, rev, d) in res.episodes:
        if end <= on:
            continue
        if mode == "oracle":
            val = 1.0
        elif mode == "naive":
            val = 1.0 if rev else 0.0
        else:
            val = (1.0 / max(float(G_true(c2, np.array([H - on]))[0]), g_min)) if rev else 0.0
        if val:
            w[bi, on:end, di] = val
    return w
