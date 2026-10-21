"""AoD-N: a neural device policy (parameters shared by all devices) trained with
Lagrangian PPO on per-device rewards built only from deployment-available signals:

  r_t = - c_fn * w_t        (w: delayed-label weight; oracle / naive / IPCW)
        - c_s  * 1[isolated]
        - price_t * 1[escalate]          (Lagrangian energy price, dual ascent)
        + gamma*Phi(b_{t+1}) - Phi(b_t)  (belief-potential shaping, Phi = -kappa*c_fn*b)

Flags turn it into the deep-RL baselines: allow_E=False (uncertainty-as-input responder,
no escalation) and constraint="penalty" (fixed energy penalty instead of a constraint).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from .sim import SimConfig, BatchSim, weight_tensor, W_, E_, C_
from .policies import Policy

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NF = 5


def feats(obs, bucket):
    f = np.stack([np.clip(obs["lo"], -15, 15) / 5.0,
                  np.log10(np.clip(obs["hhat"], 1e-8, 1)) + 3.0,
                  np.clip(obs["price"], 0, 10),
                  obs["tokens"] / bucket,
                  np.clip(obs["llr_loc"], -5, 10) / 3.0], -1)
    return f.astype(np.float32)


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(NF, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 3))
        self.v = nn.Sequential(nn.Linear(NF, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        with torch.no_grad():                      # start near "always wait"
            self.pi[-1].bias.copy_(torch.tensor([3.0, -1.0, -3.0]))
            self.pi[-1].weight.mul_(0.1)

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


class NeuralPolicy(Policy):
    def __init__(self, net: Net, lam=0.0, allow_E=True, greedy=False, record=False, congestion=True, name="AoD-N"):
        self.net, self.lam, self.allow_E, self.greedy, self.record = net, lam, allow_E, greedy, record
        self.uses_congestion, self.name = congestion, name

    def reset(self, B, N, cfg):
        super().reset(B, N, cfg)
        self.buf = {k: [] for k in ["x", "a", "logp", "v", "online", "lo", "price"]}

    @torch.no_grad()
    def act(self, obs):
        x = feats(obs, self.cfg.bucket)
        xt = torch.from_numpy(x).to(DEV)
        logits, v = self.net(xt)
        mask = torch.zeros_like(logits)
        if not self.allow_E:
            mask[..., E_] = -1e9
        else:
            mask[..., E_] = torch.from_numpy(np.where(obs["tokens"] < 1.0, -1e9, 0.0).astype(np.float32)).to(DEV)
        logits = logits + mask
        if self.greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        a_np = a.cpu().numpy()
        if self.record:
            self.buf["x"].append(x); self.buf["a"].append(a_np); self.buf["logp"].append(logp.cpu().numpy())
            self.buf["v"].append(v.cpu().numpy()); self.buf["online"].append(obs["online"].copy())
            self.buf["lo"].append(obs["lo"].copy()); self.buf["price"].append(np.broadcast_to(obs["price"], obs["lo"].shape).copy())
        return a_np


def train_ppo(pool, cfg: SimConfig, label_mode="ipcw", iters=120, n_env=8, seed=0, allow_E=True,
              constraint="lagrangian", lam0=0.5, lam_fix=0.5, dual_lr=0.3, kappa=5.0, shaping=True,
              gamma=0.99, gae=0.95, epochs=4, mb=32768, lr=3e-4, clip=0.2, ent=0.01, verbose=False,
              lab_kw=None):
    lab_kw = lab_kw or {}
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cfg = replace(cfg, pool_split="cal")
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lam = lam0 if constraint == "lagrangian" else lam_fix
    hist = []
    for it in range(iters):
        seeds = [int(s) for s in rng.integers(0, 2**31 - 1, n_env)]
        pol = NeuralPolicy(net, lam=lam, allow_E=allow_E, record=True)
        sim = BatchSim(cfg, pool, seeds)
        res = sim.run(pol, record_dwell=True)
        B, T, N = n_env, cfg.T, cfg.N
        w = weight_tensor(res, cfg, B, label_mode, **lab_kw)                    # (B,T,N)
        X = np.stack(pol.buf["x"], 1)                                            # (B,T,N,F)
        A = np.stack(pol.buf["a"], 1); LP = np.stack(pol.buf["logp"], 1); V = np.stack(pol.buf["v"], 1)
        ON = np.stack(pol.buf["online"], 1); LO = np.stack(pol.buf["lo"], 1); PR = np.stack(pol.buf["price"], 1)
        iso = ~ON
        b = 1.0 / (1.0 + np.exp(-LO))
        r = -cfg.c_fn * w * ON - cfg.c_s * iso - PR * ((A == E_) & ON)
        # isolation that starts at t (action C) is paid from t: the online C tick is charged too
        r = r - cfg.c_s * ((A == C_) & ON)
        if shaping:
            phi = -kappa * cfg.c_fn * b * ON
            phi_next = np.concatenate([phi[:, 1:], np.zeros_like(phi[:, :1])], 1)
            r = r + gamma * phi_next - phi
        r = r.astype(np.float32)
        # GAE per device sequence
        adv = np.zeros_like(r); last = np.zeros((B, N), np.float32)
        Vn = np.concatenate([V[:, 1:], V[:, -1:]], 1)          # truncation: bootstrap with the last value
        for t in range(T - 1, -1, -1):
            delta = r[:, t] + gamma * Vn[:, t] - V[:, t]
            last = delta + gamma * gae * last
            adv[:, t] = last
        ret = adv + V
        # value is trained on every tick (it bootstraps through isolation); the policy only on decisions
        Xf = torch.from_numpy(X.reshape(-1, NF)).to(DEV)
        Af = torch.from_numpy(A.reshape(-1)).long().to(DEV)
        LPf = torch.from_numpy(LP.reshape(-1)).to(DEV)
        ONf = torch.from_numpy(ON.reshape(-1)).to(DEV)
        ADf = torch.from_numpy(adv.reshape(-1)).to(DEV)
        RTf = torch.from_numpy(ret.reshape(-1)).to(DEV)
        TKf = torch.from_numpy((X[..., 3].reshape(-1) * cfg.bucket < 1.0)).to(DEV)
        ADf = (ADf - ADf[ONf].mean()) / (ADf[ONf].std() + 1e-8)
        n = Xf.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, mb):
                j = perm[i:i + mb]
                logits, v = net(Xf[j])
                m = torch.zeros_like(logits)
                if not allow_E:
                    m[:, E_] = -1e9
                else:
                    m[:, E_] = torch.where(TKf[j], torch.tensor(-1e9, device=DEV), torch.tensor(0.0, device=DEV))
                logits = logits + m
                logp_all = torch.log_softmax(logits, -1)
                lp = logp_all.gather(-1, Af[j].unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(lp - LPf[j])
                s1, s2 = ratio * ADf[j], torch.clamp(ratio, 1 - clip, 1 + clip) * ADf[j]
                on = ONf[j].float(); non = on.sum().clamp(min=1.0)
                entropy = (-(torch.exp(logp_all) * logp_all).nan_to_num().sum(-1) * on).sum() / non
                pol_loss = -(torch.min(s1, s2) * on).sum() / non
                loss = pol_loss + 0.5 * ((v - RTf[j]) ** 2).mean() - ent * entropy
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        rate = float(res.metrics["esc_rate"].mean())
        if constraint == "lagrangian" and allow_E:
            lam = max(0.0, lam + dual_lr * (rate / cfg.rho - 1.0))
        hist.append(dict(it=it, cost=float(res.metrics["cost"].mean()), rate=rate, lam=lam))
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  ppo it {it:3d} cost {res.metrics['cost'].mean():6.3f} cdh {res.metrics['cdh'].mean():6.3f} "
                  f"wi {res.metrics['wi'].mean():5.2f} rate {rate:.4f} lam {lam:.3f}", flush=True)
    return net, lam, hist
