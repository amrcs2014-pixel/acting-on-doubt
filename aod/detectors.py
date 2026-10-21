"""Train a weak device detector and a strong gateway detector on real IoT flows.

Leakage control: flows are split by the dataset's `group` id (capture block) into
train / calibration / test. Detectors are fit on train, probability-calibrated and
their score histograms estimated on calibration, and the simulator draws flows only
from test. The device filter therefore never sees the flows it is evaluated on.

Outputs (per dataset) a ScorePool:
  weak_ben, weak_att    : weak-detector bin index of each benign / attack test flow
  strong_ben, strong_att: strong-detector bin index of the same flows
  fam_att               : attack family of each attack test flow
  hist tables (calibration split) used to build likelihood ratios
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "results", "cache")

N_BINS = 32          # score bins on the logit scale (a 32-entry INT8 table on the MCU)
WEAK_K = 8           # features available to the device detector
BENIGN_NAMES = {"benign", "normal"}


@dataclass
class ScorePool:
    name: str
    bin_edges: np.ndarray        # logit-scale bin edges, len N_BINS+1
    weak_ben: np.ndarray         # bin ids
    weak_att: np.ndarray
    strong_ben: np.ndarray
    strong_att: np.ndarray
    fam_att: np.ndarray          # family string per attack test flow
    # calibration-split flows: used for training/tuning every method (test flows only for evaluation)
    cweak_ben: np.ndarray
    cweak_att: np.ndarray
    cstrong_ben: np.ndarray
    cstrong_att: np.ndarray
    cfam_att: np.ndarray
    # calibration-split histograms (probabilities over bins)
    h_weak_ben: np.ndarray
    h_weak_att: np.ndarray
    h_strong_ben: np.ndarray
    h_strong_att: np.ndarray
    meta: dict


def _features(name: str) -> list[str]:
    with open(os.path.join(DATA, f"{name}.json"), encoding="utf8") as f:
        return json.load(f)["features"]


def _split(df: pd.DataFrame, seed: int = 0):
    g = df["group"].values
    s1 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    tr, rest = next(s1.split(df, groups=g))
    s2 = GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=seed + 1)
    ca_r, te_r = next(s2.split(df.iloc[rest], groups=g[rest]))
    return tr, rest[ca_r], rest[te_r]


def _bin(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return np.clip(np.searchsorted(edges, z, side="right") - 1, 0, N_BINS - 1)


def _hist(b: np.ndarray, eps: float = 0.5) -> np.ndarray:
    c = np.bincount(b, minlength=N_BINS).astype(float) + eps   # Laplace smoothing
    return c / c.sum()


def build_pool(name: str, seed: int = 0, exclude_families: tuple[str, ...] = (), force: bool = False) -> ScorePool:
    """Train detectors and return the score pool. Cached on disk."""
    tag = f"{name}_s{seed}" + ("_x" + "-".join(sorted(exclude_families)) if exclude_families else "")
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"pool_{tag}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        return ScorePool(**{k: (z[k].item() if k in ("meta", "name") else z[k]) for k in z.files})

    df = pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))
    feats = [c for c in _features(name) if c in df.columns]
    X = df[feats].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = df["binary"].values.astype(int)
    fam = df["family"].astype(str).values
    tr, ca, te = _split(df, seed)

    # optional "unseen family" shift: drop families from train+calibration only
    if exclude_families:
        keep = ~np.isin(fam, list(exclude_families))
        tr, ca = tr[keep[tr]], ca[keep[ca]]

    signed_log = FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)))

    # weak features: top-K by strong-model permutation-free proxy (HGB gain via a quick fit)
    probe = HistGradientBoostingClassifier(max_iter=60, random_state=seed).fit(X[tr], y[tr])
    # rank features by drop in train AUC when replaced by its mean (cheap, deterministic)
    base = roc_auc_score(y[ca], probe.predict_proba(X[ca])[:, 1])
    drops = []
    for j in range(X.shape[1]):
        Xc = X[ca].copy(); Xc[:, j] = X[tr][:, j].mean()
        drops.append(base - roc_auc_score(y[ca], probe.predict_proba(Xc)[:, 1]))
    weak_idx = list(np.argsort(drops)[::-1][:WEAK_K])

    weak = make_pipeline(signed_log, StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    weak = CalibratedClassifierCV(weak, method="isotonic", cv=3).fit(X[tr][:, weak_idx], y[tr])
    strong = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed),
        method="isotonic", cv=3).fit(X[tr], y[tr])

    p_w_ca = weak.predict_proba(X[ca][:, weak_idx])[:, 1]
    p_s_ca = strong.predict_proba(X[ca])[:, 1]
    p_w_te = weak.predict_proba(X[te][:, weak_idx])[:, 1]
    p_s_te = strong.predict_proba(X[te])[:, 1]

    edges = np.linspace(-9.0, 9.0, N_BINS + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    bw_ca, bs_ca = _bin(p_w_ca, edges), _bin(p_s_ca, edges)
    bw_te, bs_te = _bin(p_w_te, edges), _bin(p_s_te, edges)
    yc, yt = y[ca], y[te]

    meta = dict(
        dataset=name, seed=seed, n_train=int(len(tr)), n_cal=int(len(ca)), n_test=int(len(te)),
        weak_features=[feats[i] for i in weak_idx], n_features=len(feats),
        weak_auc=float(roc_auc_score(yt, p_w_te)), strong_auc=float(roc_auc_score(yt, p_s_te)),
        weak_ap=float(average_precision_score(yt, p_w_te)), strong_ap=float(average_precision_score(yt, p_s_te)),
        exclude_families=list(exclude_families),
        test_attack_families=sorted(set(fam[te][yt == 1])),
    )
    pool = ScorePool(
        name=name, bin_edges=edges,
        weak_ben=bw_te[yt == 0], weak_att=bw_te[yt == 1],
        strong_ben=bs_te[yt == 0], strong_att=bs_te[yt == 1],
        fam_att=fam[te][yt == 1],
        cweak_ben=bw_ca[yc == 0], cweak_att=bw_ca[yc == 1],
        cstrong_ben=bs_ca[yc == 0], cstrong_att=bs_ca[yc == 1],
        cfam_att=fam[ca][yc == 1],
        h_weak_ben=_hist(bw_ca[yc == 0]), h_weak_att=_hist(bw_ca[yc == 1]),
        h_strong_ben=_hist(bs_ca[yc == 0]), h_strong_att=_hist(bs_ca[yc == 1]),
        meta=meta,
    )
    np.savez_compressed(path, **{k: (np.array(v, dtype=object) if k == "meta" else v) for k, v in asdict(pool).items()})
    return pool


# Propagation-phase attack families: what a compromised IoT device emits while it scans,
# brute-forces and exploits neighbours. Flooding families (DoS/DDoS, Mirai floods) are
# excluded: they belong to the later impact phase and are separable by both detectors
# (per-family AUC 1.000 on every dataset), so they carry no detection dilemma.
PROPAGATION = {
    "ciciot2023": ["recon-hostdiscovery", "recon-osscan", "recon-pingsweep", "recon-portscan",
                   "vulnerabilityscan", "dictionarybruteforce", "dns_spoofing", "mitm-arpspoofing",
                   "commandinjection", "sqlinjection", "xss", "backdoor_malware", "browserhijacking",
                   "uploading_attack"],
    "edgeiiot": ["port_scanning", "fingerprinting", "vulnerability_scanner", "password", "backdoor",
                 "sql_injection", "xss", "uploading", "ransomware", "mitm"],
    "nfton_v3": ["scanning", "backdoor", "injection", "password", "xss", "mitm", "ransomware"],
}


def profile_view(pool: ScorePool, families=None, hist_from: ScorePool | None = None) -> ScorePool:
    """Restrict attack flows (test and calibration) to `families`; recompute attack histograms
    from calibration flows of those families only.

    hist_from: keep that view's attack histograms (the likelihood tables the policies were tuned on)
    instead. Used for unseen families, whose calibration flows were removed: their own histograms
    would be empty, and the deployed filter would not know them anyway."""
    if families is None:
        families = PROPAGATION[pool.name]
    st, sc = np.isin(pool.fam_att, families), np.isin(pool.cfam_att, families)
    d = asdict(pool)
    d.update(weak_att=pool.weak_att[st], strong_att=pool.strong_att[st], fam_att=pool.fam_att[st],
             cweak_att=pool.cweak_att[sc], cstrong_att=pool.cstrong_att[sc], cfam_att=pool.cfam_att[sc],
             h_weak_att=_hist(pool.cweak_att[sc]) if hist_from is None else hist_from.h_weak_att,
             h_strong_att=_hist(pool.cstrong_att[sc]) if hist_from is None else hist_from.h_strong_att)
    meta = dict(pool.meta); meta["profile_families"] = list(families)
    meta["profile_n_test_att"], meta["profile_n_cal_att"] = int(st.sum()), int(sc.sum())
    d["meta"] = meta
    return ScorePool(**d)


if __name__ == "__main__":
    for ds in ["ciciot2023", "edgeiiot", "nfton_v3"]:
        p = build_pool(ds, force=True)
        print(ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.meta.items() if k != "test_attack_families"})
        print("   benign/attack test flows:", len(p.weak_ben), len(p.weak_att))
