"""Build the study datasets data/<name>.parquet + data/<name>.json from the original public downloads.

  python scripts/prep_data.py <RAW_DIR> [nfton] [ciciot] [edge]

RAW_DIR must contain
  netflow/NF-ToN-IoT-v3.zip                       (UQ NetFlow v3 datasets)
  ciciot2023/CSV/CSV/<attack type>/*.csv           (CICIoT2023, official per-pcap CSVs)
  edgeiiot/DNN-EdgeIIoT-dataset.csv.zip            (Edge-IIoTset, DNN subset)
See data/README.md for where to download them.

Every parquet has the feature columns, `y` (class label), `family` (finest attack label), `group` (int64 id of a
block of temporally adjacent records, used by the leakage-controlled split) and `binary` (0 benign, 1 attack).
Identifiers (IPs, ports, MACs, timestamps, stream ids, payload strings) are dropped.

You do not need this step to rerun the experiments: results/cache/pool_*.npz already holds the detector score
pools the simulator uses. It is needed only to retrain the detectors (aod/detectors.py with force=True).
"""
import glob
import hashlib
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RNG = np.random.default_rng(20261011)
RAW = None


def save(name, df, feats, note):
    df = df.reset_index(drop=True)
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    df["group"] = df["group"].astype(np.int64)
    df.to_parquet(os.path.join(OUT, name + ".parquet"))
    meta = dict(name=name, features=feats, classes=sorted(df["y"].unique().tolist()),
                families=sorted(df["family"].unique().tolist()), n=len(df),
                class_counts=df["y"].value_counts().to_dict(), n_groups=int(df["group"].nunique()), note=note)
    json.dump(meta, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(name, df.shape, "groups", meta["n_groups"])


def read_zip_csv(path, chunksize=2_000_000):
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as fh:
        for ch in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
            yield ch


# ------------------------------------------------------------------ NetFlow v3 (UQ)
NF_DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "Label", "Attack",
           "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


def prep_nf(name, zipname, cap, bucket_ms=300_000):
    counts = {}
    for ch in read_zip_csv(os.path.join(RAW, "netflow", zipname)):
        for k, v in ch["Attack"].value_counts().items():
            counts[k] = counts.get(k, 0) + v
    frac = {k: min(1.0, 1.3 * cap / v) for k, v in counts.items()}
    parts = []
    for ch in read_zip_csv(os.path.join(RAW, "netflow", zipname)):
        keep = RNG.random(len(ch)) < ch["Attack"].map(frac).values
        parts.append(ch.loc[keep])
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby("Attack", group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=0))
    df["family"] = df["Attack"].str.lower()
    df["y"] = df["family"]
    df["binary"] = (df["Label"] != 0).astype(int)
    df["group"] = (df["FLOW_START_MILLISECONDS"] // bucket_ms).astype(np.int64)
    feats = [c for c in df.columns if c not in NF_DROP + ["family", "y", "binary", "group"]]
    save(name, df, feats, f"class-capped sample (cap {cap}); group = {bucket_ms//60000}-min start-time bucket; "
                          f"IPs/ports/timestamps dropped; full-file class counts {counts}")


# ------------------------------------------------------------------ CICIoT2023 (official CSVs, 39 features)
CIC_CAT = {"Benign_Final": "benign", "DictionaryBruteForce": "bruteforce", "DNS_Spoofing": "spoofing",
           "MITM-ArpSpoofing": "spoofing", "VulnerabilityScan": "recon"}
for c in ["SqlInjection", "XSS", "CommandInjection", "BrowserHijacking", "Uploading_Attack", "Backdoor_Malware"]:
    CIC_CAT[c] = "web"


def cic_category(d):
    if d in CIC_CAT:
        return CIC_CAT[d]
    return d.split("-")[0].lower()  # DDoS-*, DoS-*, Mirai-*, Recon-*


def prep_ciciot(cap=6000, benign_cap=30000, block=200, max_files=3):
    root = os.path.join(RAW, "ciciot2023", "CSV", "CSV")
    parts = []
    for d in sorted(os.listdir(root)):
        fs = sorted(glob.glob(os.path.join(root, d, "*.csv")))
        if not fs:
            continue
        c = benign_cap if d == "Benign_Final" else cap
        pick = fs if len(fs) <= max_files else list(RNG.choice(fs, max_files, replace=False))
        per = int(np.ceil(c / len(pick)))
        for f in pick:
            x = pd.read_csv(f, low_memory=False)
            x["pos"] = np.arange(len(x))
            x = x.sample(min(len(x), per), random_state=0)
            fid = int(hashlib.md5(os.path.basename(f).encode()).hexdigest()[:8], 16)
            x["group"] = fid * 100000 + (x["pos"] // block)
            x["family"] = d.lower()
            parts.append(x)
        print(d, len(fs), "files read", len(pick), flush=True)
    df = pd.concat(parts, ignore_index=True)
    dirs = {k.lower(): k for k in os.listdir(root) if os.path.isdir(os.path.join(root, k))}
    df["y"] = df["family"].map(lambda s: cic_category(dirs[s]))
    df["binary"] = (df["y"] != "benign").astype(int)
    feats = [c for c in df.columns if c not in ["pos", "group", "family", "y", "binary"]]
    save("ciciot2023", df, feats, f"official per-pcap CSVs; <={max_files} files per attack type; cap {cap} per type, "
                                  f"benign {benign_cap}; y = 8 categories, family = 34 labels; group = file x "
                                  f"{block}-window block")


# ------------------------------------------------------------------ Edge-IIoTset (DNN subset)
EDGE_DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri", "http.request.uri.query", "tcp.payload", "tcp.options",
             "dns.qry.name", "mqtt.msg", "icmp.transmit_timestamp", "tcp.srcport", "tcp.dstport", "udp.port",
             "udp.stream", "tcp.ack", "tcp.ack_raw", "tcp.seq", "tcp.checksum", "icmp.checksum", "mbtcp.trans_id",
             # protocol-identity shortcut fields
             "dns.qry.name.len", "mqtt.topic", "mqtt.protoname", "mqtt.conack.flags"]


def prep_edge(cap=15000, block=200):
    path = os.path.join(RAW, "edgeiiot", "DNN-EdgeIIoT-dataset.csv.zip")
    counts = {}
    for ch in read_zip_csv(path, chunksize=500_000):
        for k, v in ch["Attack_type"].value_counts().items():
            counts[k] = counts.get(k, 0) + v
    codes = {k: i for i, k in enumerate(sorted(counts))}
    frac = {k: min(1.0, 1.3 * cap / v) for k, v in counts.items()}
    parts, offset = [], 0
    for ch in read_zip_csv(path, chunksize=500_000):
        ch["pos"] = np.arange(offset, offset + len(ch)); offset += len(ch)
        ch = ch.drop(columns=[c for c in EDGE_DROP if c in ch.columns])
        keep = RNG.random(len(ch)) < ch["Attack_type"].map(frac).values
        parts.append(ch.loc[keep])
    df = pd.concat(parts, ignore_index=True)
    # contiguous blocks of the file (file order = capture order, one segment per attack type)
    df["group"] = df["Attack_type"].map(codes).astype(np.int64) * 10_000_000 + (df["pos"] // block)
    # "0" vs "0.0" placeholders differ by class (a serialisation artefact): normalise object columns
    obj = [c for c in df.columns if df[c].dtype == object and c not in ["Attack_type"]]
    for c in obj:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.99:
            df[c] = num.fillna(0)
        else:
            s = df[c].astype(str).str.strip()
            s = s.where(~s.isin(["0", "0.0", "0.00", "nan", "None", ""]), "0")
            df[c] = pd.factorize(s)[0]
    df = df.groupby("Attack_type", group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=0))
    df["family"] = df["Attack_type"].str.lower()
    df["y"] = df["family"]
    df["binary"] = df["Attack_label"].astype(int)
    feats = [c for c in df.columns if c not in ["Attack_type", "Attack_label", "pos", "group", "family", "y", "binary"]]
    save("edgeiiot", df, feats, f"DNN-EdgeIIoT subset; identifiers + 4 DNS/MQTT identity fields dropped; '0'/'0.0' "
                                f"placeholders unified; cap {cap} per class; group = {block}-row block within an "
                                f"attack segment")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    RAW = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[2:] or ["nfton", "ciciot", "edge"]
    if "nfton" in which:
        prep_nf("nfton_v3", "NF-ToN-IoT-v3.zip", cap=20000)
    if "ciciot" in which:
        prep_ciciot()
    if "edge" in which:
        prep_edge()
