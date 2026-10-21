"""Precompute the AoD-DP threshold tables (one per dataset) in parallel."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

from .sim import SimConfig
from .exp import dp_table, DATASETS, pool_for, WEB


def one(ds, excl=()):
    pool = pool_for(ds, excl)[0]
    tab = dp_table(ds, SimConfig(), pool, tag=("_x" + "-".join(excl)) if excl else "")
    return ds, excl, float(tab["ok"].mean()), float(tab["hasE"].mean())


if __name__ == "__main__":
    jobs = [(ds, ()) for ds in DATASETS] + [("ciciot2023", tuple(WEB))]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(lambda a: one(*a), jobs) if False else [f.result() for f in [ex.submit(one, *j) for j in jobs]]:
            print("table", r, flush=True)
