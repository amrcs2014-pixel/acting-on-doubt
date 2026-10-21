import json, numpy as np
from aod.exp import pool_for
from aod import dp
R = [json.loads(l) for l in open('results/raw/E0.jsonl')]
bad = [r for r in R if not r['ok'] and r['price'] > 0]
out = []
for grid in [641, 1281]:
    dp.XG = np.linspace(-16.0, 16.0, grid)
    for r in bad:
        pool = pool_for(r['ds'])[0]
        s = dp.solve(pool, r['q'], r['h'], r['price'], W=r['W'], c_s=r['c_s'])
        ok, bE, bC, sw, hasE = dp.structure(s['act'], dp.XG)
        g = dp.two_threshold_gap(pool, r['q'], r['h'], r['price'], W=r['W'], c_s=r['c_s']) if not ok else {}
        rec = dict(grid=grid, ds=r['ds'], q=r['q'], h=r['h'], price=r['price'], c_s=r['c_s'], W=r['W'], ok=ok,
                   switches=sw, **g)
        out.append(rec); print(rec, flush=True)
json.dump(out, open('results/raw/E0_refine.json', 'w'), indent=1)
