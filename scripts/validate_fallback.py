import json, sys
from aod.exp import task_aodt
seed = int(sys.argv[1])
r = task_aodt("ciciot2023", seed, dict(rho=0.005), "VALID")[0]
print(json.dumps({k: r[k] for k in ["seed", "cost", "cdh", "wi", "esc_rate", "esc_blocked", "params"]}), flush=True)
